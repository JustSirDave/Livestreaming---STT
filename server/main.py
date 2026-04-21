import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect

load_dotenv()

from asr_model import ASRModel
from audio_processor import AudioProcessor
from post_processor import PostProcessor
from session_manager import SessionManager
from vad_engine import VadEngine

_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server.log")
_file_handler = logging.FileHandler(_log_path, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[logging.StreamHandler(), _file_handler],
    force=True,   # override any handlers uvicorn already set up
)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
async def startup():
    try:
        logger.info("Loading Silero VAD model...")
        vad_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        app.state.vad_model = vad_model
        logger.info("Silero VAD loaded.")

        executor = ThreadPoolExecutor(max_workers=2)
        app.state.executor = executor

        app.state.asr      = ASRModel(executor=executor)
        app.state.post     = PostProcessor()
        app.state.audio_proc = AudioProcessor()

        logger.info("Groq ASR client initialised.")
        logger.info("All components initialised — server ready.")
    except Exception:
        logger.exception("Startup failed — refusing to start.")
        raise


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] NEW CONNECTION from {websocket.client}", flush=True)
    logger.info("WebSocket connected from %s", websocket.client)
    session = SessionManager()
    await websocket.send_text(
        json.dumps({"type": "session", "session_id": session.session_id})
    )

    # VAD is stateful (Silero LSTM) — create a fresh instance per session
    # so state never leaks between connections or reconnects
    vad = VadEngine(model=app.state.vad_model)
    vad.reset()

    pipeline_task = asyncio.create_task(
        session.pipeline_loop(
            vad,
            app.state.asr,
            app.state.asr,
            app.state.post,
            app.state.audio_proc,
        )
    )

    receive_task = asyncio.create_task(receive_loop(websocket, session))
    send_task = asyncio.create_task(send_loop(websocket, session))
    keepalive_task = asyncio.create_task(keepalive_loop(websocket))

    try:
        # pipeline_task included so a crash there surfaces and tears down the session
        await asyncio.gather(pipeline_task, receive_task, send_task, keepalive_task)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s: %s", type(e).__name__, e)
    finally:
        session.teardown()
        for t in (pipeline_task, receive_task, send_task, keepalive_task):
            t.cancel()
        await asyncio.gather(pipeline_task, receive_task, send_task, keepalive_task, return_exceptions=True)
        try:
            await websocket.close()
        except Exception:
            pass  # already closed by client


async def receive_loop(websocket: WebSocket, session: SessionManager):
    frame_count = 0
    while True:
        msg = await websocket.receive()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        data = msg.get("bytes")
        if not data:
            continue
        frame_count += 1
        if frame_count <= 5 or frame_count % 100 == 0:
            print(f"[FRAME] #{frame_count} received, {len(data)} bytes", flush=True)
            logger.debug("Frame #%d received, %d bytes", frame_count, len(data))
        if session.audio_queue.full():
            try:
                session.audio_queue.get_nowait()
            except Exception:
                pass
            await websocket.send_text(
                json.dumps({"type": "warning", "message": "audio queue full, frame dropped"})
            )
        await session.audio_queue.put(data)


async def send_loop(websocket: WebSocket, session: SessionManager):
    while True:
        message = await session.transcript_queue.get()
        if message is None:
            break
        await websocket.send_text(json.dumps(message))


async def keepalive_loop(websocket: WebSocket):
    while True:
        await asyncio.sleep(20)
        await websocket.send_text(json.dumps({"type": "ping"}))


# Serve frontend — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory="../client", html=True), name="client")
