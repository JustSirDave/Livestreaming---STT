import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import torch
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
from faster_whisper import WhisperModel

from asr_model import ASRModel
from audio_processor import AudioProcessor
from post_processor import PostProcessor
from session_manager import SessionManager
from vad_engine import VadEngine

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

        logger.info("Loading Whisper model...")
        whisper = WhisperModel("base", device="cpu", compute_type="int8")
        app.state.whisper = whisper
        logger.info("Whisper model loaded.")

        app.state.executor = ThreadPoolExecutor(max_workers=2)

        app.state.vad = VadEngine(model=app.state.vad_model)
        app.state.asr = ASRModel(model=app.state.whisper, executor=app.state.executor)
        app.state.post = PostProcessor()
        app.state.audio_proc = AudioProcessor()

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
    session = SessionManager()
    await websocket.send_text(
        json.dumps({"type": "session", "session_id": session.session_id})
    )

    pipeline_task = asyncio.create_task(
        session.pipeline_loop(
            app.state.vad,
            app.state.asr,
            app.state.post,
            app.state.audio_proc,
        )
    )

    receive_task = asyncio.create_task(receive_loop(websocket, session))
    send_task = asyncio.create_task(send_loop(websocket, session))

    try:
        await asyncio.gather(receive_task, send_task)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("WebSocket error: %s: %s", type(e).__name__, e)
    finally:
        session.teardown()
        pipeline_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass  # already closed by client


async def receive_loop(websocket: WebSocket, session: SessionManager):
    while True:
        msg = await websocket.receive()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect(msg.get("code", 1000))
        data = msg.get("bytes")
        if not data:
            continue
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
        await websocket.send_text(json.dumps(message))


# Serve frontend — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory="../client", html=True), name="client")
