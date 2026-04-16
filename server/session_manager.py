import asyncio
import enum
import logging
import time
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

import numpy as np

from config import (
    INTERIM_INTERVAL_SEC,
    INTERIM_TAIL_FRAMES,
    MIN_SEGMENT_DURATION,
    SAMPLE_RATE,
    VAD_FLUSH_FRAMES,
    WS_QUEUE_MAX,
)
from audio_processor import AudioProcessingError

logger = logging.getLogger(__name__)


class VadState(enum.Enum):
    SILENCE = "silence"
    SPEECH  = "speech"


@dataclass(frozen=True)
class Segment:
    segment_id: str
    text: str
    start: float
    end: float
    confidence: float


class SessionManager:
    def __init__(self):
        self.session_id: str = str(uuid4())
        self.audio_queue: asyncio.Queue = asyncio.Queue(maxsize=WS_QUEUE_MAX)
        self.transcript_queue: asyncio.Queue = asyncio.Queue()
        self.ring_buffer: np.ndarray = np.zeros(SAMPLE_RATE * 30, dtype=np.int16)
        self._ring_pos: int = 0
        self.vad_state: VadState = VadState.SILENCE
        self.silence_frame_count: int = 0
        self.speech_frames: List[bytes] = []
        self.segment_start_time: float = 0.0
        self.segments: List[Segment] = []
        self._interim_task: Optional[asyncio.Task] = None

    async def pipeline_loop(self, vad, asr, post, audio_proc) -> None:
        try:
            while True:
                frame = await self.audio_queue.get()
                if frame is None:
                    break

                # Write into ring buffer (circular)
                arr = np.frombuffer(frame, dtype=np.int16)
                n = len(arr)
                end_pos = self._ring_pos + n
                if end_pos <= len(self.ring_buffer):
                    self.ring_buffer[self._ring_pos:end_pos] = arr
                else:
                    first = len(self.ring_buffer) - self._ring_pos
                    self.ring_buffer[self._ring_pos:] = arr[:first]
                    self.ring_buffer[:n - first] = arr[first:]
                self._ring_pos = end_pos % len(self.ring_buffer)

                is_speech = vad.classify(frame)
                action = self._handle_vad(is_speech)

                if action == "SPEECH_START":
                    self.speech_frames.append(frame)
                    await self.transcript_queue.put({"type": "interim", "text": "…",
                                                      "segment_id": "", "start": 0,
                                                      "end": 0, "confidence": 0, "words": []})
                    self._start_interim(asr, post, audio_proc)

                elif action == "ACCUMULATE":
                    self.speech_frames.append(frame)

                elif action == "FLUSH":
                    self._cancel_interim()
                    vad.reset()
                    snapshot = list(self.speech_frames)
                    self.speech_frames = []
                    asyncio.create_task(
                        self._flush_and_transcribe(snapshot, self.segment_start_time, asr, post, audio_proc)
                    )

        except Exception:
            logger.exception("pipeline_loop crashed — session %s", self.session_id)
            raise

    def _handle_vad(self, is_speech: bool) -> str:
        if self.vad_state == VadState.SILENCE:
            if is_speech:
                self.vad_state = VadState.SPEECH
                self.silence_frame_count = 0
                self.segment_start_time = time.time()
                return "SPEECH_START"

        elif self.vad_state == VadState.SPEECH:
            if is_speech:
                self.silence_frame_count = 0
                return "ACCUMULATE"
            else:
                self.silence_frame_count += 1
                if self.silence_frame_count >= VAD_FLUSH_FRAMES:
                    self.vad_state = VadState.SILENCE
                    self.silence_frame_count = 0
                    return "FLUSH"
                return "ACCUMULATE"

        return "CONTINUE"

    def _start_interim(self, asr, post, audio_proc) -> None:
        self._cancel_interim()
        self._interim_task = asyncio.create_task(
            self._interim_worker(asr, post, audio_proc)
        )

    def _cancel_interim(self) -> None:
        if self._interim_task and not self._interim_task.done():
            self._interim_task.cancel()
        self._interim_task = None

    async def _interim_worker(self, asr, post, audio_proc) -> None:
        """Fires every INTERIM_INTERVAL_SEC while speech is active.
        Transcribes the last ~2s of audio and sends an interim result."""
        try:
            while True:
                await asyncio.sleep(INTERIM_INTERVAL_SEC)

                if not self.speech_frames:
                    continue

                tail = list(self.speech_frames[-INTERIM_TAIL_FRAMES:])
                total_samples = sum(len(f) // 2 for f in tail)
                if total_samples / SAMPLE_RATE < MIN_SEGMENT_DURATION:
                    continue

                try:
                    processed = audio_proc.process(tail, SAMPLE_RATE)
                except AudioProcessingError:
                    continue

                result = await asr.transcribe(processed, segment_id="interim")
                if result.is_timeout or not result.text.strip():
                    continue

                message = post.process(result, type="interim")
                logger.info("[interim] %s", message.text)
                await self.transcript_queue.put(message.to_dict())

        except asyncio.CancelledError:
            pass

    async def _flush_and_transcribe(self, frames_snapshot: List[bytes], segment_start: float, asr, post, audio_proc) -> None:
        try:
            if not frames_snapshot:
                return

            total_samples = sum(len(f) // 2 for f in frames_snapshot)
            duration = total_samples / SAMPLE_RATE
            if duration < MIN_SEGMENT_DURATION:
                return

            try:
                processed = audio_proc.process(frames_snapshot, SAMPLE_RATE)
            except AudioProcessingError as e:
                logger.error("AudioProcessingError during flush: %s", e)
                await self.transcript_queue.put({"type": "error", "message": str(e)})
                return

            result = await asr.transcribe(processed, segment_id=str(uuid4()))
            if result.is_timeout or not result.text.strip():
                logger.info("[final] discarded — empty/timeout (%.2fs)", duration)
                return

            message = post.process(result, type="final")
            segment = Segment(
                segment_id=result.segment_id,
                text=message.text,
                start=segment_start,
                end=time.time(),
                confidence=result.confidence,
            )
            self.segments.append(segment)
            logger.info("[final] %s", message.text)
            await self.transcript_queue.put(message.to_dict())

        except Exception:
            logger.exception("_flush_and_transcribe crashed")

    def teardown(self) -> None:
        self._cancel_interim()
        self.audio_queue.put_nowait(None)
        self.transcript_queue.put_nowait(None)
