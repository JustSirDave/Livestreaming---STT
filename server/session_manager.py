import asyncio
import enum
import logging
import time
from dataclasses import dataclass
from typing import List
from uuid import uuid4

import numpy as np

from config import (
    INTERIM_INTERVAL_FRAMES,
    MIN_SEGMENT_DURATION,
    SAMPLE_RATE,
    VAD_FLUSH_FRAMES,
    VAD_PAUSE_FRAMES,
    WS_QUEUE_MAX,
)
from audio_processor import AudioProcessingError

logger = logging.getLogger(__name__)


class VadState(enum.Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    PAUSED = "paused"


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
        self.partial_text: str = ""
        self.segments: List[Segment] = []
        self._speech_frame_count: int = 0
        self._interim_running: bool = False

    async def pipeline_loop(self, vad, asr, post, audio_proc) -> None:
        while True:
            frame = await self.audio_queue.get()
            if frame is None:
                break

            # write into ring buffer (circular)
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
            action = self._handle_vad_result(is_speech)

            if action == "SPEECH_START":
                self.speech_frames.append(frame)
                self._speech_frame_count = 1
                await self.transcript_queue.put({"type": "speech_start"})
            elif action == "ACCUMULATE":
                self.speech_frames.append(frame)
                self._speech_frame_count += 1
                if self._speech_frame_count % INTERIM_INTERVAL_FRAMES == 0 and not self._interim_running:
                    snapshot = list(self.speech_frames)
                    self._interim_running = True
                    asyncio.create_task(
                        self._transcribe_interim(snapshot, asr, post, audio_proc)
                    )
            elif action == "FLUSH":
                # Snapshot and clear frames immediately so pipeline keeps consuming
                # while ASR runs in the background (Whisper takes 2–5s on CPU).
                snapshot = list(self.speech_frames)
                self.speech_frames = []
                self._speech_frame_count = 0
                self._interim_running = False
                asyncio.create_task(
                    self._flush_and_transcribe(snapshot, self.segment_start_time, asr, post, audio_proc)
                )
            elif action == "PAUSE":
                await self.transcript_queue.put({"type": "pause"})
            elif action == "RESUME":
                await self.transcript_queue.put({"type": "resume"})

    def _handle_vad_result(self, is_speech: bool) -> str:
        state = self.vad_state

        if state == VadState.SILENCE:
            if is_speech:
                self.vad_state = VadState.SPEECH
                self.silence_frame_count = 0
                self.segment_start_time = time.time()
                return "SPEECH_START"
            else:
                self.silence_frame_count += 1
                if self.silence_frame_count >= VAD_PAUSE_FRAMES:
                    self.vad_state = VadState.PAUSED
                    return "PAUSE"
                return "CONTINUE"

        elif state == VadState.SPEECH:
            if is_speech:
                return "ACCUMULATE"
            else:
                self.silence_frame_count += 1
                if self.silence_frame_count >= VAD_FLUSH_FRAMES:
                    self.vad_state = VadState.SILENCE
                    self.silence_frame_count = 0
                    return "FLUSH"
                return "ACCUMULATE"

        elif state == VadState.PAUSED:
            if is_speech:
                self.vad_state = VadState.SPEECH
                self.silence_frame_count = 0
                self.segment_start_time = time.time()
                return "RESUME"
            else:
                return "CONTINUE"

        return "CONTINUE"

    async def _flush_and_transcribe(self, frames_snapshot: List[bytes], segment_start: float, asr, post, audio_proc) -> None:
        if not frames_snapshot:
            return

        # Compute duration from snapshot length
        total_samples = sum(len(f) // 2 for f in frames_snapshot)  # int16 = 2 bytes
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
            return

        message = post.process(result, type="final")
        segment = Segment(
            segment_id=result.segment_id,
            text=message.text,
            start=segment_start,
            end=time.time(),
            confidence=result.confidence,
        )
        self.add_final_segment(segment)
        await self.transcript_queue.put(message.to_dict())

    async def _transcribe_interim(self, frames_snapshot: List[bytes], asr, post, audio_proc) -> None:
        try:
            total_samples = sum(len(f) // 2 for f in frames_snapshot)
            if total_samples / SAMPLE_RATE < MIN_SEGMENT_DURATION:
                return
            try:
                processed = audio_proc.process(frames_snapshot, SAMPLE_RATE)
            except AudioProcessingError:
                return
            result = await asr.transcribe(processed, segment_id="interim")
            if result.is_timeout or not result.text.strip():
                return
            message = post.process(result, type="interim")
            await self.transcript_queue.put(message.to_dict())
        finally:
            self._interim_running = False

    def add_final_segment(self, segment: Segment) -> None:
        self.segments.append(segment)
        self.partial_text = ""

    def teardown(self) -> None:
        self.audio_queue.put_nowait(None)
