import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

import numpy as np

from config import (
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
        self.speech_frames_snapshot: List[bytes] = []
        self.segment_start_time: float = 0.0
        self.partial_text: str = ""
        self.segments: List[Segment] = []

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
                await self.transcript_queue.put({"type": "speech_start"})
            elif action == "ACCUMULATE":
                self.speech_frames.append(frame)
            elif action == "FLUSH":
                await self._flush_and_transcribe(asr, post, audio_proc)
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

    async def _flush_and_transcribe(self, asr, post, audio_proc) -> None:
        frames_snapshot = list(self.speech_frames)
        audio = self._flush_segment()
        duration = len(audio) / SAMPLE_RATE
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
            start=self.segment_start_time,
            end=time.time(),
            confidence=result.confidence,
        )
        self.add_final_segment(segment)
        await self.transcript_queue.put(message.to_dict())

    def _flush_segment(self) -> np.ndarray:
        if not self.speech_frames:
            return np.array([], dtype=np.int16)
        audio = np.concatenate([
            np.frombuffer(f, dtype=np.int16) for f in self.speech_frames
        ])
        self.speech_frames = []
        return audio

    def add_final_segment(self, segment: Segment) -> None:
        self.segments.append(segment)
        self.partial_text = ""

    def teardown(self) -> None:
        self.audio_queue.put_nowait(None)
