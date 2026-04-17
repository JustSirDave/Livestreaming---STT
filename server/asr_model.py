import asyncio
import logging
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import List

import numpy as np

from config import ASR_BEAM_SIZE, ASR_TIMEOUT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class ASRResult:
    text: str
    words: List[WordTimestamp]
    confidence: float
    duration: float
    is_timeout: bool
    segment_id: str = ""


def _empty_result(segment_id: str = "", is_timeout: bool = False) -> ASRResult:
    return ASRResult(
        text="",
        words=[],
        confidence=0.0,
        duration=0.0,
        is_timeout=is_timeout,
        segment_id=segment_id,
    )


class ASRModel:
    def __init__(
        self,
        model,
        executor: Executor,
        language: str = "en",
        beam_size: int = ASR_BEAM_SIZE,
        word_timestamps: bool = True,
        timeout: float = ASR_TIMEOUT,
    ):
        self.model = model
        self.executor = executor
        self.language = language
        self.beam_size = beam_size
        self.word_timestamps = word_timestamps
        self.timeout = timeout
        # CTranslate2 (faster-whisper) is not thread-safe for concurrent inference
        # on the same model instance — serialize all calls with a semaphore
        self._sem = asyncio.Semaphore(1)

    async def transcribe(self, audio: np.ndarray, segment_id: str = "") -> ASRResult:
        loop = asyncio.get_event_loop()
        try:
            async with self._sem:
                future = loop.run_in_executor(
                    self.executor, self._run_inference, audio, segment_id
                )
                return await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError:
            return _empty_result(segment_id=segment_id, is_timeout=True)
        except Exception as e:
            logger.error("ASR transcription error for segment %s: %s", segment_id, e)
            return _empty_result(segment_id=segment_id, is_timeout=False)

    def _run_inference(self, audio: np.ndarray, segment_id: str) -> ASRResult:
        segments_gen, info = self.model.transcribe(
            audio,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            language=self.language,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            temperature=0.0,
            log_prob_threshold=-1.0,
        )
        segments = list(segments_gen)
        return self._build_result(segments, info, segment_id)

    def _build_result(self, segments: list, info, segment_id: str) -> ASRResult:
        text = " ".join(seg.text for seg in segments).strip()

        words: List[WordTimestamp] = []
        for seg in segments:
            for w in (seg.words or []):
                words.append(WordTimestamp(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                ))

        no_speech_prob = getattr(info, "no_speech_prob", 0.0)
        confidence = 1.0 - no_speech_prob
        duration = getattr(info, "duration", 0.0)

        return ASRResult(
            text=text,
            words=words,
            confidence=confidence,
            duration=duration,
            is_timeout=False,
            segment_id=segment_id,
        )
