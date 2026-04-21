import asyncio
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from groq import Groq

from config import ASR_TIMEOUT, SAMPLE_RATE

load_dotenv()
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
        executor: ThreadPoolExecutor,
        language: str = "en",
        timeout: float = ASR_TIMEOUT,
    ):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.executor = executor
        self.language = language
        self.timeout = timeout

    async def transcribe(self, audio: np.ndarray, segment_id: str = "") -> ASRResult:
        loop = asyncio.get_event_loop()
        try:
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
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        buffer.name = "audio.wav"

        response = self.client.audio.transcriptions.create(
            file=buffer,
            model="whisper-large-v3",
            language=self.language,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            temperature=0.0,
        )
        return self._build_result(response, segment_id)

    def _build_result(self, response, segment_id: str) -> ASRResult:
        text = response.text.strip()

        words: List[WordTimestamp] = []
        if hasattr(response, "words") and response.words:
            for w in response.words:
                # Groq SDK 0.9 returns word objects as plain dicts
                if isinstance(w, dict):
                    words.append(WordTimestamp(
                        word=w.get("word", ""),
                        start=w.get("start", 0.0),
                        end=w.get("end", 0.0),
                        probability=w.get("probability", 1.0),
                    ))
                else:
                    words.append(WordTimestamp(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=getattr(w, "probability", 1.0),
                    ))

        duration = words[-1].end if words else 0.0
        confidence = 1.0

        return ASRResult(
            text=text,
            words=words,
            confidence=confidence,
            duration=duration,
            is_timeout=False,
            segment_id=segment_id,
        )
