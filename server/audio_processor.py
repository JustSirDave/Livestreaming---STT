import numpy as np
from scipy.signal import resample
from typing import List

from config import SAMPLE_RATE, MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION


class AudioProcessingError(Exception):
    pass


class AudioProcessor:
    def process(self, frames: List[bytes], input_rate: int) -> np.ndarray:
        audio = self._assemble(frames)
        audio = self._to_float32(audio)
        if input_rate != SAMPLE_RATE:
            audio = self._resample(audio, input_rate)
        audio = self._normalise(audio)
        self._validate(audio)
        return audio

    def _assemble(self, frames: List[bytes]) -> np.ndarray:
        return np.concatenate([np.frombuffer(f, dtype=np.int16) for f in frames])

    def _to_float32(self, audio: np.ndarray) -> np.ndarray:
        return (audio / 32768.0).astype(np.float32)

    def _resample(self, audio: np.ndarray, input_rate: int) -> np.ndarray:
        target_length = int(len(audio) * SAMPLE_RATE / input_rate)
        return resample(audio, target_length).astype(np.float32)

    def _normalise(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        if peak < 0.001:
            return audio
        return (audio / peak).astype(np.float32)

    def _validate(self, audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            raise AudioProcessingError(f"dtype must be float32, got {audio.dtype}")
        if audio.ndim != 1:
            raise AudioProcessingError(f"audio must be 1D, got ndim={audio.ndim}")
        duration = len(audio) / SAMPLE_RATE
        if not (MIN_SEGMENT_DURATION <= duration <= MAX_SEGMENT_DURATION):
            raise AudioProcessingError(
                f"duration {duration:.3f}s out of range [{MIN_SEGMENT_DURATION}, {MAX_SEGMENT_DURATION}]"
            )
