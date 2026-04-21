import logging
import numpy as np
import torch

from config import SAMPLE_RATE, VAD_THRESHOLD

logger = logging.getLogger(__name__)


class VadEngine:
    def __init__(self, model, threshold: float = VAD_THRESHOLD):
        self.model = model
        self.threshold = threshold

    def classify(self, frame: bytes) -> bool:
        try:
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio ** 2)))
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            confidence = self.model(tensor, SAMPLE_RATE).item()
            is_speech = confidence >= self.threshold
            print(f"[VAD] rms={rms:.5f} conf={confidence:.3f} thresh={self.threshold} speech={is_speech}", flush=True)
            logger.debug("VAD rms=%.5f confidence=%.3f threshold=%.2f speech=%s", rms, confidence, self.threshold, is_speech)
            return is_speech
        except Exception as e:
            print(f"[VAD] EXCEPTION: {e}", flush=True)
            logger.warning("VAD classify failed, returning False: %s", e)
            return False

    def reset(self) -> None:
        """Reset Silero's internal LSTM state. Call between utterances to prevent state accumulation."""
        try:
            self.model.reset_states()
        except Exception as e:
            logger.warning("VAD reset failed: %s", e)

    def _preprocess(self, frame: bytes) -> torch.Tensor:
        audio = np.frombuffer(frame, dtype=np.int16) / 32768.0
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # shape (1, 512)

    def get_confidence(self, frame: bytes) -> float:
        tensor = self._preprocess(frame)
        return self.model(tensor, SAMPLE_RATE).item()
