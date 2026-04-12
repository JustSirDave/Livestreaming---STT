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
            tensor = self._preprocess(frame)
            confidence = self.model(tensor, SAMPLE_RATE).item()
            return confidence >= self.threshold
        except Exception as e:
            logger.warning("VAD classify failed, returning False: %s", e)
            return False

    def _preprocess(self, frame: bytes) -> torch.Tensor:
        audio = np.frombuffer(frame, dtype=np.int16) / 32768.0
        return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # shape (1, 320)

    def get_confidence(self, frame: bytes) -> float:
        tensor = self._preprocess(frame)
        return self.model(tensor, SAMPLE_RATE).item()
