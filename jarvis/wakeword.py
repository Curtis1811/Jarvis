"""Always-on wake word detection with openWakeWord (free, runs on CPU).

openWakeWord ships a ready-made "hey_jarvis" model, so no training is needed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


class WakeWord:
    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5) -> None:
        import openwakeword
        from openwakeword.model import Model

        openwakeword.utils.download_models([model])
        self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self.threshold = threshold

    def wait(self, frames: Iterator[np.ndarray]) -> None:
        """Blocks until the wake word is heard."""
        for frame in frames:
            pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16)
            scores = self._model.predict(pcm)
            if max(scores.values(), default=0.0) >= self.threshold:
                self._model.reset()
                return
