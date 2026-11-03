"""Local voice with Kokoro (82M params). Free, quick even on CPU, good quality."""

from __future__ import annotations

import numpy as np


class KokoroTTS:
    RATE = 24_000

    def __init__(self, voice: str = "bm_george", speed: float = 1.0, lang_code: str | None = None) -> None:
        from kokoro import KPipeline

        # Voice names start with a language letter: a = American, b = British.
        self._pipeline = KPipeline(lang_code=lang_code or voice[0])
        self._voice = voice
        self._speed = speed

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        parts = [
            np.asarray(audio, dtype=np.float32)
            for _gs, _ps, audio in self._pipeline(text, voice=self._voice, speed=self._speed)
            if audio is not None
        ]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, self.RATE
