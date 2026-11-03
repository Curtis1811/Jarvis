"""Cloud speech to text with Deepgram (paid, very fast and accurate)."""

from __future__ import annotations

import httpx
import numpy as np

from ..audio import MIC_RATE, float_to_wav_bytes
from ..config import require_env


class DeepgramSTT:
    def __init__(self, model: str = "nova-3", language: str = "en") -> None:
        self._key = require_env("DEEPGRAM_API_KEY")
        self._params = {"model": model, "language": language, "smart_format": "true"}
        self._http = httpx.Client(timeout=30)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        resp = self._http.post(
            "https://api.deepgram.com/v1/listen",
            params=self._params,
            headers={"Authorization": f"Token {self._key}", "Content-Type": "audio/wav"},
            content=float_to_wav_bytes(audio, MIC_RATE),
        )
        resp.raise_for_status()
        alts = resp.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"].strip() if alts else ""
