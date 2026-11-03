"""Cloud voice with OpenAI's speech API (paid, cheap, steerable tone)."""

from __future__ import annotations

import httpx
import numpy as np

from ..audio import pcm16_to_float
from ..config import require_env


class OpenAITTS:
    RATE = 24_000  # the "pcm" format is 24 kHz 16-bit mono

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "onyx",
        instructions: str = "Speak like a calm, witty British butler.",
    ) -> None:
        self._key = require_env("OPENAI_API_KEY")
        self._body = {"model": model, "voice": voice, "instructions": instructions, "response_format": "pcm"}
        self._http = httpx.Client(timeout=60)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        resp = self._http.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {self._key}"},
            json={**self._body, "input": text},
        )
        resp.raise_for_status()
        return pcm16_to_float(resp.content), self.RATE
