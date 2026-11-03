"""Cloud voice with ElevenLabs (paid). The most natural-sounding option."""

from __future__ import annotations

import httpx
import numpy as np

from ..audio import pcm16_to_float
from ..config import require_env


class ElevenLabsTTS:
    RATE = 24_000

    def __init__(
        self,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",  # "George", a calm British voice
        model_id: str = "eleven_flash_v2_5",  # lowest latency; eleven_multilingual_v2 sounds richer
    ) -> None:
        self._key = require_env("ELEVENLABS_API_KEY")
        self._url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        self._model_id = model_id
        self._http = httpx.Client(timeout=60)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        resp = self._http.post(
            self._url,
            params={"output_format": f"pcm_{self.RATE}"},
            headers={"xi-api-key": self._key},
            json={"text": text, "model_id": self._model_id},
        )
        resp.raise_for_status()
        return pcm16_to_float(resp.content), self.RATE
