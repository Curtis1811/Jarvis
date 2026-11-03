"""Local speech to text with faster-whisper. Free, runs on your GPU or CPU."""

from __future__ import annotations

import numpy as np


class FasterWhisperSTT:
    def __init__(
        self,
        model: str = "small.en",
        device: str = "auto",
        compute_type: str = "default",
        language: str | None = "en",
    ) -> None:
        from faster_whisper import WhisperModel

        # "auto" uses CUDA when an NVIDIA GPU and its libraries are present.
        self._model = WhisperModel(model, device=device, compute_type=compute_type)
        self._language = language

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio, language=self._language, beam_size=5, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
