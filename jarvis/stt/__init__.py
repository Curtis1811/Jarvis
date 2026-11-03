"""Speech to text. Each provider turns 16 kHz float32 audio into a string."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SpeechToText(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


def make_stt(provider: str, options: dict[str, Any]) -> SpeechToText:
    if provider == "faster_whisper":
        from .faster_whisper_stt import FasterWhisperSTT

        return FasterWhisperSTT(**options)
    if provider == "deepgram":
        from .deepgram_stt import DeepgramSTT

        return DeepgramSTT(**options)
    raise ValueError(f"Unknown stt_provider '{provider}'. Use faster_whisper or deepgram.")
