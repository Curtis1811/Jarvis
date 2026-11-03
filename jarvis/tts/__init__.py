"""Text to speech. Each provider turns one sentence into (audio, sample_rate)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> tuple[np.ndarray, int]: ...


def make_tts(provider: str, options: dict[str, Any]) -> TextToSpeech:
    if provider == "kokoro":
        from .kokoro_tts import KokoroTTS

        return KokoroTTS(**options)
    if provider == "csm":
        from .csm_tts import SesameCSMTTS

        return SesameCSMTTS(**options)
    if provider == "elevenlabs":
        from .elevenlabs_tts import ElevenLabsTTS

        return ElevenLabsTTS(**options)
    if provider == "openai":
        from .openai_tts import OpenAITTS

        return OpenAITTS(**options)
    if provider == "system":
        from .system_tts import SystemTTS

        return SystemTTS(**options)
    if provider == "print":
        return PrintTTS()
    if provider == "dessa":
        raise NotImplementedError(
            "Dessa (Block) has no public API or download yet; it is waitlist-only and "
            "speech-to-speech, so it cannot read text. Pick another tts_provider for now."
        )
    raise ValueError(
        f"Unknown tts_provider '{provider}'. "
        "Use kokoro, csm, elevenlabs, openai, system or print."
    )


class PrintTTS:
    """Silent provider for testing: prints instead of speaking."""

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        print(f"🔊 {text}")
        return np.zeros(0, dtype=np.float32), 24_000
