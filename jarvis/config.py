"""Settings for Jarvis, read from config.yaml and .env.

Every stage (listen, think, speak) has a provider you can switch in config.yaml.
API keys come from environment variables or a .env file, never from config.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a voice assistant. Everything you write is read aloud by a "
    "text-to-speech voice, so answer in short, natural spoken sentences. Do not use "
    "markdown, bullet points, tables, code blocks or emoji. Spell out symbols. When a "
    "task needs real work, do it with your tools, then say briefly what you did and "
    "what you found. If something would be long, give the short version and offer more."
)


@dataclass
class Settings:
    # How a turn starts: "wakeword" (say "Hey Jarvis"), "push_to_talk" (press Enter)
    # or "text" (type, for testing without a microphone).
    input_mode: str = "push_to_talk"
    wakeword_model: str = "hey_jarvis"
    wakeword_threshold: float = 0.5

    # Listening: end of speech is detected when the mic stays quiet this long.
    silence_seconds: float = 0.9
    max_utterance_seconds: float = 30.0
    # RMS level (0-1) above which the mic counts as speech. Raise it in a noisy room.
    speech_threshold: float = 0.015

    stt_provider: str = "faster_whisper"  # faster_whisper | deepgram
    stt: dict[str, Any] = field(default_factory=dict)

    brain_provider: str = "agent"  # agent (Claude Agent SDK, can do things) | chat (talk only)
    brain: dict[str, Any] = field(default_factory=dict)

    tts_provider: str = "kokoro"  # kokoro | csm | elevenlabs | openai | system | print
    tts: dict[str, Any] = field(default_factory=dict)

    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def section(self, name: str, provider: str) -> dict[str, Any]:
        """Options for one provider, e.g. settings.section("tts", "elevenlabs")."""
        return dict((getattr(self, name) or {}).get(provider) or {})


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    load_dotenv()
    config_path = Path(path or os.environ.get("JARVIS_CONFIG", "config.yaml"))
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    settings = Settings()
    for key, value in data.items():
        if not hasattr(settings, key):
            raise ValueError(f"Unknown setting '{key}' in {config_path}")
        setattr(settings, key, value)
    return settings


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to your .env file (see .env.example)."
        )
    return value
