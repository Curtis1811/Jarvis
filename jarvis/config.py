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

from .personalities import BUILTIN as _BUILTIN_PERSONALITIES

# Jarvis's instructions. Other personalities live in personalities.py.
DEFAULT_SYSTEM_PROMPT = _BUILTIN_PERSONALITIES["jarvis"]["system_prompt"]


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

    # agent (Claude Agent SDK, can do things) | chat (Claude API, talk only)
    # | openai_compatible (DeepSeek and others, talk only)
    brain_provider: str = "agent"
    brain: dict[str, Any] = field(default_factory=dict)

    tts_provider: str = "kokoro"  # kokoro | csm | elevenlabs | openai | system | print
    tts: dict[str, Any] = field(default_factory=dict)

    # Who Jarvis is: jarvis | friday | karen, or one you add under `personalities`.
    # Each personality has its own instructions and voice.
    personality: str = "jarvis"
    personalities: dict[str, Any] = field(default_factory=dict)

    # Set this to replace the personality's instructions entirely.
    system_prompt: str = ""

    def apply_personality(self) -> dict[str, Any]:
        """Use the chosen personality's instructions and voice. Returns the personality."""
        from .personalities import get

        persona = get(self.personality, self.personalities)
        if not (self.system_prompt or "").strip():
            self.system_prompt = persona["system_prompt"]
        tts = dict(self.tts or {})
        for provider, options in (persona.get("tts") or {}).items():
            tts[provider] = {**(tts.get(provider) or {}), **(options or {})}
        self.tts = tts
        return persona

    def section(self, name: str, provider: str) -> dict[str, Any]:
        """Options for one provider, e.g. settings.section("tts", "elevenlabs")."""
        return dict((getattr(self, name) or {}).get(provider) or {})


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    load_dotenv()
    # A blank line like "ANTHROPIC_API_KEY=" in .env would otherwise hide your
    # Claude subscription login from the agent brain.
    blank = [k for k, v in os.environ.items() if k.endswith(("_API_KEY", "_AUTH_TOKEN")) and not v.strip()]
    for name in blank:
        del os.environ[name]
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
