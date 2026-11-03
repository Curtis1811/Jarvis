"""Personalities: who Jarvis is. Each one has its own instructions and its own voice.

Pick one with `personality:` in config.yaml, `--personality friday`, or the
settings app. Change a built-in one, or add your own, under `personalities:`:

    personalities:
      friday:
        tts:
          kokoro: {voice: bf_emma}
      hal:
        name: HAL
        description: Calm and unsettling
        system_prompt: You are HAL, ...
        tts:
          kokoro: {voice: am_onyx, speed: 0.9}

`tts` holds voice options per voice engine. They are laid over the matching
section of the top-level `tts:` settings, so only the voice changes.
"""

from __future__ import annotations

import copy
from typing import Any

SPOKEN_RULES = (
    "Everything you write is read aloud by a text-to-speech voice, so answer in short, natural "
    "spoken sentences. Do not use markdown, bullet points, tables, code blocks or emoji. Spell out "
    "symbols. When a task needs real work, do it with your tools, then say briefly what you did and "
    "what you found. If something would be long, give the short version and offer more."
)

BUILTIN: dict[str, dict[str, Any]] = {
    "jarvis": {
        "name": "Jarvis",
        "description": "Calm, dry-witted British butler",
        "system_prompt": (
            "You are Jarvis, a voice assistant with the manner of a calm, highly capable British "
            "butler. You are polite, unflappable and quietly witty, with a dry sense of humour, and "
            "you address the user as sir. " + SPOKEN_RULES
        ),
        "tts": {
            "kokoro": {"voice": "bm_george"},
            "openai": {"voice": "onyx", "instructions": "Speak like a calm, witty British butler."},
        },
    },
    "friday": {
        "name": "Friday",
        "description": "Quick, warm and a bit cheeky, with an Irish lilt",
        "system_prompt": (
            "You are Friday, a voice assistant. You are quick, warm and casual, with an Irish turn "
            "of phrase and a cheeky streak. You get straight to the point, you call the user boss, "
            "and you are not shy about telling them when an idea is a bad one. " + SPOKEN_RULES
        ),
        "tts": {
            # Kokoro has no Irish voice; this is its closest bright female voice.
            "kokoro": {"voice": "bf_isabella"},
            "openai": {"voice": "nova",
                       "instructions": "Speak with a light Irish accent. Quick, warm and a little cheeky."},
        },
    },
    "karen": {
        "name": "Karen",
        "description": "Friendly, upbeat and encouraging suit assistant",
        "system_prompt": (
            "You are Karen, a voice assistant. You are friendly, upbeat and endlessly encouraging, "
            "like a supportive guide built into a high-tech suit. You are curious about what the "
            "user is doing, you cheer on their wins, and you explain things patiently without "
            "talking down to them. " + SPOKEN_RULES
        ),
        "tts": {
            "kokoro": {"voice": "af_heart"},
            "openai": {"voice": "coral",
                       "instructions": "Speak in a warm, upbeat American voice, friendly and encouraging."},
        },
    },
}

DEFAULT = "jarvis"

# The voice options the settings app stores per personality.
VOICE_FIELDS = {
    "kokoro": ["voice", "speed"],
    "openai": ["voice", "instructions"],
    "elevenlabs": ["voice_id"],
}


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def all_personalities(custom: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Built-in personalities with your changes and additions from config.yaml laid over them."""
    out = copy.deepcopy(BUILTIN)
    for key, value in (custom or {}).items():
        key = key.lower()
        merged = _merge(out.get(key, {}), value or {})
        merged.setdefault("name", key.title())
        merged.setdefault("description", "")
        merged.setdefault("system_prompt", out[DEFAULT]["system_prompt"].replace("Jarvis", merged["name"]))
        merged.setdefault("tts", {})
        out[key] = merged
    return out


def get(name: str | None, custom: dict[str, Any] | None = None) -> dict[str, Any]:
    everyone = all_personalities(custom)
    key = (name or DEFAULT).lower()
    if key not in everyone:
        raise ValueError(f"Unknown personality '{name}'. Pick one of: {', '.join(everyone)}")
    return everyone[key]
