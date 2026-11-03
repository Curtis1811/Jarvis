"""Turns streamed text into speakable sentences.

Claude's reply arrives a few words at a time. Waiting for the whole answer would
add seconds of silence, so this cuts the stream into sentences and speaks each one
as soon as it is complete. It also strips markdown the voice would read literally.
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_MIN_CHARS = 25
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_CODE_NOTE = "I've put the code on screen."  # avoid tiny clips like "Sure." that sound choppy


def clean_for_speech(text: str) -> str:
    text = _CODE_BLOCK.sub(f" {_CODE_NOTE} ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"https?://\S+", "the link", text)
    text = re.sub(r"^\s{0,3}(#{1,6}|[-*+]|\d+\.)\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~#>|]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        self._buf += chunk
        out: list[str] = []
        while True:
            # Hold back while inside a code block so it is summarised, not read out.
            if self._buf.count("```") % 2 == 1:
                return out
            self._buf = _CODE_BLOCK.sub("\n" + _CODE_NOTE + "\n", self._buf)
            match = _SENTENCE_END.search(self._buf)
            if not match:
                return out
            end = match.end()
            candidate = self._buf[:end]
            if len(candidate.strip()) < _MIN_CHARS and "\n" not in match.group(0):
                nxt = _SENTENCE_END.search(self._buf, end)
                if not nxt:
                    return out
                end = nxt.end()
                candidate = self._buf[:end]
            self._buf = self._buf[end:]
            spoken = clean_for_speech(candidate)
            if spoken:
                out.append(spoken)

    def flush(self) -> list[str]:
        spoken = clean_for_speech(self._buf)
        self._buf = ""
        return [spoken] if spoken else []
