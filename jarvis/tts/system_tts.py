"""Built-in Windows voice (SAPI) through pyttsx3. Robotic, but needs no setup."""

from __future__ import annotations

import os
import tempfile
import wave

import numpy as np


class SystemTTS:
    def __init__(self, rate: int = 185) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            self._engine.save_to_file(text, path)
            self._engine.runAndWait()
            with wave.open(path, "rb") as wav:
                rate = wav.getframerate()
                raw = wav.readframes(wav.getnframes())
                channels = wav.getnchannels()
            audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            return audio, rate
        finally:
            os.remove(path)
