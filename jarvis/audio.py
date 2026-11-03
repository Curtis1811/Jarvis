"""Microphone input and speaker output.

Audio inside Jarvis is always mono float32 numpy arrays. Mic audio is 16 kHz,
which is what Whisper, Deepgram and openWakeWord expect.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator

import numpy as np

MIC_RATE = 16_000
FRAME = 1280  # 80 ms at 16 kHz, the frame size openWakeWord wants


def _sd():
    # Imported lazily so text mode and tests work without PortAudio.
    import sounddevice as sd

    return sd


class Microphone:
    """Streams 80 ms frames from the default input device."""

    def __init__(self, rate: int = MIC_RATE, frame: int = FRAME) -> None:
        self.rate = rate
        self.frame = frame
        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None

    def __enter__(self) -> Microphone:
        sd = _sd()

        def callback(indata, frames, time_info, status):
            self._q.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def flush(self) -> None:
        while not self._q.empty():
            self._q.get_nowait()

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            yield self._q.get()


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0


def record_utterance(
    frames: Iterator[np.ndarray],
    *,
    rate: int = MIC_RATE,
    speech_threshold: float = 0.015,
    silence_seconds: float = 0.9,
    max_seconds: float = 30.0,
    start_timeout: float = 6.0,
) -> np.ndarray:
    """Collects one spoken request: waits for speech, stops after a pause.

    Returns an empty array if nobody spoke within ``start_timeout`` seconds.
    """
    chunks: list[np.ndarray] = []
    pre_roll: list[np.ndarray] = []  # keeps the start of the first word
    heard_speech = False
    quiet_for = 0.0
    waited = 0.0
    total = 0.0

    for frame in frames:
        seconds = len(frame) / rate
        loud = rms(frame) >= speech_threshold

        if not heard_speech:
            waited += seconds
            pre_roll = (pre_roll + [frame])[-4:]
            if loud:
                heard_speech = True
                chunks.extend(pre_roll)
            elif waited >= start_timeout:
                return np.zeros(0, dtype=np.float32)
            continue

        chunks.append(frame)
        total += seconds
        quiet_for = 0.0 if loud else quiet_for + seconds
        if quiet_for >= silence_seconds - 1e-6 or total >= max_seconds:
            break

    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


class Speaker:
    """Plays audio clips one after another on a background thread.

    Clips are queued as they are made, so Jarvis starts talking after the first
    sentence instead of waiting for the whole answer.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self._idle = threading.Event()
        self._idle.set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def play(self, audio: np.ndarray, rate: int) -> None:
        if audio.size:
            self._idle.clear()
            self._q.put((audio.astype(np.float32), rate))

    def stop(self) -> None:
        """Drops everything queued and cuts off the current clip."""
        while not self._q.empty():
            self._q.get_nowait()
        self._stop.set()
        try:
            _sd().stop()
        except Exception:
            pass

    def wait(self) -> None:
        while not self._idle.wait(0.1):
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            audio, rate = item
            self._stop.clear()
            try:
                sd = _sd()
                sd.play(audio, rate)
                duration = len(audio) / rate
                end = time.monotonic() + duration + 0.2
                while time.monotonic() < end and not self._stop.is_set():
                    time.sleep(0.05)
                sd.stop()
            except Exception as exc:  # keep the assistant alive if a device hiccups
                print(f"[speaker] {exc}")
            if self._q.empty():
                self._idle.set()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_wav_bytes(audio: np.ndarray, rate: int) -> bytes:
    import io
    import wave

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buf.getvalue()
