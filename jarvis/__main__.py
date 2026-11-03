"""Run Jarvis: `python -m jarvis` (see README.md for setup).

Each turn: wait for you (wake word, Enter key, or typed text) -> listen ->
speech to text -> Claude -> speak the reply sentence by sentence.
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import threading
import time

from .brain import make_brain
from .config import Settings, load_settings
from .speech import SentenceSplitter
from .stt import make_stt
from .tts import make_tts

EXIT_WORDS = {"exit", "quit", "goodbye", "goodbye jarvis", "shut down"}


class Voice:
    """Makes audio for sentences on a worker thread and queues it on the speaker,
    so the next sentence is being generated while the current one plays."""

    def __init__(self, settings: Settings, speaker) -> None:
        self._tts = make_tts(settings.tts_provider, settings.section("tts", settings.tts_provider))
        self._speaker = speaker
        self._q: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, sentence: str) -> None:
        self._q.put(sentence)

    def wait(self) -> None:
        self._q.join()
        if self._speaker is not None:
            self._speaker.wait()

    def _run(self) -> None:
        while True:
            sentence = self._q.get()
            try:
                if sentence:
                    audio, rate = self._tts.synthesize(sentence)
                    if self._speaker is not None:
                        self._speaker.play(audio, rate)
            except Exception as exc:
                print(f"[voice] {exc}")
            finally:
                self._q.task_done()


async def respond(brain, voice: Voice, text: str) -> str:
    splitter = SentenceSplitter()
    reply = []
    async for chunk in brain.ask(text):
        reply.append(chunk)
        print(chunk, end="", flush=True)
        for sentence in splitter.feed(chunk):
            voice.say(sentence)
    for sentence in splitter.flush():
        voice.say(sentence)
    print()
    await asyncio.to_thread(voice.wait)
    return "".join(reply)


async def main_async(settings: Settings) -> None:
    mode = settings.input_mode
    speaker = None
    if settings.tts_provider != "print":
        from .audio import Speaker

        speaker = Speaker()

    print(f"Loading voice ({settings.tts_provider})...")
    voice = Voice(settings, speaker)

    stt = mic = wake = None
    if mode != "text":
        from .audio import Microphone, record_utterance

        print(f"Loading speech recognition ({settings.stt_provider})...")
        stt = make_stt(settings.stt_provider, settings.section("stt", settings.stt_provider))
        mic = Microphone().__enter__()
        if mode == "wakeword":
            from .wakeword import WakeWord

            wake = WakeWord(settings.wakeword_model, settings.wakeword_threshold)

    print(f"Connecting to Claude ({settings.brain_provider})...")
    brain = make_brain(
        settings.brain_provider,
        settings.section("brain", settings.brain_provider),
        settings.system_prompt,
    )
    await brain.start()

    prompts = {
        "text": "Type a message (or 'exit').",
        "push_to_talk": "Press Enter, then speak. Type 'exit' to quit.",
        "wakeword": "Say 'Hey Jarvis', then your request. Ctrl+C to quit.",
    }
    print(f"\nJarvis is ready. {prompts.get(mode, '')}\n")

    try:
        while True:
            if mode == "text":
                text = (await asyncio.to_thread(input, "you> ")).strip()
            else:
                if mode == "push_to_talk":
                    typed = (await asyncio.to_thread(input, "[Enter to talk] ")).strip()
                    if typed.lower() in EXIT_WORDS:
                        break
                    if typed:  # typing still works in voice mode
                        text = typed
                    else:
                        text = None
                else:
                    mic.flush()
                    await asyncio.to_thread(wake.wait, mic.frames())
                    text = None
                if text is None:
                    mic.flush()
                    print("Listening...")
                    started = time.monotonic()
                    audio = await asyncio.to_thread(
                        record_utterance,
                        mic.frames(),
                        speech_threshold=settings.speech_threshold,
                        silence_seconds=settings.silence_seconds,
                        max_seconds=settings.max_utterance_seconds,
                    )
                    text = (await asyncio.to_thread(stt.transcribe, audio)).strip()
                    print(f"you> {text}  ({time.monotonic() - started:.1f}s)")
            if not text:
                continue
            if text.lower().strip(" .!?") in EXIT_WORDS:
                voice.say("Goodbye.")
                await asyncio.to_thread(voice.wait)
                break
            print("jarvis> ", end="", flush=True)
            await respond(brain, voice, text)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        await brain.close()
        if mic is not None:
            mic.__exit__(None, None, None)


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis", description="Voice assistant powered by Claude.")
    parser.add_argument("--config", help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--mode", choices=["wakeword", "push_to_talk", "text"], help="override input_mode")
    parser.add_argument("--voice", help="override tts_provider, e.g. kokoro, elevenlabs, print")
    parser.add_argument("--ears", help="override stt_provider, e.g. faster_whisper, deepgram")
    parser.add_argument("--brain", help="override brain_provider: agent or chat")
    args = parser.parse_args()

    settings = load_settings(args.config)
    if args.mode:
        settings.input_mode = args.mode
    if args.voice:
        settings.tts_provider = args.voice
    if args.ears:
        settings.stt_provider = args.ears
    if args.brain:
        settings.brain_provider = args.brain
    try:
        asyncio.run(main_async(settings))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
