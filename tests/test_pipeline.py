import asyncio

import numpy as np
import pytest

from jarvis.__main__ import Voice, respond
from jarvis.audio import float_to_wav_bytes, pcm16_to_float, record_utterance
from jarvis.config import Settings, load_settings
from jarvis.speech import SentenceSplitter, clean_for_speech
from jarvis.tts import make_tts


def feed_all(chunks):
    splitter = SentenceSplitter()
    out = []
    for c in chunks:
        out += splitter.feed(c)
    return out + splitter.flush()


def test_sentences_come_out_as_they_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("Good evening, sir. All systems are online. The weather tod") == [
        "Good evening, sir. All systems are online."
    ]
    assert splitter.feed("ay is mild and sunny. ") == ["The weather today is mild and sunny."]
    assert splitter.flush() == []


def test_short_sentences_are_merged():
    assert feed_all(["Sure. ", "I'll check the forecast for you now. "]) == [
        "Sure. I'll check the forecast for you now."
    ]


def test_markdown_is_not_read_aloud():
    assert clean_for_speech("## Result\n- **Bold** item with `code`") == "Result Bold item with code"
    assert clean_for_speech("See [the docs](https://x.y/z) or https://a.b") == "See the docs or the link"


def test_code_blocks_are_summarised():
    out = feed_all(["Here you go:\n```py\nprint('hi')\n", "```\nThat prints hi. "])
    assert "'hi'" not in " ".join(out)
    assert any("on screen" in s for s in out)


def test_record_utterance_stops_after_silence():
    frame = 1280
    loud = [np.full(frame, 0.1, np.float32)] * 10
    quiet = [np.zeros(frame, np.float32)] * 30
    audio = record_utterance(iter(quiet[:3] + loud + quiet), silence_seconds=0.8)
    # three quiet pre-roll frames, ten loud frames, then ten quiet frames (0.8 s)
    assert len(audio) == (3 + 10 + 10) * frame
    assert record_utterance(iter(quiet * 100), start_timeout=1.0).size == 0


def test_wav_and_pcm_round_trip():
    wav = float_to_wav_bytes(np.array([0.0, 0.5, -0.5], np.float32), 16_000)
    assert wav[:4] == b"RIFF"
    pcm = pcm16_to_float(np.array([0, 16384, -16384], "<i2").tobytes())
    assert np.allclose(pcm, [0, 0.5, -0.5])


def test_config_loads_and_rejects_typos(tmp_path):
    good = tmp_path / "c.yaml"
    good.write_text("tts_provider: elevenlabs\ntts:\n  elevenlabs:\n    voice_id: abc\n")
    s = load_settings(good)
    assert s.tts_provider == "elevenlabs" and s.section("tts", "elevenlabs") == {"voice_id": "abc"}
    bad = tmp_path / "bad.yaml"
    bad.write_text("tts_provder: kokoro\n")
    with pytest.raises(ValueError, match="tts_provder"):
        load_settings(bad)


def test_example_config_is_valid():
    s = load_settings("config.example.yaml")
    assert s.brain_provider == "agent" and s.tts_provider == "kokoro"


def test_unknown_and_dessa_providers_explain_themselves():
    with pytest.raises(ValueError):
        make_tts("nope", {})
    with pytest.raises(NotImplementedError, match="waitlist"):
        make_tts("dessa", {})


class FakeBrain:
    async def ask(self, text):
        for chunk in ["Right away. ", "I have looked it ", "up and it is sunny in London.", "\n"]:
            yield chunk


def test_respond_speaks_each_sentence(capsys):
    voice = Voice(Settings(tts_provider="print"), speaker=None)
    reply = asyncio.run(respond(FakeBrain(), voice, "weather?"))
    assert reply.startswith("Right away.")
    spoken = [line for line in capsys.readouterr().out.splitlines() if line.startswith("🔊")]
    assert spoken == ["🔊 Right away. I have looked it up and it is sunny in London."]


def test_blank_keys_in_env_do_not_hide_subscription_login(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=\n")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_settings()
    assert "ANTHROPIC_API_KEY" not in __import__("os").environ


def test_openai_compatible_brain_streams_and_keeps_history(monkeypatch):
    import httpx

    from jarvis.brain import make_brain

    seen = []

    def handler(request):
        import json

        seen.append(json.loads(request.content))
        sse = (
            'data: {"choices":[{"delta":{"content":"Hello "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"sir."}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    brain = make_brain("openai_compatible", {}, "be brief")
    brain._http = httpx.AsyncClient(base_url="https://x", transport=httpx.MockTransport(handler))

    async def two_turns():
        a = "".join([c async for c in brain.ask("hi")])
        b = "".join([c async for c in brain.ask("again")])
        return a, b

    assert asyncio.run(two_turns()) == ("Hello sir.", "Hello sir.")
    assert [m["role"] for m in seen[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert seen[0]["model"] == "deepseek-chat" and seen[0]["stream"] is True
