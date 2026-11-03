import logging

import pytest

from jarvis.config import Settings, load_settings
from jarvis.personalities import all_personalities, get
from jarvis.quiet import quiet


def test_builtin_personalities_have_their_own_voice_and_prompt():
    everyone = all_personalities()
    assert {"jarvis", "friday", "karen"} <= set(everyone)
    for p in everyone.values():
        assert p["name"] in p["system_prompt"]
        assert "read aloud" in p["system_prompt"]
        assert p["tts"]["kokoro"]["voice"]
    assert len({p["tts"]["kokoro"]["voice"] for p in everyone.values()}) == 3


def test_personality_sets_prompt_and_voice():
    s = Settings(personality="friday", tts={"kokoro": {"voice": "bm_george", "speed": 1.2}})
    persona = s.apply_personality()
    assert persona["name"] == "Friday" and s.system_prompt.startswith("You are Friday")
    # the voice changes, other voice settings stay
    assert s.section("tts", "kokoro") == {"voice": "bf_isabella", "speed": 1.2}


def test_explicit_system_prompt_still_wins():
    s = Settings(personality="karen", system_prompt="Be terse.")
    s.apply_personality()
    assert s.system_prompt == "Be terse." and s.section("tts", "kokoro")["voice"] == "af_heart"


def test_custom_and_changed_personalities_from_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "personality: edith\n"
        "personalities:\n"
        "  friday:\n    tts: {kokoro: {voice: bf_emma}}\n"
        "  Edith:\n    name: Edith\n    tts: {kokoro: {voice: af_nova}}\n"
    )
    s = load_settings(cfg)
    persona = s.apply_personality()
    assert persona["name"] == "Edith" and "You are Edith" in s.system_prompt
    assert s.section("tts", "kokoro")["voice"] == "af_nova"
    friday = get("friday", s.personalities)
    assert friday["tts"]["kokoro"]["voice"] == "bf_emma" and friday["tts"]["openai"]["voice"] == "nova"


def test_unknown_personality_lists_choices():
    with pytest.raises(ValueError, match="jarvis, friday, karen"):
        Settings(personality="ultron").apply_personality()


def test_quiet_hides_library_notices(caplog):
    quiet()
    log = logging.getLogger("torch.distributed.elastic.multiprocessing.redirects")
    log.setLevel(logging.WARNING)  # torch resets the level; the filter must survive that
    with caplog.at_level(logging.WARNING):
        log.warning("NOTE: Redirects are currently not supported in Windows or MacOs.")
        log.warning("something real")
    assert [r.getMessage() for r in caplog.records] == ["something real"]
