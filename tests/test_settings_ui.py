import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from jarvis.config import load_settings
from jarvis.settings_ui import App, ConfigStore, JarvisProcess, make_handler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def folder(tmp_path):
    shutil.copy(ROOT / "config.example.yaml", tmp_path / "config.example.yaml")
    return tmp_path


def store_for(folder):
    return ConfigStore(folder / "config.yaml", folder / ".env", folder / "config.example.yaml")


def test_save_changes_voice_and_brain_and_jarvis_can_load_it(folder):
    store = store_for(folder)
    assert store.effective()["tts"]["kokoro"]["voice"] == "bm_george"  # from the example before first save
    store.save({"tts_provider": "openai", "tts": {"kokoro": {"voice": "am_michael", "speed": 1.1}},
                "brain_provider": "openai_compatible", "system_prompt": ""})
    s = load_settings(folder / "config.yaml")
    assert s.tts_provider == "openai" and s.brain_provider == "openai_compatible"
    assert s.section("tts", "kokoro") == {"voice": "am_michael", "speed": 1.1}
    assert s.section("tts", "openai")["voice"] == "onyx"  # other providers kept
    assert s.section("brain", "agent")["allowed_tools"]  # untouched sections kept


def test_first_save_keeps_a_backup_of_the_commented_file(folder):
    (folder / "config.yaml").write_text("# my notes\ninput_mode: text\n")
    store_for(folder).save({"input_mode": "wakeword"})
    assert (folder / "config.yaml.bak").read_text() == "# my notes\ninput_mode: text\n"
    assert yaml.safe_load((folder / "config.yaml").read_text())["input_mode"] == "wakeword"


def test_unknown_settings_are_refused(folder):
    with pytest.raises(ValueError, match="voice_provider"):
        store_for(folder).save({"voice_provider": "x"})


def test_keys_are_written_to_env_without_losing_other_lines(folder):
    env = folder / ".env"
    env.write_text("# comment\nANTHROPIC_API_KEY=\nOTHER=1\n")
    store = store_for(folder)
    assert store.env_status()["ELEVENLABS_API_KEY"] is False
    status = store.set_env({"ELEVENLABS_API_KEY": "el-123", "ANTHROPIC_API_KEY": ""})
    assert status["ELEVENLABS_API_KEY"] is True and status["ANTHROPIC_API_KEY"] is False
    assert env.read_text() == "# comment\nANTHROPIC_API_KEY=\nOTHER=1\nELEVENLABS_API_KEY=el-123\n"
    with pytest.raises(ValueError):
        store.set_env({"PATH": "x"})


def test_process_runs_takes_input_and_stops(tmp_path):
    script = tmp_path / "fake_jarvis.py"
    script.write_text(
        "import sys\n"
        "print('ready', flush=True)\n"
        "for line in sys.stdin:\n"
        "    if line.strip() == 'exit':\n"
        "        break\n"
        "    print('heard', line.strip(), flush=True)\n"
    )
    proc = JarvisProcess(tmp_path, command=[sys.executable, "-u", str(script)])
    proc.start("text", tmp_path / "config.yaml")
    assert proc.running
    proc.send("hello there")
    deadline = time.time() + 10
    while "heard hello there" not in proc.output()[0] and time.time() < deadline:
        time.sleep(0.05)
    assert "heard hello there" in proc.output()[0]
    proc.stop()
    assert not proc.running
    _text, end = proc.output()
    assert proc.output(end)[0] == ""


@pytest.fixture
def server(folder):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = srv.server_address[1]
    srv.RequestHandlerClass = make_handler(App(folder), port)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()


def call(url, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req) as res:
        return res.headers.get("Content-Type"), res.read()


def test_page_and_api_work_over_http(server, folder):
    ctype, page = call(server + "/")
    assert ctype.startswith("text/html") and b"Jarvis" in page
    state = json.loads(call(server + "/api/state")[1])
    assert "bm_george" in state["catalog"]["kokoro_voices"]["British male"]
    state["config"]["tts"]["kokoro"]["voice"] = "bf_emma"
    call(server + "/api/config", {"tts": state["config"]["tts"]})
    assert load_settings(folder / "config.yaml").section("tts", "kokoro")["voice"] == "bf_emma"
    ctype, wav = call(server + "/api/preview", {"provider": "print", "options": {}, "text": "hi"})
    assert ctype == "audio/wav" and wav[:4] == b"RIFF"
    assert json.loads(call(server + "/api/status")[1])["running"] is False


def test_other_websites_cannot_use_the_api(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        call(server + "/api/state", headers={"Host": "evil.example:80"})
    assert err.value.code == 403
    req = urllib.request.Request(server + "/api/stop", data=b"{}", headers={"Content-Type": "text/plain"})
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req)
    assert err.value.code == 415
