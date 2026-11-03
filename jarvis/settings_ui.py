"""Settings app: change Jarvis's voice, brain and ears from a web page, and start
or stop Jarvis. Run `python -m jarvis.settings_ui` (or double-click
`Jarvis Settings.bat`) and it opens in your browser at http://127.0.0.1:8765.

It only uses the Python standard library, and it only listens on your own PC.
Settings are saved to config.yaml and API keys to .env, the same files Jarvis
reads when it starts.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import subprocess
import sys
import threading
import webbrowser
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from . import personalities as persona_lib
from .config import Settings

PAGE = Path(__file__).with_name("settings_ui.html")

# Keys the page may set in .env. Values are never sent back to the browser.
API_KEYS = {
    "ANTHROPIC_API_KEY": "Claude API (leave blank to use your Claude subscription)",
    "DEEPSEEK_API_KEY": "DeepSeek",
    "OPENROUTER_API_KEY": "OpenRouter",
    "GROQ_API_KEY": "Groq",
    "ELEVENLABS_API_KEY": "ElevenLabs voice",
    "OPENAI_API_KEY": "OpenAI voice",
    "DEEPGRAM_API_KEY": "Deepgram speech to text",
}

# Choices shown in the page. Free text is still allowed for models and voices.
CATALOG: dict[str, Any] = {
    "input_modes": {
        "push_to_talk": "Press Talk (or Enter), then speak",
        "wakeword": 'Hands-free: say "Hey Jarvis"',
        "text": "Type instead of talking",
    },
    "tts_providers": {
        "kokoro": "Kokoro (local, free, fast)",
        "csm": "Sesame CSM (local, most human, needs GPU)",
        "elevenlabs": "ElevenLabs (paid, most natural)",
        "openai": "OpenAI (paid, cheap)",
        "system": "Windows voice (robotic, no setup)",
        "print": "Silent (print replies only)",
    },
    "kokoro_voices": {
        "American female": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_nova",
                            "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_river"],
        "American male": ["am_michael", "am_fenrir", "am_puck", "am_adam", "am_echo", "am_eric",
                          "am_liam", "am_onyx", "am_santa"],
        "British female": ["bf_emma", "bf_isabella", "bf_alice", "bf_lily"],
        "British male": ["bm_george", "bm_fable", "bm_lewis", "bm_daniel"],
    },
    "openai_voices": ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage",
                      "shimmer", "verse"],
    "openai_tts_models": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
    "elevenlabs_models": ["eleven_flash_v2_5", "eleven_turbo_v2_5", "eleven_multilingual_v2", "eleven_v3"],
    "stt_providers": {
        "faster_whisper": "Whisper (local, free)",
        "deepgram": "Deepgram (paid, fastest)",
    },
    "whisper_models": ["distil-large-v3", "large-v3-turbo", "large-v3", "medium.en", "small.en",
                       "base.en", "tiny.en"],
    "devices": ["auto", "cuda", "cpu"],
    "brain_providers": {
        "agent": "Claude agent (can do things, uses your Claude subscription)",
        "chat": "Claude chat (API key, talk only, a bit faster)",
        "openai_compatible": "Other model: DeepSeek, Ollama, OpenRouter, Groq (talk only)",
    },
    "claude_models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "agent_tools": ["WebSearch", "WebFetch", "Read", "Glob", "Grep", "Write", "Edit", "Bash"],
    "efforts": ["low", "medium", "high"],
    "openai_compatible_presets": {
        "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                     "api_key_env": "DEEPSEEK_API_KEY"},
        "Ollama (on this PC)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1",
                                "api_key_env": None},
        "LM Studio (on this PC)": {"base_url": "http://localhost:1234/v1", "model": "local-model",
                                   "api_key_env": None},
        "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "deepseek/deepseek-chat",
                       "api_key_env": "OPENROUTER_API_KEY"},
        "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile",
                 "api_key_env": "GROQ_API_KEY"},
    },
    "api_keys": API_KEYS,
}

PREVIEW_TEXT = "Good evening. All systems are online, and I am ready when you are."


# ---------------------------------------------------------------- config files


class ConfigStore:
    """Reads and writes config.yaml and .env in the Jarvis folder."""

    def __init__(self, config_path: Path, env_path: Path, example_path: Path | None = None) -> None:
        self.config_path = config_path
        self.env_path = env_path
        self.example_path = example_path

    def raw(self) -> dict[str, Any]:
        """What is in config.yaml, or config.example.yaml before the first save."""
        for path in (self.config_path, self.example_path):
            if path and path.exists():
                return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {}

    def effective(self) -> dict[str, Any]:
        """Every setting with Jarvis's built-in defaults filled in."""
        data = asdict(Settings())
        for key, value in self.raw().items():
            data[key] = value
        for section in ("stt", "brain", "tts", "personalities"):
            data[section] = data.get(section) or {}
        return data

    def save(self, updates: dict[str, Any]) -> dict[str, Any]:
        known = set(asdict(Settings()))
        unknown = sorted(set(updates) - known)
        if unknown:
            raise ValueError(f"Unknown setting(s): {', '.join(unknown)}")
        data = self.raw()
        for key, value in updates.items():
            if key in ("stt", "brain", "tts", "personalities") and isinstance(value, dict):
                section = dict(data.get(key) or {})
                for provider, options in value.items():
                    if key == "personalities":
                        provider = provider.lower()
                    if options is None:  # remove (a personality back to how it came)
                        section.pop(provider, None)
                    else:
                        section[provider] = options
                if section:
                    data[key] = section
                else:
                    data.pop(key, None)
            elif key == "system_prompt" and not (value or "").strip():
                data.pop(key, None)  # blank means "use the default personality"
            else:
                data[key] = value
        # The first save replaces the commented file, so keep a copy of it.
        backup = self.config_path.with_name(self.config_path.name + ".bak")
        if self.config_path.exists() and not backup.exists():
            backup.write_text(self.config_path.read_text(encoding="utf-8"), encoding="utf-8")
        header = (
            "# Saved by the Jarvis settings app. config.example.yaml explains every option.\n"
        )
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
        self.config_path.write_text(header + body, encoding="utf-8")
        return self.effective()

    def env_status(self) -> dict[str, bool]:
        values = self._read_env()
        return {name: bool(values.get(name, "").strip()) for name in API_KEYS}

    def set_env(self, updates: dict[str, str | None]) -> dict[str, bool]:
        """Set or clear keys in .env, keeping every other line as it was."""
        for name in updates:
            if name not in API_KEYS:
                raise ValueError(f"Unknown key {name}")
        lines = self.env_path.read_text(encoding="utf-8").splitlines() if self.env_path.exists() else []
        pending = {k: (v or "").strip() for k, v in updates.items()}
        for i, line in enumerate(lines):
            name = line.split("=", 1)[0].strip()
            if name in pending and "=" in line and not line.lstrip().startswith("#"):
                lines[i] = f"{name}={pending.pop(name)}"
        lines += [f"{k}={v}" for k, v in pending.items()]
        self.env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return self.env_status()

    def apply_env(self) -> None:
        """Copy the keys in .env into this process, so a voice preview, and the
        Jarvis it starts, use what is in the file now."""
        for name, value in self._read_env().items():
            if name not in API_KEYS:
                continue
            if value.strip():
                os.environ[name] = value.strip()
            else:  # blank in .env means "not set" (and keeps the Claude login working)
                os.environ.pop(name, None)

    def _read_env(self) -> dict[str, str]:
        if not self.env_path.exists():
            return {}
        from dotenv import dotenv_values

        return {k: v or "" for k, v in dotenv_values(self.env_path).items()}


# ---------------------------------------------------------------- Jarvis process


class JarvisProcess:
    """Runs `python -m jarvis` in the background and keeps what it prints."""

    MAX_LOG = 200_000

    def __init__(self, cwd: Path, command: list[str] | None = None) -> None:
        self.cwd = cwd
        self._command = command
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._log = ""
        self._dropped = 0  # characters trimmed off the front of the log
        self.mode: str | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, mode: str, config_path: Path) -> None:
        with self._lock:
            if self.running:
                raise RuntimeError("Jarvis is already running.")
            cmd = self._command or [sys.executable, "-u", "-m", "jarvis", "--config", str(config_path)]
            if mode:
                cmd = [*cmd, "--mode", mode]
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       HF_HUB_DISABLE_SYMLINKS_WARNING="1", PYTHONWARNINGS="ignore")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._append(f"\n--- Starting Jarvis ({mode or 'default mode'}) ---\n")
            self._proc = subprocess.Popen(
                cmd, cwd=self.cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, creationflags=flags,
            )
            self.mode = mode
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()

    def send(self, text: str) -> None:
        """Type a line into Jarvis: a message in text mode, or "" to talk in push-to-talk."""
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            raise RuntimeError("Jarvis is not running.")
        proc.stdin.write((text.replace("\n", " ") + "\n").encode("utf-8"))
        proc.stdin.flush()

    def stop(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:  # ask nicely first: "exit" works in push-to-talk and text mode
            if self.mode == "wakeword":
                raise RuntimeError("wake word mode does not read typed input")
            self.send("exit")
            proc.wait(timeout=4)
        except Exception:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._append("--- Jarvis stopped ---\n")

    def output(self, since: int = 0) -> tuple[str, int]:
        with self._lock:
            end = self._dropped + len(self._log)
            start = max(since, self._dropped)
            return self._log[start - self._dropped:], end

    def _append(self, text: str) -> None:
        self._log += text
        if len(self._log) > self.MAX_LOG:
            cut = len(self._log) - self.MAX_LOG
            self._log = self._log[cut:]
            self._dropped += cut

    def _pump(self, proc: subprocess.Popen[bytes]) -> None:
        import codecs

        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        assert proc.stdout is not None
        while True:
            # read1 returns as soon as anything arrives, so prompts without a
            # newline (like "you> ") still show up.
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            with self._lock:
                self._append(decoder.decode(chunk).replace("\r\n", "\n"))
        code = proc.wait()
        with self._lock:
            self._append(f"--- Jarvis exited (code {code}) ---\n" if code else "")


# ---------------------------------------------------------------- voice preview


class Previewer:
    """Speaks a test sentence with the chosen voice, keeping the last voice loaded."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._tts = None
        self._lock = threading.Lock()

    def wav(self, provider: str, options: dict[str, Any], text: str) -> bytes:
        from .audio import float_to_wav_bytes
        from .tts import make_tts

        key = json.dumps([provider, options], sort_keys=True)
        with self._lock:
            if key != self._key:
                self._tts, self._key = None, None
                self._tts = make_tts(provider, copy.deepcopy(options))
                self._key = key
            audio, rate = self._tts.synthesize(text or PREVIEW_TEXT)
        return float_to_wav_bytes(audio, rate)


# ---------------------------------------------------------------- web server


class App:
    def __init__(self, folder: Path, config_path: Path | None = None,
                 command: list[str] | None = None) -> None:
        folder = folder.resolve()
        self.config_path = (config_path or folder / "config.yaml").resolve()
        self.store = ConfigStore(self.config_path, folder / ".env", folder / "config.example.yaml")
        self.jarvis = JarvisProcess(folder, command)
        self.previewer = Previewer()

    def state(self) -> dict[str, Any]:
        return {
            "config": self.store.effective(),
            "saved": self.config_path.exists(),
            "config_path": str(self.config_path),
            "keys": self.store.env_status(),
            "catalog": CATALOG,
            "personalities": persona_lib.all_personalities(self.store.raw().get("personalities")),
            "builtin_personalities": list(persona_lib.BUILTIN),
        }

    def status(self, since: int) -> dict[str, Any]:
        text, nxt = self.jarvis.output(since)
        return {"running": self.jarvis.running, "mode": self.jarvis.mode, "output": text, "next": nxt}

    def handle(self, method: str, path: str, query: dict[str, list[str]], body: Any) -> Any:
        if method == "GET" and path == "/api/state":
            return self.state()
        if method == "GET" and path == "/api/status":
            return self.status(int((query.get("since") or ["0"])[0]))
        if method == "POST" and path == "/api/config":
            self.store.save(body)
            return self.state()
        if method == "POST" and path == "/api/keys":
            return {"keys": self.store.set_env(body)}
        if method == "POST" and path == "/api/start":
            if not self.config_path.exists():  # first run: save the defaults
                self.store.save({})
            self.store.apply_env()
            self.jarvis.start(body.get("mode") or "", self.config_path)
            return self.status(0)
        if method == "POST" and path == "/api/stop":
            self.jarvis.stop()
            return self.status(0)
        if method == "POST" and path == "/api/send":
            self.jarvis.send(body.get("text", ""))
            return {"ok": True}
        raise LookupError(path)


def make_handler(app: App, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data).encode("utf-8"), "application/json")

        def _route(self, method: str) -> None:
            # Only answer pages opened on this PC, so other websites can't drive Jarvis.
            if self.headers.get("Host") not in allowed_hosts:
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Open this page from 127.0.0.1"})
            url = urlparse(self.path)
            if method == "GET" and url.path in ("/", "/index.html"):
                return self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            body: Any = {}
            if method == "POST":
                if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                    return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "Send JSON"})
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
            try:
                if method == "POST" and url.path == "/api/preview":
                    app.store.apply_env()
                    wav = app.previewer.wav(body.get("provider", ""), body.get("options") or {},
                                            body.get("text", ""))
                    return self._send(HTTPStatus.OK, wav, "audio/wav")
                return self._json(HTTPStatus.OK, app.handle(method, url.path, parse_qs(url.query), body))
            except LookupError:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            except Exception as exc:  # shown to the user in the page
                return self._json(HTTPStatus.BAD_REQUEST, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def serve(folder: Path, port: int = 8765, open_browser: bool = True, config_path: Path | None = None) -> None:
    app = App(folder, config_path)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    atexit.register(app.jarvis.stop)
    url = f"http://127.0.0.1:{port}/"
    print(f"Jarvis settings are open at {url}\nKeep this window open. Close it to stop.")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.jarvis.stop()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="jarvis-settings", description="Jarvis settings app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--config", help="path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()
    from .quiet import quiet

    quiet()
    serve(Path.cwd(), args.port, not args.no_browser, Path(args.config) if args.config else None)


if __name__ == "__main__":
    main()
