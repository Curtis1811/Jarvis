# Jarvis

A voice assistant you talk to. You speak, Claude works on what you asked (it can
search the web, read files and more), and it answers out loud in a natural voice.

```
 "Hey Jarvis"  ->  listen  ->  speech to text  ->  Claude  ->  voice  ->  speakers
  wake word        mic          Whisper or         Agent SDK    Kokoro, CSM,
                                Deepgram           or API       ElevenLabs, OpenAI
```

Every stage can be switched in `config.yaml`, with a free local option and a paid
cloud option for each:

| Stage | Local (free, on your PC) | Paid (cloud) |
|---|---|---|
| Wake word | openWakeWord, built-in "Hey Jarvis" | |
| Ears (speech to text) | `faster_whisper` | `deepgram` |
| Brain | Ollama or LM Studio via `openai_compatible` (talk only) | `agent` (Claude with tools, works with your Claude subscription), `chat` (Claude API, talk only), DeepSeek and others via `openai_compatible` (talk only) |
| Voice | `kokoro` (fast), `csm` (Sesame, most human, needs NVIDIA GPU), `system` (Windows voice) | `elevenlabs` (most natural), `openai` (cheap) |

Jarvis starts speaking after the first sentence of Claude's reply, while the rest
is still being written, so you are not waiting for the whole answer.

### What about Dessa?

Dessa is Block's speech-to-speech model, announced in October 2026. For now it is
waitlist-only, with no API or download, and it only runs on high-end Macs. It also
takes audio in and gives audio out, so it cannot read Claude's text replies aloud.
`tts_provider: dessa` is reserved so it can be added if Block opens it up.

## Setup on Windows

1. **Install Python 3.12** from [python.org](https://www.python.org/downloads/).
   Tick "Add python.exe to PATH" in the installer. (3.12 is the safest choice for
   the local voice models.)
2. **Get the code** and open a terminal (PowerShell) in the folder:
   ```powershell
   git clone https://github.com/<you>/jarvis.git
   cd jarvis
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   If PowerShell refuses to run the activate script, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
3. **Install Jarvis** with the local extras:
   ```powershell
   pip install -e ".[local]"
   ```
   On an NVIDIA card (an RTX 4070 or 3070 is plenty), install the CUDA build of
   PyTorch **before** the line above, so Kokoro and CSM run on the GPU:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
   ```
   The second line gives faster-whisper the CUDA libraries it needs. If Whisper
   still fails to load on the GPU, set its `device: cpu` and `model: small.en`.
4. **Copy the settings files**:
   ```powershell
   copy config.example.yaml config.yaml
   copy .env.example .env
   ```
5. **Sign in to Claude.** Install [Claude Code](https://claude.com/claude-code)
   and run `claude login` to use your Claude subscription with the default `agent`
   brain. Leave `ANTHROPIC_API_KEY` blank in `.env`, or it is used instead of
   your subscription.

## Running it

```powershell
python -m jarvis --mode text --voice print   # type, read replies: checks Claude works
python -m jarvis --mode text                  # type, hear replies: checks the voice
python -m jarvis                              # press Enter and talk
python -m jarvis --mode wakeword              # always listening for "Hey Jarvis"
```

Say or type "goodbye" to stop. `--voice`, `--ears` and `--brain` switch providers
for one run, for example `python -m jarvis --voice elevenlabs`.

## Trying the voices

| Voice | What to do |
|---|---|
| `kokoro` | Works after `pip install -e ".[kokoro]"`. Voices: `bm_george`, `bm_lewis` (British male), `am_michael` (American male), `af_heart`, `bf_emma`. |
| `csm` | `pip install -e ".[csm]"`, accept the terms at [huggingface.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b), then run `huggingface-cli login`. Needs about 8 GB of GPU memory. Slower to start, most natural local option. |
| `elevenlabs` | Add `ELEVENLABS_API_KEY` to `.env`. Change `voice_id` to any voice from their library. |
| `openai` | Add `OPENAI_API_KEY` to `.env`. Change the speaking style with `instructions`. |
| `system` | `pip install -e ".[system]"`. The built-in Windows voice, no download. |

## Personalities

Jarvis comes with three personalities, each with its own way of talking and its
own voice:

| Personality | Style | Kokoro voice | OpenAI voice |
|---|---|---|---|
| `jarvis` | Calm, dry-witted British butler | `bm_george` | `onyx` |
| `friday` | Quick, warm and cheeky, with an Irish turn of phrase | `bf_isabella` | `nova` (Irish accent) |
| `karen` | Friendly, upbeat and encouraging | `af_heart` | `coral` |

Pick one in the settings app, with `personality:` in `config.yaml`, or for one run
with `python -m jarvis --personality friday`. In the settings app you can change a
personality's voice and instructions, or make a new one. The wake word stays
"Hey Jarvis" for all of them. Kokoro has no Irish voice, so Friday's accent only
comes through with the OpenAI voice.

To download every Kokoro voice up front, so nothing downloads mid-conversation:

```powershell
python -m jarvis.tts.kokoro_tts
```

## Settings app

Double-click `Jarvis Settings.bat` (or run `python -m jarvis.settings_ui`) and a
settings page opens in your browser at http://127.0.0.1:8765. From there you can:

- pick the voice engine and voice (all the Kokoro voices, like `bm_george`), and
  press **Test voice** to hear it before saving;
- switch the brain between the Claude agent, Claude chat, or DeepSeek, Ollama,
  LM Studio, OpenRouter and Groq, and choose which tools the agent may use;
- change speech to text, mic sensitivity and the wake word;
- switch personality (Jarvis, Friday, Karen or your own) and edit its voice and
  instructions;
- save API keys to `.env`;
- start and stop Jarvis, watch what it hears and says, and press **Talk** or type
  to it.

The page only works on your own PC. Keep the settings window open while you use
it. Saving rewrites `config.yaml` without its comments (the first save keeps a
copy as `config.yaml.bak`); `config.example.yaml` still explains every option.
If Jarvis is running, press **Restart to apply** after saving.

## Choosing a brain

`agent` is the default and the only brain that can act: search the web, read
files, run tasks. It runs on your Claude subscription once you have run
`claude login`, so there is no extra bill.

`openai_compatible` works with DeepSeek (cheap), OpenRouter, Groq, or a model
running on your own PC through Ollama or LM Studio. It is good for plain
conversation, but it does not get Claude's tools, so it cannot go and do things.
Set `base_url`, `model` and `api_key_env` under `brain.openai_compatible`.

## Letting Jarvis do things

The `agent` brain is the Claude Agent SDK, the same engine as Claude Code. By
default it may search the web and read files without asking. Nobody is at a
keyboard to approve actions, so anything not in `allowed_tools` is refused.

To let it change files or run commands, add `Write`, `Edit` or `Bash` to
`allowed_tools` and set `cwd` to a folder you are happy for it to work in.
`max_budget_usd` caps what one request can spend.

You can also connect MCP servers (Google Calendar, Gmail, Spotify, Home Assistant
and so on) under `brain.agent.mcp_servers`, using the same format as Claude Code.

## Costs, roughly

- Claude: depends on the model and how much work each request does. `max_budget_usd`
  caps each request. Switch `model` to `claude-sonnet-5-5` or `claude-haiku-5-5` to
  spend less.
- ElevenLabs: their Starter and Creator plans cover a lot of casual talking.
- OpenAI voice and Deepgram: cents per hour of speech.
- Everything local is free once downloaded.

## Development

```powershell
pip install -e ".[dev]"
pytest
```

The code is small: `jarvis/__main__.py` runs the loop, `brain.py` talks to Claude,
`speech.py` splits replies into sentences, `audio.py` handles the mic and speakers,
and `stt/` and `tts/` hold one file per provider. To add a voice, write a class with
`synthesize(text) -> (audio, sample_rate)` and register it in `jarvis/tts/__init__.py`.
