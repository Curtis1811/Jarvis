"""The thinking part: sends what you said to Claude and streams the reply back.

Options (set brain_provider in config.yaml):
- "agent": the Claude Agent SDK (the engine behind Claude Code). Claude can use
  tools such as web search, reading and writing files and running commands, so
  you can ask it to go and do things. Signs in with ANTHROPIC_API_KEY, or with
  your Claude login if you have run `claude login` on this PC.
- "chat": a plain Claude API conversation. Talk only, no tools, a bit faster.
  Needs ANTHROPIC_API_KEY.
- "openai_compatible": any OpenAI-style chat API, such as DeepSeek, OpenRouter,
  Groq or a local Ollama/LM Studio server. Talk only: it does not get Claude's
  tools, so it cannot go and do things.

Both keep the conversation going between turns, so follow-up questions work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

DEFAULT_MODEL = "claude-opus-5-5"


class Brain(Protocol):
    async def start(self) -> None: ...
    def ask(self, text: str) -> AsyncIterator[str]: ...
    async def close(self) -> None: ...


def make_brain(provider: str, options: dict[str, Any], system_prompt: str) -> Brain:
    if provider == "agent":
        return AgentBrain(system_prompt=system_prompt, **options)
    if provider == "chat":
        return ChatBrain(system_prompt=system_prompt, **options)
    if provider == "openai_compatible":
        return OpenAICompatibleBrain(system_prompt=system_prompt, **options)
    raise ValueError(
        f"Unknown brain_provider '{provider}'. Use agent, chat or openai_compatible."
    )


class AgentBrain:
    def __init__(
        self,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "default",
        cwd: str | None = None,
        max_turns: int | None = 25,
        max_budget_usd: float | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        from claude_agent_sdk import ClaudeAgentOptions

        self._options = ClaudeAgentOptions(
            model=model,
            system_prompt={"type": "preset", "preset": "claude_code", "append": system_prompt},
            # Tools Jarvis may use without asking. Anything else is refused,
            # because there is nobody at a keyboard to approve it.
            allowed_tools=allowed_tools
            if allowed_tools is not None
            else ["WebSearch", "WebFetch", "Read", "Glob", "Grep"],
            permission_mode=permission_mode,
            cwd=cwd,
            max_turns=max_turns,
            max_budget_usd=max_budget_usd,
            mcp_servers=mcp_servers or {},
            include_partial_messages=True,
        )
        self._client = None

    async def start(self) -> None:
        from claude_agent_sdk import ClaudeSDKClient

        self._client = ClaudeSDKClient(options=self._options)
        await self._client.connect()

    async def ask(self, text: str) -> AsyncIterator[str]:
        from claude_agent_sdk import AssistantMessage, ResultMessage, StreamEvent, TextBlock

        assert self._client is not None, "call start() first"
        await self._client.query(text)
        streamed = False
        async for msg in self._client.receive_response():
            if isinstance(msg, StreamEvent):
                event = msg.event
                if (
                    msg.parent_tool_use_id is None
                    and event.get("type") == "content_block_delta"
                    and event.get("delta", {}).get("type") == "text_delta"
                ):
                    streamed = True
                    yield event["delta"]["text"]
                elif event.get("type") == "message_stop":
                    yield "\n"  # end of one assistant message, flush the sentence
            elif isinstance(msg, AssistantMessage) and not streamed and msg.parent_tool_use_id is None:
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        yield block.text
            elif isinstance(msg, ResultMessage) and msg.is_error:
                yield f"\nSorry, something went wrong: {msg.result or msg.subtype}."

    async def interrupt(self) -> None:
        if self._client is not None:
            await self._client.interrupt()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.disconnect()
            self._client = None


class ChatBrain:
    def __init__(
        self,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        effort: str = "low",
        max_tokens: int = 4096,
    ) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic()
        self._system = system_prompt
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._messages: list[Any] = []

    async def start(self) -> None:
        return None

    async def ask(self, text: str) -> AsyncIterator[str]:
        self._messages.append({"role": "user", "content": text})
        async with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            # If a safety check declines, the server retries on a suitable model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for chunk in stream.text_stream:
                yield chunk
            final = await stream.get_final_message()

        if final.stop_reason == "refusal":
            self._messages.pop()
            yield " Sorry, I can't help with that one."
            return
        # Keep the full content (including thinking blocks) so the next turn is valid.
        self._messages.append({"role": "assistant", "content": final.content})

    async def close(self) -> None:
        await self._client.close()


class OpenAICompatibleBrain:
    """Talks to any OpenAI-style /chat/completions endpoint, streaming the reply."""

    def __init__(
        self,
        system_prompt: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        api_key_env: str | None = "DEEPSEEK_API_KEY",
        max_tokens: int = 2048,
        timeout: float = 120,
    ) -> None:
        import httpx

        from .config import require_env

        headers = {"Authorization": f"Bearer {require_env(api_key_env)}"} if api_key_env else {}
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    async def start(self) -> None:
        return None

    async def ask(self, text: str) -> AsyncIterator[str]:
        import json

        self._messages.append({"role": "user", "content": text})
        body = {
            "model": self._model,
            "messages": self._messages,
            "max_tokens": self._max_tokens,
            "stream": True,
        }
        reply: list[str] = []
        try:
            async with self._http.stream("POST", "/chat/completions", json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or []
                    delta = (choices[0].get("delta") or {}).get("content") if choices else None
                    if delta:
                        reply.append(delta)
                        yield delta
        except Exception:
            self._messages.pop()
            raise
        self._messages.append({"role": "assistant", "content": "".join(reply)})

    async def close(self) -> None:
        await self._http.aclose()
