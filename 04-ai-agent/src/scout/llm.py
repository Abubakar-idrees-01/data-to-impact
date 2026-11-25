"""An async chat client for OpenAI-compatible APIs (Groq, OpenAI, a local llama.cpp/Ollama server) with tool calling.

What an agent needs beyond plain chat:
- **Tool calls**: the model replies with `tool_calls` (a function name + JSON arguments) instead of text.
- **Rate limits**: free tiers allow a few thousand tokens per minute; on 429/5xx we wait as long as the provider
  asks (Retry-After header, or "try again in 5.9s" in the message) and retry.
- **Malformed tool calls**: Groq rejects a reply whose tool call isn't valid JSON (400 `tool_use_failed`);
  sampling again usually fixes it, so we retry that a couple of times too.
- **Accounting**: tokens, calls and seconds per task, so cost and latency can be reported.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str            # raw JSON text, exactly as the model wrote it


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0

    def as_message(self) -> dict:
        """The assistant message to append to the conversation (only the fields every provider accepts)."""
        msg: dict = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.name, "arguments": c.arguments}} for c in self.tool_calls]
        return msg


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    waits: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def retry_wait(headers: dict, body: str, attempt: int, max_wait: float) -> float:
    """Seconds to wait: Retry-After header, else the provider's "try again in 1m5.8s" hint, else 2^attempt."""
    header = {k.lower(): v for k, v in headers.items()}.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, max_wait)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", body)
    if m:
        return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5, max_wait)
    return min(2.0 ** attempt, max_wait)


def parse_reply(data: dict, seconds: float = 0.0) -> Reply:
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    calls = [ToolCall(id=c.get("id") or f"call_{i}", name=c["function"]["name"],
                      arguments=c["function"].get("arguments") or "{}")
             for i, c in enumerate(msg.get("tool_calls") or [])]
    usage = data.get("usage") or {}
    return Reply(content=_THINK.sub("", msg.get("content") or "").strip(), tool_calls=calls,
                 finish_reason=choice.get("finish_reason") or "", prompt_tokens=usage.get("prompt_tokens", 0),
                 completion_tokens=usage.get("completion_tokens", 0), seconds=seconds)


class LLM:
    """`await llm.chat(messages, tools)` -> Reply. One instance per task keeps that task's usage."""

    def __init__(self, base_url: str, model: str, api_key: str = "", *, reasoning_effort: str | None = None,
                 timeout: float = 120.0, max_retries: int = 8, max_wait: float = 65.0,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, transport: httpx.AsyncBaseTransport | None = None):
        self.url = base_url.strip().rstrip("/") + "/chat/completions"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.reasoning_effort = reasoning_effort if "gpt-oss" in self.model else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_wait = max_wait
        self.sleep = sleep
        self.transport = transport
        self.usage = Usage()

    @classmethod
    def from_settings(cls, s, **kwargs) -> "LLM":
        if not s.api_key and "localhost" not in s.api_base_url and "127.0.0.1" not in s.api_base_url:
            raise LLMError("set SCOUT_API_KEY in 04-ai-agent/.env (see docs/SETUP.md)")
        return cls(s.api_base_url, s.api_model, s.api_key, reasoning_effort=s.reasoning_effort, **kwargs)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, *, max_tokens: int = 2048,
                   temperature: float = 0.0) -> Reply:
        body: dict = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        t0 = time.time()
        data = await self._post(body)
        reply = parse_reply(data, time.time() - t0)
        self.usage.calls += 1
        self.usage.prompt_tokens += reply.prompt_tokens
        self.usage.completion_tokens += reply.completion_tokens
        self.usage.seconds += reply.seconds
        if not reply.content and not reply.tool_calls and reply.finish_reason == "length":
            raise LLMError(f"empty reply: the {max_tokens}-token budget ran out while the model was reasoning")
        return reply

    async def _post(self, body: dict) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "scout-agent/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        malformed = 0
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    r = await client.post(self.url, json=body, headers=headers)
                except (httpx.TransportError, TimeoutError) as e:
                    if attempt < self.max_retries:
                        wait = min(2.0 ** attempt, 60.0)
                        log.warning("network error (%s), retrying in %.0f s", e, wait)
                        await self.sleep(wait)
                        continue
                    raise LLMError(f"network error: {e}") from None
                if r.status_code == 200:
                    return r.json()
                text = r.text
                if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                    wait = retry_wait(dict(r.headers), text, attempt, self.max_wait)
                    log.warning("API %s, waiting %.1f s (attempt %d/%d)", r.status_code, wait, attempt + 1, self.max_retries)
                    self.usage.waits.append(wait)
                    await self.sleep(wait)
                    continue
                if r.status_code == 400 and "tool_use_failed" in text and malformed < 2:
                    malformed += 1                     # the model wrote an invalid tool call: sample again
                    log.warning("model produced a malformed tool call, retrying (%d/2)", malformed)
                    body = {**body, "temperature": max(body.get("temperature", 0.0), 0.3)}
                    continue
                try:
                    detail = r.json().get("error", {}).get("message") or text
                except (ValueError, AttributeError):
                    detail = text
                raise LLMError(f"API error {r.status_code}: {detail[:300]}")
        raise LLMError("gave up after retries")
