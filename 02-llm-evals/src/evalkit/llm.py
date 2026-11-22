"""A minimal chat client for any OpenAI-compatible API (Groq, OpenAI, OpenRouter, a local llama.cpp/Ollama server).

Standard library only. Three things matter for evaluation work:
1. **Disk cache.** The same request (model, messages, settings) is answered from `.cache/llm/` without an API
   call. Re-running an experiment, or a report, costs nothing and gives identical results.
2. **Rate limits.** Free tiers allow a few thousand tokens per minute. On 429/5xx we wait as long as the
   provider asks (Retry-After header or "try again in 5.9s" in the message) and retry.
3. **Accounting.** Calls, cache hits, tokens and time are counted, so cost can be reported.

Configuration from environment variables (or a `.env` file):
    EVALKIT_API_BASE_URL=https://api.groq.com/openai/v1
    EVALKIT_API_MODEL=qwen/qwen3.8-27b
    EVALKIT_API_KEY=...
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def load_dotenv(path: str | Path = ".env", override: bool = False) -> None:
    """Read KEY=VALUE lines into os.environ (no dependency on python-dotenv)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip("'\"")
        if override or key not in os.environ:
            os.environ[key] = value


def retry_wait(status: int, headers: dict, body: str, attempt: int, max_wait: float = 65.0) -> float:
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


class APIError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str
    cached: bool = False
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    waits: list[float] = field(default_factory=list)


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0,
                 cache_dir: str | Path | None = ".cache/llm", max_retries: int = 8,
                 sleep: Callable[[float], None] = time.sleep):
        self.url = base_url.strip().rstrip("/") + "/chat/completions"
        self.model = model.strip()
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        self.sleep = sleep
        self.usage = Usage()

    @classmethod
    def from_env(cls, prefix: str = "EVALKIT_", dotenv: str | Path = ".env", **kwargs) -> "ChatClient":
        load_dotenv(dotenv)
        try:
            base = os.environ[f"{prefix}API_BASE_URL"]
            model = os.environ[f"{prefix}API_MODEL"]
        except KeyError as e:
            raise APIError(f"set {e.args[0]} (see docs/SETUP.md)") from None
        return cls(base, kwargs.pop("model", model), os.environ.get(f"{prefix}API_KEY", ""), **kwargs)

    # ------------------------------------------------------------------ cache
    def _key(self, body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        return self.cache_dir / key[:2] / f"{key}.json" if self.cache_dir else None

    # ------------------------------------------------------------------ calls
    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
             json_mode: bool = False, use_cache: bool = True) -> ChatResult:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        path = self._cache_path(self._key(body))
        if use_cache and path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.usage.cache_hits += 1
            return ChatResult(text=data["text"], cached=True, prompt_tokens=data.get("prompt_tokens", 0),
                              completion_tokens=data.get("completion_tokens", 0))
        t0 = time.time()
        resp = self._post(body)
        msg = resp["choices"][0]["message"]
        text = _THINK.sub("", msg.get("content") or "").strip()
        usage = resp.get("usage") or {}
        result = ChatResult(text=text, seconds=time.time() - t0, prompt_tokens=usage.get("prompt_tokens", 0),
                            completion_tokens=usage.get("completion_tokens", 0))
        self.usage.calls += 1
        self.usage.prompt_tokens += result.prompt_tokens
        self.usage.completion_tokens += result.completion_tokens
        self.usage.seconds += result.seconds
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text, "prompt_tokens": result.prompt_tokens,
                                        "completion_tokens": result.completion_tokens, "model": self.model}),
                            encoding="utf-8")
        return result

    def _post(self, body: dict) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "User-Agent": "evalkit/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                if e.code in RETRY_STATUS and attempt < self.max_retries:
                    wait = retry_wait(e.code, dict(e.headers or {}), text, attempt)
                    log.warning("API %s, waiting %.1f s (attempt %d/%d)", e.code, wait, attempt + 1, self.max_retries)
                    self.usage.waits.append(wait)
                    self.sleep(wait)
                    continue
                try:
                    detail = json.loads(text).get("error", {}).get("message") or text
                except (ValueError, AttributeError):
                    detail = text
                raise APIError(f"API error {e.code}: {detail[:300]}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries:
                    wait = min(2.0 ** attempt, 30.0)
                    log.warning("network error (%s), retrying in %.0f s", e, wait)
                    self.sleep(wait)
                    continue
                raise APIError(f"network error: {e}") from None
        raise APIError("unreachable")


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences and text around it."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
