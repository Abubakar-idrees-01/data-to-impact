"""Optional API backend for any OpenAI-compatible chat endpoint.

Works with OpenAI, Groq, OpenRouter, Together, Mistral, Google's OpenAI-compatible Gemini
endpoint, a local Ollama or llama.cpp server... anything exposing POST /chat/completions.
Use it when the local model is too slow on your machine. Note: your question and the retrieved
passages are sent to that provider, so the app is no longer fully private.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Iterator

import httpx

from .base import Message, parse_json_loose, strip_thinking

# Some models (e.g. gpt-oss) write citations with CJK lenticular brackets: 【1】 -> [1]
_BRACKETS = str.maketrans({"【": "[", "】": "]"})


log = logging.getLogger(__name__)

# Free tiers limit tokens per minute (Groq free: 1,000 output tokens/min on some models), so a burst of calls
# (a summary, 10 MCQs) gets "429 Too Many Requests". Those, and temporary server errors, are retried after the
# wait the provider asks for.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 8
MAX_WAIT_S = 65.0                              # never sleep longer than this for a single retry


def retry_wait(r: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retrying: the provider's Retry-After header, the "try again in 1m5.8s" hint in
    its message, or exponential backoff."""
    header = r.headers.get("retry-after")
    if header:
        try:
            return min(float(header) + 0.5, MAX_WAIT_S)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", r.text)
    if m:
        return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5, MAX_WAIT_S)
    return min(2.0 ** attempt, MAX_WAIT_S)


def _check(r: httpx.Response) -> None:
    """Raise with the provider's own message (e.g. "model ... does not exist"), not just the status code."""
    if r.is_success:
        return
    try:
        detail = r.json().get("error", {}).get("message") or r.text
    except ValueError:
        detail = r.text
    raise RuntimeError(f"API error {r.status_code} from {r.request.url}: {detail[:300]}")


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 120.0):
        base_url, model, api_key = base_url.strip(), model.strip(), api_key.strip()   # pasted values
        self.name = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.timeout = timeout
        self.on_wait = None                       # optional callback(seconds) so a UI can show rate-limit waits

    def _notify(self, wait: float) -> None:
        if self.on_wait:
            try:
                self.on_wait(wait)
            except Exception:                     # a UI callback must never break generation
                log.debug("on_wait callback failed", exc_info=True)

    def _post(self, body: dict) -> httpx.Response:
        for attempt in range(MAX_RETRIES + 1):
            r = httpx.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
            if r.status_code not in RETRY_STATUS or attempt == MAX_RETRIES:
                return r
            wait = retry_wait(r, attempt)
            log.warning("API %s, retrying in %.1f s (attempt %d/%d)", r.status_code, wait, attempt + 1, MAX_RETRIES)
            self._notify(wait)
            time.sleep(wait)
        return r

    def _payload(self, messages, max_tokens, temperature, stream, json_mode=False) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "stream": stream}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.2,
                 stream: bool = False) -> str | Iterator[str]:
        body = self._payload(messages, max_tokens, temperature, stream)
        if not stream:
            r = self._post(body)
            _check(r)
            return strip_thinking(r.json()["choices"][0]["message"]["content"] or "").translate(_BRACKETS)
        return self._stream(body)

    def _stream(self, body: dict) -> Iterator[str]:
        for attempt in range(MAX_RETRIES + 1):
            with httpx.stream("POST", self.url, headers=self.headers, json=body, timeout=self.timeout) as r:
                if not r.is_success:
                    r.read()
                    if r.status_code in RETRY_STATUS and attempt < MAX_RETRIES:
                        wait = retry_wait(r, attempt)
                        log.warning("API %s, retrying in %.1f s", r.status_code, wait)
                        self._notify(wait)
                        time.sleep(wait)
                        continue
                    _check(r)
                for line in r.iter_lines():           # Server-Sent Events: "data: {...}"
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"].translate(_BRACKETS)
                return

    def generate_json(self, messages: list[Message], schema: dict, max_tokens: int = 768,
                      temperature: float = 0.3) -> dict:
        # Not every provider supports strict json_schema, so we put the schema in the prompt
        # and request generic JSON mode, then validate on our side.
        hint = {"role": "system", "content": "Reply with a single JSON object matching this schema:\n"
                + json.dumps(schema)}
        body = self._payload([hint, *messages], max_tokens, temperature, False, json_mode=True)
        r = self._post(body)
        if r.status_code == 400:                      # provider without JSON mode: retry plain
            body.pop("response_format")
            r = self._post(body)
        _check(r)
        return parse_json_loose(r.json()["choices"][0]["message"]["content"] or "")
