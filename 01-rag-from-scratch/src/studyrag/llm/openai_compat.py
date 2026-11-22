"""Optional API backend for any OpenAI-compatible chat endpoint.

Works with OpenAI, Groq, OpenRouter, Together, Mistral, Google's OpenAI-compatible Gemini
endpoint, a local Ollama or llama.cpp server... anything exposing POST /chat/completions.
Use it when the local model is too slow on your machine. Note: your question and the retrieved
passages are sent to that provider, so the app is no longer fully private.
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx

from .base import Message, parse_json_loose, strip_thinking

# Some models (e.g. gpt-oss) write citations with CJK lenticular brackets: 【1】 -> [1]
_BRACKETS = str.maketrans({"【": "[", "】": "]"})


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
            r = httpx.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
            _check(r)
            return strip_thinking(r.json()["choices"][0]["message"]["content"] or "").translate(_BRACKETS)
        return self._stream(body)

    def _stream(self, body: dict) -> Iterator[str]:
        with httpx.stream("POST", self.url, headers=self.headers, json=body, timeout=self.timeout) as r:
            if not r.is_success:
                r.read()
                _check(r)
            for line in r.iter_lines():               # Server-Sent Events: "data: {...}"
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"].translate(_BRACKETS)

    def generate_json(self, messages: list[Message], schema: dict, max_tokens: int = 768,
                      temperature: float = 0.3) -> dict:
        # Not every provider supports strict json_schema, so we put the schema in the prompt
        # and request generic JSON mode, then validate on our side.
        hint = {"role": "system", "content": "Reply with a single JSON object matching this schema:\n"
                + json.dumps(schema)}
        body = self._payload([hint, *messages], max_tokens, temperature, False, json_mode=True)
        r = httpx.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
        if r.status_code == 400:                      # provider without JSON mode: retry plain
            body.pop("response_format")
            r = httpx.post(self.url, headers=self.headers, json=body, timeout=self.timeout)
        _check(r)
        return parse_json_loose(r.json()["choices"][0]["message"]["content"] or "")
