"""LLM interface shared by all backends.

Every backend offers the same two calls:
- generate(messages, ...)      -> full text, or an iterator of text pieces when stream=True
- generate_json(messages, schema) -> a parsed dict that matches the JSON schema

Keeping the interface tiny means the rest of the app never cares whether the model runs
locally (llama.cpp) or behind an API.
"""

from __future__ import annotations

import json
import re
from typing import Iterator, Protocol

Message = dict[str, str]          # {"role": "system"|"user"|"assistant", "content": "..."}


class LLM(Protocol):
    name: str

    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.2,
                 stream: bool = False) -> str | Iterator[str]: ...

    def generate_json(self, messages: list[Message], schema: dict, max_tokens: int = 768,
                      temperature: float = 0.3) -> dict: ...


_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_thinking(text: str) -> str:
    """Remove any <think>...</think> reasoning a model emitted despite thinking being disabled."""
    return _THINK.sub("", text).strip()


def parse_json_loose(text: str) -> dict:
    """Parse JSON from model output, tolerating code fences or chatter around the object."""
    text = strip_thinking(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
