"""A scripted stand-in for the model API: each request gets the next canned reply. No key, no network."""

from __future__ import annotations

import json

import httpx
import pytest

from scout.llm import LLM


def text(content: str, prompt=100, completion=20) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def calls(*items: tuple[str, dict], prompt=100, completion=20) -> dict:
    tc = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
          for i, (n, a) in enumerate(items)]
    return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": tc},
                         "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


class FakeAPI:
    """Replies in order; records every request body. Items may be dicts (200) or (status, body, headers)."""

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        item = self.replies.pop(0)
        if isinstance(item, tuple):
            status, body, headers = item
            return httpx.Response(status, json=body, headers=headers)
        return httpx.Response(200, json=item)

    def llm(self, model="openai/gpt-oss-120b", **kw) -> LLM:
        async def no_sleep(_):
            return None
        return LLM("https://api.example/v1", model, "k", reasoning_effort="low", sleep=no_sleep,
                   transport=httpx.MockTransport(self.handler), **kw)


@pytest.fixture
def fake_api():
    return FakeAPI
