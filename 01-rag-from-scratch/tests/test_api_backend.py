"""The OpenAI-compatible backend: readable provider errors and citation-bracket normalisation (no network)."""

import httpx
import pytest

from studyrag.llm import openai_compat
from studyrag.llm.openai_compat import OpenAICompatLLM

URL = "https://api.example.com/v1/chat/completions"


def _fake_post(status, payload):
    def post(url, **kwargs):
        return httpx.Response(status, json=payload, request=httpx.Request("POST", url))
    return post


def test_provider_error_message_is_shown(monkeypatch):
    monkeypatch.setattr(openai_compat.httpx, "post", _fake_post(
        404, {"error": {"message": "The model `old-model` does not exist or you do not have access to it."}}))
    llm = OpenAICompatLLM("https://api.example.com/v1 ", " old-model ", "key")
    with pytest.raises(RuntimeError, match="old-model` does not exist"):
        llm.generate([{"role": "user", "content": "hi"}])
    assert llm.url == URL and llm.model == "old-model"          # pasted whitespace is stripped


def test_lenticular_citations_become_square_brackets(monkeypatch):
    monkeypatch.setattr(openai_compat.httpx, "post", _fake_post(
        200, {"choices": [{"message": {"content": "Open rate is a key metric【1】."}}]}))
    llm = OpenAICompatLLM("https://api.example.com/v1", "m", "key")
    assert llm.generate([{"role": "user", "content": "q"}]) == "Open rate is a key metric[1]."
