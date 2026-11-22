"""LLM client (cache, retries, errors), judge parsing and the non-LLM baselines. No network: urlopen is faked."""

import io
import json
import urllib.error

import pytest

from evalkit import baselines, llm
from evalkit.judge import Judge
from evalkit.llm import APIError, ChatClient, parse_json, retry_wait


class FakeAPI:
    """Stands in for urllib.request.urlopen: returns queued responses, records request bodies."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies = []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": r}}],
                                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode())


def http_error(code, body, headers=None):
    return urllib.error.HTTPError("u", code, "err", headers or {}, io.BytesIO(body.encode()))


@pytest.fixture
def client(tmp_path):
    return ChatClient("https://api.example.com/v1/", " m ", "k", cache_dir=tmp_path / "cache", sleep=lambda s: None)


def test_cache_returns_identical_answer_without_a_call(monkeypatch, client):
    api = FakeAPI("first")
    monkeypatch.setattr(llm.urllib.request, "urlopen", api)
    a = client.chat([{"role": "user", "content": "hi"}])
    b = client.chat([{"role": "user", "content": "hi"}])
    assert (a.text, a.cached, b.text, b.cached) == ("first", False, "first", True)
    assert len(api.bodies) == 1 and client.usage.calls == 1 and client.usage.cache_hits == 1
    assert api.bodies[0]["model"] == "m" and client.url == "https://api.example.com/v1/chat/completions"


def test_rate_limit_waits_then_succeeds(monkeypatch, client):
    waits = []
    client.sleep = waits.append
    api = FakeAPI(http_error(429, '{"error": {"message": "Rate limit. Please try again in 5.88s."}}'), "ok")
    monkeypatch.setattr(llm.urllib.request, "urlopen", api)
    assert client.chat([{"role": "user", "content": "x"}]).text == "ok"
    assert waits == [pytest.approx(6.38)]


def test_other_errors_show_the_provider_message(monkeypatch, client):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        FakeAPI(http_error(404, '{"error": {"message": "The model `m` does not exist"}}')))
    with pytest.raises(APIError, match="does not exist"):
        client.chat([{"role": "user", "content": "x"}])


def test_retry_wait_and_json_parsing():
    assert retry_wait(429, {"Retry-After": "2"}, "", 0) == 2.5
    assert retry_wait(429, {}, "try again in 1m2.5s", 0) == 63.0
    assert retry_wait(503, {}, "", 3) == 8.0
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"verdict": "supported"} hope this helps') == {"verdict": "supported"}


def test_think_blocks_are_removed(monkeypatch, client):
    monkeypatch.setattr(llm.urllib.request, "urlopen", FakeAPI('<think>hmm</think>\n{"verdict": "supported"}'))
    assert client.chat([{"role": "user", "content": "x"}]).text == '{"verdict": "supported"}'


def test_faithfulness_judge(monkeypatch, client):
    reply = {"problems": [{"quote": "in 1999", "type": "contradiction", "why": "source says 1998"}],
             "verdict": "hallucinated"}
    api = FakeAPI(json.dumps(reply), '{"problems": [], "verdict": "supported"}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", api)
    judge = Judge(client, "faithfulness/claims")
    v = judge(task="Summarize.", source="Founded in 1998.", response="It was founded in 1999.")
    assert v.label is True and v.details["problems"][0]["quote"] == "in 1999"
    assert "Founded in 1998." in api.bodies[0]["messages"][1]["content"]
    assert api.bodies[0]["response_format"] == {"type": "json_object"} and api.bodies[0]["temperature"] == 0.0
    assert judge(task="t", source="s", response="r").label is False


def test_unparseable_output_is_retried_without_cache_then_reported(monkeypatch, client):
    monkeypatch.setattr(llm.urllib.request, "urlopen", FakeAPI("not json", '{"verdict": "maybe"}'))
    v = Judge(client, "faithfulness/direct")(task="t", source="s", response="r")
    assert v.label is None and "unparseable" in v.error


def test_correctness_judge(monkeypatch, client):
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        FakeAPI('{"missing": ["CPC"], "contradictions": [], "verdict": "partially correct"}'))
    v = Judge(client, "correctness")(question="q", reference="r", response="a")
    assert v.label == 0.5 and v.details["missing"] == ["CPC"]


def test_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nEVALKIT_API_BASE_URL=https://x/v1\nexport EVALKIT_API_MODEL='mm'\n")
    monkeypatch.delenv("EVALKIT_API_BASE_URL", raising=False)
    monkeypatch.delenv("EVALKIT_API_MODEL", raising=False)
    c = ChatClient.from_env(dotenv=env, cache_dir=None)
    assert c.model == "mm" and c.url == "https://x/v1/chat/completions"


def test_baselines():
    source = "The company was founded in 1998 in Paris and sells bicycles."
    assert not baselines.lexical_hallucination("The company sells bicycles.", source)
    assert baselines.lexical_hallucination("Its chief executive enjoys sailing on weekends.", source)
    assert baselines.novel_numbers("Founded in 1998 with 1,200 staff.", source) == ["1,200"]
    assert not baselines.numbers_hallucination("Founded in 1998.", source)


def test_provider_json_rejection_falls_back_to_plain_mode(monkeypatch, client):
    api = FakeAPI(http_error(400, '{"error": {"message": "Failed to validate JSON."}}'), '{"verdict": "supported"}')
    monkeypatch.setattr(llm.urllib.request, "urlopen", api)
    v = Judge(client, "faithfulness/direct")(task="t", source="s", response="r")
    assert v.label is False and "response_format" not in api.bodies[1]
