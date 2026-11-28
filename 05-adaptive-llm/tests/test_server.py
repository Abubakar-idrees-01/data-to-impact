"""The OpenAI-compatible server: tool-call parsing for both Qwen formats, prompt rendering, the HTTP round trip."""

import json
import threading
import urllib.request
from http.server import HTTPServer

from adaptive.server import ChatEngine, make_handler, parse_tool_calls, render

TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {
    "query": {"type": "string"}, "limit": {"type": "integer"}, "fresh": {"type": "boolean"}}}}}]

TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "{% for c in m.tool_calls or [] %}<tool_call>{{ c.function.arguments | tojson }}</tool_call>{% endfor %}"
            "<|im_end|>\n{% endfor %}{% if tools %}TOOLS:{{ tools | tojson }}{% endif %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% if not enable_thinking %}<think>\n\n</think>\n\n"
            "{% endif %}{% endif %}")


def test_parse_qwen35_xml_tool_call_with_types():
    text = ("Let me search.\n<tool_call>\n<function=web_search>\n<parameter=query>\ncapital of Peru\n</parameter>\n"
            "<parameter=limit>\n3\n</parameter>\n<parameter=fresh>\ntrue\n</parameter>\n</function>\n</tool_call>")
    rest, calls = parse_tool_calls(text, TOOLS)
    assert rest == "Let me search."
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "capital of Peru", "limit": 3, "fresh": True}


def test_parse_qwen25_json_tool_calls_and_plain_text():
    text = ('<tool_call>\n{"name": "web_search", "arguments": {"query": "a"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "web_search", "arguments": "{\\"query\\": \\"b\\"}"}\n</tool_call>')
    rest, calls = parse_tool_calls(text, TOOLS)
    assert rest == "" and [json.loads(c["function"]["arguments"])["query"] for c in calls] == ["a", "b"]
    assert parse_tool_calls("<think>\nhm\n</think>\n\nJust an answer.") == ("Just an answer.", [])
    broken = parse_tool_calls("<tool_call>\n{broken json\n</tool_call>")
    assert broken[1] == [] and "{broken json" in broken[0]                    # not a crash, and not swallowed
    # Qwen2.5's template shows the format with doubled braces, and the 1.5B copies them (seen with Scout)
    doubled = ('<tool_call>\n{{"name": "web_search", "arguments": {"query": "supervised learning", "limit": 4, '
               '"fresh": null}}}}\n</tool_call>')
    rest, calls = parse_tool_calls(doubled, TOOLS)
    assert rest == "" and json.loads(calls[0]["function"]["arguments"])["query"] == "supervised learning"
    assert parse_tool_calls("<tool_call>\n<function=web_search>\n<parameter=query>\nx")[1][0]["function"]["name"] \
        == "web_search"                                                                   # cut off by max_tokens


def test_render_turns_json_string_arguments_into_dicts():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "web_search", "arguments": '{"query": "x"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "r"}]
    p = render(TEMPLATE, msgs, TOOLS)
    assert '<tool_call>{"query": "x"}</tool_call>' in p and p.endswith("<think>\n\n</think>\n\n")
    assert 'TOOLS:[{"type": "function"' in p


class FakeLLM:
    def __init__(self, reply):
        self.metadata = {"tokenizer.chat_template": TEMPLATE}
        self.reply, self.prompts = reply, []

    def create_completion(self, prompt, **kw):
        self.prompts.append(prompt)
        return {"choices": [{"text": self.reply, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class FakeModel:
    def __init__(self, name, reply):
        self.name, self.llm = name, FakeLLM(reply)


class FakeLadder:
    def __init__(self, replies):
        self.names = list(replies)
        self.models = [FakeModel(n, r) for n, r in replies.items()]

    def get(self, i):
        return self.models[i]


LADDER = {"small": "Lima.", "mid": "42", "big": '<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'}


def test_engine_picks_models_and_returns_openai_shape():
    e = ChatEngine(FakeLadder(LADDER))
    r = e.chat({"model": "auto", "messages": [{"role": "user", "content": "Capital of Peru?"}]})
    assert r["model"] == "small" and r["choices"][0]["message"]["content"] == "Lima."
    assert r["choices"][0]["finish_reason"] == "stop" and r["usage"]["total_tokens"] == 15
    r = e.chat({"model": "auto", "messages": [{"role": "user", "content": "find x"}], "tools": TOOLS})
    assert r["model"] == "big" and r["choices"][0]["finish_reason"] == "tool_calls"       # tools -> largest model
    assert r["choices"][0]["message"]["content"] is None
    assert e.chat({"model": "mid", "messages": [{"role": "user", "content": "hi"}]})["model"] == "mid"


def test_http_round_trip():
    srv = HTTPServer(("127.0.0.1", 0), make_handler(ChatEngine(FakeLadder(LADDER))))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/v1"
    try:
        models = json.load(urllib.request.urlopen(base + "/models"))
        assert [m["id"] for m in models["data"]] == ["auto", "small", "mid", "big"]
        req = urllib.request.Request(base + "/chat/completions", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "auto", "messages": [{"role": "user", "content": "Capital?"}]}).encode())
        assert json.load(urllib.request.urlopen(req))["choices"][0]["message"]["content"] == "Lima."
    finally:
        srv.shutdown()
