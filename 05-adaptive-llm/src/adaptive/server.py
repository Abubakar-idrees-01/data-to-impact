"""An OpenAI-compatible chat server over the local models, so other programs can use the engine offline.

    python -m adaptive.server --port 8090
    # project 4's agent, fully offline:
    SCOUT_API_BASE_URL=http://127.0.0.1:8090/v1 SCOUT_API_MODEL=auto python -m scout ask "..."

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (messages, tools, max_tokens, temperature; no streaming).

- **Model choice.** `model` = a ladder model's name (or part of it), or `auto`: requests with tools go to the
  largest model (agents need the strongest one), others are routed by the rule router (concept 6).
- **Prompt.** Rendered with the model's *own* chat template from its GGUF file (Jinja2), so tool descriptions,
  tool calls and tool results are laid out exactly as the model was trained to read them. Thinking is turned off
  (`enable_thinking=False`), as everywhere in this project.
- **Tool calls.** The two Qwen families write them differently, and both are parsed into OpenAI `tool_calls`:
  Qwen2.5 `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`; Qwen3.5
  `<tool_call><function=name><parameter=p>value</parameter></function></tool_call>`, whose string values are
  converted to the types the tool's JSON schema declares.

Standard library HTTP server, one request at a time (one CPU-bound model can't serve two at once anyway).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .hardware import profile
from .router import features, rule_route
from .runtime import CHATML_STOP, Ladder

ROOT = Path(__file__).resolve().parents[2]
TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
THINK = re.compile(r"<think>.*?</think>\s*", re.S)


# ---------- prompt
def render(template: str, messages: list[dict], tools: list[dict] | None) -> str:
    """The model's own chat template, as Hugging Face / llama.cpp render it."""
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(msg):
        raise ValueError(msg)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(x, ensure_ascii=ensure_ascii,
                                                                                  indent=indent)
    msgs = []
    for m in messages:                        # OpenAI sends tool-call arguments as a JSON string; templates want a dict
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function", c))
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except ValueError:
                        fn["arguments"] = {}
                calls.append({**c, "function": fn})
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        msgs.append(m)
    return env.from_string(template).render(messages=msgs, tools=tools or None, add_generation_prompt=True,
                                            enable_thinking=False, raise_exception=raise_exception,
                                            bos_token="", eos_token="<|im_end|>")


# ---------- tool calls
def _coerce(value: str, schema: dict):
    kind = schema.get("type")
    if kind in ("integer", "number", "boolean", "object", "array"):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def parse_tool_calls(text: str, tools: list[dict] | None = None) -> tuple[str, list[dict]]:
    """(remaining text, OpenAI-style tool calls) from a Qwen reply."""
    schemas = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {})
               for t in tools or [] if "function" in t}
    calls = []
    for block in TOOL_BLOCK.findall(text):
        name, args = None, {}
        if block.startswith("{"):                                       # Qwen2.5: JSON
            try:
                obj = json.loads(block)
                name, args = obj.get("name"), obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args or "{}")
            except ValueError:
                continue
        else:                                                           # Qwen3.5: XML-like
            fn = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|\Z)", block, re.S)
            if not fn:
                continue
            name = fn.group(1).strip()
            props = schemas.get(name, {})
            for p, v in re.findall(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", fn.group(2), re.S):
                args[p.strip()] = _coerce(v, props.get(p.strip(), {}))
        if name:
            calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    rest = TOOL_BLOCK.sub("", THINK.sub("", text)).strip()
    return rest, calls


# ---------- the engine behind the endpoint
class ChatEngine:
    def __init__(self, ladder: Ladder):
        self.ladder = ladder

    def pick(self, requested: str, messages: list[dict], tools) -> int:
        names = [n.lower() for n in self.ladder.names]
        if requested and requested != "auto":
            for i, n in enumerate(names):
                if requested.lower() in n or n in requested.lower():
                    return i
        if tools:
            return len(names) - 1
        return rule_route(features(messages), len(names))

    def chat(self, body: dict) -> dict:
        messages, tools = body["messages"], body.get("tools")
        rung = self.pick(body.get("model", "auto"), messages, tools)
        m = self.ladder.get(rung)
        prompt = render(m.llm.metadata["tokenizer.chat_template"], messages, tools)
        t0 = time.perf_counter()
        out = m.llm.create_completion(prompt, max_tokens=int(body.get("max_tokens") or 1024),
                                      temperature=float(body.get("temperature") or 0.0), top_p=0.9,
                                      repeat_penalty=1.05, stop=CHATML_STOP)
        text = out["choices"][0]["text"]
        content, calls = parse_tool_calls(text, tools)
        usage = out.get("usage", {})
        finish = "tool_calls" if calls else out["choices"][0].get("finish_reason") or "stop"
        msg = {"role": "assistant", "content": content or None}
        if calls:
            msg["tool_calls"] = calls
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:16]}", "object": "chat.completion", "created": int(time.time()),
                "model": m.name, "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                          "completion_tokens": usage.get("completion_tokens", 0),
                          "total_tokens": usage.get("total_tokens", 0)},
                "x_seconds": round(time.perf_counter() - t0, 2)}


def make_handler(engine: ChatEngine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [{"id": "auto", "object": "model"}] +
                                        [{"id": n, "object": "model"} for n in engine.ladder.names]})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("stream"):
                    return self._send(400, {"error": {"message": "streaming is not supported by this server"}})
                self._send(200, engine.chat(body))
            except (ValueError, KeyError) as e:
                self._send(400, {"error": {"message": f"bad request: {e}"}})

        def log_message(self, fmt, *args):            # one short line per request
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this machine only")
    ap.add_argument("--models", default=str(ROOT.parent / "01-rag-from-scratch" / "models" / "llm"))
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()
    hw = profile(bandwidth=False)
    ladder = Ladder(sorted(Path(a.models).glob("*.gguf")), budget_gb=hw.ram_available_gb - 1.0, n_ctx=a.ctx,
                    threads=hw.physical_cores)
    print(f"models: {', '.join(ladder.names)} | budget {hw.ram_available_gb - 1.0:.1f} GB | "
          f"http://{a.host}:{a.port}/v1", flush=True)
    HTTPServer((a.host, a.port), make_handler(ChatEngine(ladder))).serve_forever()


if __name__ == "__main__":
    main()
