"""MCP from scratch: the Model Context Protocol in ~250 lines of standard library.

This is the "how it works underneath" version. The real server (server.py) uses the official SDK;
this one exists to show, and test, what actually goes over the wire. It speaks enough MCP for real
clients (including the official SDK's client: see tests/test_scratch.py) to connect, list and call tools,
read resources and get prompts.

The wire format (stdio transport):
- one JSON-RPC 2.0 message per line on stdin/stdout, UTF-8, no embedded newlines;
- stdout is ONLY for protocol messages; logs go to stderr (a stray print() corrupts the stream).

The lifecycle:
    client -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":..., "capabilities":{},
               "clientInfo":{...}}}
    server -> {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":..., "capabilities":{"tools":{},...},
               "serverInfo":{...}}}
    client -> {"jsonrpc":"2.0","method":"notifications/initialized"}          (a notification: no id, no reply)
    client -> tools/list, tools/call, resources/list, resources/read, prompts/list, prompts/get, ping ...
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

# Protocol versions this implementation understands, newest first. If the client asks for one of these we
# answer with the same; otherwise we answer with our newest and the client decides whether to continue.
SUPPORTED_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"]

# JSON-RPC 2.0 error codes
PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS, INTERNAL_ERROR = -32700, -32600, -32601, -32602, -32603

_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}


class RPCError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., Any]


@dataclass
class Resource:
    uri: str
    name: str
    description: str
    mime_type: str
    read: Callable[[], str]


@dataclass
class Prompt:
    name: str
    description: str
    arguments: list[dict]                     # [{"name", "description", "required"}]
    render: Callable[..., str]


def validate(args: dict, schema: dict) -> None:
    """A deliberately small subset of JSON Schema: required keys, basic types, enums, numeric bounds.
    Enough to reject bad tool calls with a clear message before any work is done."""
    if not isinstance(args, dict):
        raise RPCError(INVALID_PARAMS, "arguments must be an object")
    props = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args]
    if missing:
        raise RPCError(INVALID_PARAMS, f"missing required argument(s): {', '.join(missing)}")
    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(props))
        if unknown:
            raise RPCError(INVALID_PARAMS, f"unknown argument(s): {', '.join(unknown)}")
    for key, value in args.items():
        spec = props.get(key, {})
        expected = _JSON_TYPES.get(spec.get("type"))
        is_bool = isinstance(value, bool)
        if expected and (not isinstance(value, expected) or (is_bool and spec.get("type") in ("integer", "number"))):
            raise RPCError(INVALID_PARAMS, f"'{key}' must be of type {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            raise RPCError(INVALID_PARAMS, f"'{key}' must be one of {spec['enum']}")
        if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
            raise RPCError(INVALID_PARAMS, f"'{key}' must be between {spec.get('minimum')} and {spec.get('maximum')}")
        if spec.get("type") == "string" and "minLength" in spec and len(value.strip()) < spec["minLength"]:
            raise RPCError(INVALID_PARAMS, f"'{key}' must not be empty")


@dataclass
class ScratchServer:
    name: str
    version: str = "0.1.0"
    instructions: str | None = None
    tools: dict[str, Tool] = field(default_factory=dict)
    resources: dict[str, Resource] = field(default_factory=dict)
    prompts: dict[str, Prompt] = field(default_factory=dict)
    initialized: bool = False

    # ------------------------------------------------------------------ registration (decorators)
    def tool(self, name: str, description: str, input_schema: dict):
        def deco(fn):
            self.tools[name] = Tool(name, description, input_schema, fn)
            return fn
        return deco

    def resource(self, uri: str, name: str, description: str, mime_type: str = "text/plain"):
        def deco(fn):
            self.resources[uri] = Resource(uri, name, description, mime_type, fn)
            return fn
        return deco

    def prompt(self, name: str, description: str, arguments: list[dict]):
        def deco(fn):
            self.prompts[name] = Prompt(name, description, arguments, fn)
            return fn
        return deco

    # ------------------------------------------------------------------ method handlers
    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
        caps = {}
        if self.tools:
            caps["tools"] = {"listChanged": False}
        if self.resources:
            caps["resources"] = {"listChanged": False, "subscribe": False}
        if self.prompts:
            caps["prompts"] = {"listChanged": False}
        result = {"protocolVersion": version, "capabilities": caps,
                  "serverInfo": {"name": self.name, "version": self.version}}
        if self.instructions:
            result["instructions"] = self.instructions
        return result

    def _tools_list(self, params: dict) -> dict:
        return {"tools": [{"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                          for t in self.tools.values()]}

    def _tools_call(self, params: dict) -> dict:
        name = params.get("name")
        if name not in self.tools:
            raise RPCError(INVALID_PARAMS, f"unknown tool: {name}")
        tool, args = self.tools[name], params.get("arguments") or {}
        validate(args, tool.input_schema)
        try:
            value = tool.fn(**args)
        except Exception as e:                # a failing tool is a RESULT with isError, so the model can react
            return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
        if isinstance(value, (dict, list)):
            return {"content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=1)}],
                    "structuredContent": value if isinstance(value, dict) else {"result": value}, "isError": False}
        return {"content": [{"type": "text", "text": str(value)}], "isError": False}

    def _resources_list(self, params: dict) -> dict:
        return {"resources": [{"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                              for r in self.resources.values()]}

    def _resources_read(self, params: dict) -> dict:
        uri = params.get("uri")
        if uri not in self.resources:
            raise RPCError(-32002, f"resource not found: {uri}")        # MCP's "resource not found" code
        r = self.resources[uri]
        return {"contents": [{"uri": uri, "mimeType": r.mime_type, "text": r.read()}]}

    def _prompts_list(self, params: dict) -> dict:
        return {"prompts": [{"name": p.name, "description": p.description, "arguments": p.arguments}
                            for p in self.prompts.values()]}

    def _prompts_get(self, params: dict) -> dict:
        name = params.get("name")
        if name not in self.prompts:
            raise RPCError(INVALID_PARAMS, f"unknown prompt: {name}")
        p, args = self.prompts[name], params.get("arguments") or {}
        missing = [a["name"] for a in p.arguments if a.get("required") and a["name"] not in args]
        if missing:
            raise RPCError(INVALID_PARAMS, f"missing prompt argument(s): {', '.join(missing)}")
        return {"description": p.description,
                "messages": [{"role": "user", "content": {"type": "text", "text": p.render(**args)}}]}

    # ------------------------------------------------------------------ dispatch
    def handle(self, message: dict) -> dict | None:
        """Process one JSON-RPC message; return the response, or None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "method" not in message:
            return _error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST,
                          "not a JSON-RPC 2.0 request")
        method, params, msg_id = message["method"], message.get("params") or {}, message.get("id")
        is_notification = "id" not in message
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None                                          # notifications never get a reply
        handlers = {"initialize": self._initialize, "ping": lambda p: {},
                    "tools/list": self._tools_list, "tools/call": self._tools_call,
                    "resources/list": self._resources_list, "resources/read": self._resources_read,
                    "resources/templates/list": lambda p: {"resourceTemplates": []},
                    "prompts/list": self._prompts_list, "prompts/get": self._prompts_get}
        if method not in handlers:
            return _error(msg_id, METHOD_NOT_FOUND, f"method not found: {method}")
        try:
            return {"jsonrpc": "2.0", "id": msg_id, "result": handlers[method](params)}
        except RPCError as e:
            return _error(msg_id, e.code, e.message, e.data)
        except Exception as e:                                   # a bug in the server, not the client's fault
            traceback.print_exc(file=sys.stderr)
            return _error(msg_id, INTERNAL_ERROR, f"internal error: {e}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        """The stdio transport: read a line, handle it, write the reply as one line. Ends at EOF."""
        for line in stdin:
            if not line.strip():
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as e:
                reply = _error(None, PARSE_ERROR, f"parse error: {e}")
            else:
                reply = self.handle(message)
            if reply is not None:
                stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
                stdout.flush()


def _error(msg_id, code: int, message: str, data: Any = None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def demo_server() -> ScratchServer:
    """A tiny server for tests and for reading along in docs/concepts: two tools, one resource, one prompt."""
    s = ScratchServer("scratch-demo", instructions="A from-scratch MCP server for learning.")

    @s.tool("add", "Add two integers.", {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                                         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}})
    def add(a: int, b: int) -> dict:
        return {"sum": a + b}

    @s.tool("word_count", "Count the words in a text.",
            {"type": "object", "required": ["text"], "properties": {"text": {"type": "string", "minLength": 1}}})
    def word_count(text: str) -> str:
        return f"{len(text.split())} words"

    @s.tool("divide", "Divide a by b.", {"type": "object", "required": ["a", "b"],
                                         "properties": {"a": {"type": "number"}, "b": {"type": "number"}}})
    def divide(a: float, b: float) -> dict:
        return {"result": a / b}                         # b = 0 -> a tool error result, not a crash

    @s.resource("demo://readme", "readme", "What this server is", "text/markdown")
    def readme() -> str:
        return "# scratch-demo\nAn MCP server in ~250 lines of standard-library Python."

    @s.prompt("explain", "Ask for a beginner explanation.", [{"name": "topic", "required": True}])
    def explain(topic: str) -> str:
        return f"Explain {topic} to a beginner in three sentences."

    return s


if __name__ == "__main__":                                # python -m studyrag_mcp.scratch
    demo_server().serve()
