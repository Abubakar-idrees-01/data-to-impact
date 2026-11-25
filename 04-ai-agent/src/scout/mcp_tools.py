"""Use any MCP server's tools as agent tools (project 3's studyrag-mcp by default).

The agent is an MCP *client*: it launches the server as a subprocess over stdio, asks it which tools it has
(`tools/list`), and turns each one into a `Tool` whose function forwards the call (`tools/call`). The tool's
JSON Schema comes from the server, so nothing about the server is hard-coded here.

Safety: MCP tools declare hints (`readOnlyHint`). A tool that doesn't declare itself read-only is treated as
**outward** and needs your approval before each call, so plugging in a new server can't quietly give the model
write access to anything.

    async with McpTools(["/path/to/python", "-m", "studyrag_mcp"]) as mcp:
        registry.add(*mcp.tools())
"""

from __future__ import annotations

import json
import os

from mcp import Client, StdioServerParameters

from .approval import Action
from .tools import Tool, ToolError

PASS_ENV = ("PATH", "HOME", "LANG", "STUDYRAG_HOME", "STUDYRAG_STORAGE_DIR", "STUDYRAG_MCP_EVAL_RESULTS", "HF_HOME")


class McpTools:
    def __init__(self, command: list[str], *, env: dict | None = None, include: set[str] | None = None,
                 timeout: float = 180.0, client: Client | None = None):
        self.command = command
        self.env = env if env is not None else {k: os.environ[k] for k in PASS_ENV if k in os.environ}
        self.include = include              # only expose these tool names (None: all)
        self.timeout = timeout
        self._client = client
        self._listed: list = []
        self.server_name = ""

    async def __aenter__(self) -> "McpTools":
        if self._client is None:
            params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=self.env)
            self._client = Client(params, read_timeout_seconds=self.timeout)
        await self._client.__aenter__()
        self._listed = (await self._client.list_tools()).tools
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.__aexit__(*exc)

    def tools(self) -> list[Tool]:
        out = []
        for t in self._listed:
            if self.include is not None and t.name not in self.include:
                continue
            read_only = bool(t.annotations and t.annotations.read_only_hint)
            schema = dict(t.input_schema or {"type": "object", "properties": {}})
            schema.pop("title", None)
            out.append(Tool(t.name, (t.description or t.name).strip(), schema, self._caller(t.name),
                            outward=not read_only,
                            preview=None if read_only else self._preview(t.name)))
        return out

    def _caller(self, name: str):
        async def call(**args):
            r = await self._client.call_tool(name, args)
            text = "\n".join(getattr(c, "text", "") for c in r.content if getattr(c, "text", None))
            if r.is_error:
                raise ToolError(text or f"{name} failed")
            return r.structured_content if r.structured_content is not None else text
        return call

    @staticmethod
    def _preview(name: str):
        return lambda args: Action(name, f"call the MCP tool {name} (not marked read-only)",
                                   json.dumps(args, indent=1, ensure_ascii=False))
