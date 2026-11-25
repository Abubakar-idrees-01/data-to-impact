"""MCP tools as agent tools, against a small in-process MCP server built with the official SDK."""

import asyncio

from mcp import Client
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as ServerToolError
from mcp_types import ToolAnnotations

from conftest import calls, text
from scout.agent import Agent
from scout.approval import DenyAll
from scout.mcp_tools import McpTools
from scout.tools import Registry

RO = ToolAnnotations(readOnlyHint=True)


def server():
    s = MCPServer("test-notes")

    @s.tool(annotations=RO)
    def search_notes(query: str, top_k: int = 3) -> dict:
        """Search my notes."""
        if query == "bad":
            raise ServerToolError("no such document")
        return {"found": True, "passages": [{"n": 1, "citation": "Lec 1 — Slide 2", "text": f"about {query}"}][:top_k]}

    @s.tool()
    def delete_note(name: str) -> str:
        """Delete a note (not marked read-only)."""
        return f"deleted {name}"
    return s


def test_mcp_tools_are_listed_and_called_through_the_agent(fake_api):
    async def go():
        async with McpTools([], client=Client(server())) as mcp:
            tools = {t.name: t for t in mcp.tools()}
            assert set(tools) == {"search_notes", "delete_note"}
            assert not tools["search_notes"].outward and tools["delete_note"].outward
            assert tools["search_notes"].parameters["required"] == ["query"]
            approver = DenyAll()
            api = fake_api([calls(("search_notes", {"query": "backlinks"}), ("delete_note", {"name": "x"}),
                                  ("search_notes", {"query": "bad"})), text("From your notes: backlinks [Lec 1 — Slide 2].")])
            r = await Agent(api.llm(), Registry(approver).add(*tools.values())).run("what are backlinks?")
            return r, approver
    r, approver = asyncio.run(go())
    results = [e["result"] for e in r.events if e["type"] == "tool"]
    assert '"citation": "Lec 1 — Slide 2"' in results[0]
    assert "declined" in results[1] and approver.seen[0].tool == "delete_note"     # not read-only -> asked, refused
    assert results[2].startswith("error:") and "no such document" in results[2] and r.tool_errors == 1


def test_include_filters_tools():
    async def go():
        async with McpTools([], client=Client(server()), include={"search_notes"}) as mcp:
            return [t.name for t in mcp.tools()]
    assert asyncio.run(go()) == ["search_notes"]
