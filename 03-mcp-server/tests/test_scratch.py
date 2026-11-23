"""The from-scratch MCP server: raw JSON-RPC behaviour, and conformance with the official SDK's client."""

import asyncio
import io
import json
import os
import sys

from mcp import Client, StdioServerParameters

from studyrag_mcp.scratch import INVALID_PARAMS, METHOD_NOT_FOUND, PARSE_ERROR, SUPPORTED_VERSIONS, demo_server


def rpc(server, method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return server.handle(msg)


def test_handshake_negotiates_version_and_capabilities():
    s = demo_server()
    r = rpc(s, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                              "clientInfo": {"name": "t", "version": "1"}})["result"]
    assert r["protocolVersion"] == "2025-06-18"
    assert set(r["capabilities"]) == {"tools", "resources", "prompts"} and r["serverInfo"]["name"] == "scratch-demo"
    unknown = rpc(s, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert unknown["protocolVersion"] == SUPPORTED_VERSIONS[0]      # counter-offer our newest
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None and s.initialized


def test_tools_list_and_call():
    s = demo_server()
    names = [t["name"] for t in rpc(s, "tools/list")["result"]["tools"]]
    assert names == ["add", "word_count", "divide"]
    r = rpc(s, "tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}})["result"]
    assert r["structuredContent"] == {"sum": 5} and not r["isError"] and json.loads(r["content"][0]["text"]) == {"sum": 5}


def test_validation_errors_are_protocol_errors_but_tool_failures_are_results():
    s = demo_server()
    err = rpc(s, "tools/call", {"name": "add", "arguments": {"a": 2}})["error"]
    assert err["code"] == INVALID_PARAMS and "missing required" in err["message"]
    assert "must be of type integer" in rpc(s, "tools/call", {"name": "add", "arguments": {"a": "2", "b": 1}})["error"]["message"]
    assert "must be of type integer" in rpc(s, "tools/call", {"name": "add", "arguments": {"a": True, "b": 1}})["error"]["message"]
    assert "unknown argument" in rpc(s, "tools/call", {"name": "add", "arguments": {"a": 1, "b": 1, "c": 1}})["error"]["message"]
    assert rpc(s, "tools/call", {"name": "nope", "arguments": {}})["error"]["code"] == INVALID_PARAMS
    # the tool ran and failed: the MODEL should see this and react, so it is a result with isError
    r = rpc(s, "tools/call", {"name": "divide", "arguments": {"a": 1, "b": 0}})["result"]
    assert r["isError"] and "ZeroDivisionError" in r["content"][0]["text"]


def test_resources_prompts_ping_and_unknown_methods():
    s = demo_server()
    assert rpc(s, "resources/list")["result"]["resources"][0]["uri"] == "demo://readme"
    assert "scratch-demo" in rpc(s, "resources/read", {"uri": "demo://readme"})["result"]["contents"][0]["text"]
    assert rpc(s, "resources/read", {"uri": "demo://nope"})["error"]["code"] == -32002
    msg = rpc(s, "prompts/get", {"name": "explain", "arguments": {"topic": "MCP"}})["result"]["messages"][0]
    assert msg["role"] == "user" and "MCP" in msg["content"]["text"]
    assert rpc(s, "prompts/get", {"name": "explain", "arguments": {}})["error"]["code"] == INVALID_PARAMS
    assert rpc(s, "ping")["result"] == {}
    assert rpc(s, "server/discover")["error"]["code"] == METHOD_NOT_FOUND


def test_stdio_framing_one_message_per_line():
    s = demo_server()
    stdin = io.StringIO('{"jsonrpc":"2.0","id":1,"method":"ping"}\n\nnot json\n'
                        '{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
    stdout = io.StringIO()
    s.serve(stdin, stdout)
    lines = stdout.getvalue().splitlines()
    assert len(lines) == 2                                   # ping reply + parse error; notification: no reply
    assert json.loads(lines[0]) == {"jsonrpc": "2.0", "id": 1, "result": {}}
    assert json.loads(lines[1])["error"]["code"] == PARSE_ERROR


def test_official_sdk_client_talks_to_the_scratch_server():
    """Conformance: the official MCP client, over a real stdio subprocess, against our hand-written server."""
    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "studyrag_mcp.scratch"],
                                       env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        async with Client(params) as c:
            tools = await c.list_tools()
            added = await c.call_tool("add", {"a": 20, "b": 22})
            failed = await c.call_tool("divide", {"a": 1, "b": 0})
            res = await c.read_resource("demo://readme")
            prompt = await c.get_prompt("explain", {"topic": "JSON-RPC"})
            return tools, added, failed, res, prompt
    tools, added, failed, res, prompt = asyncio.run(run())
    assert [t.name for t in tools.tools] == ["add", "word_count", "divide"]
    assert added.structured_content == {"sum": 42} and not added.is_error
    assert failed.is_error
    assert "scratch-demo" in res.contents[0].text
    assert "JSON-RPC" in prompt.messages[0].content.text
