"""The real server (official SDK), driven by the official client in-process, with fake models."""

import asyncio
import json

from mcp import Client

from studyrag_mcp.server import build_server


def call(server, tool, args=None):
    async def run():
        async with Client(server) as c:
            return await c.call_tool(tool, args or {})
    return asyncio.run(run())


def test_lists_tools_with_read_only_annotations(notes, evals):
    async def run():
        async with Client(build_server(notes, evals)) as c:
            return (await c.list_tools()).tools
    tools = {t.name: t for t in asyncio.run(run())}
    assert set(tools) == {"search_notes", "list_documents", "document_outline", "get_passage", "check_answer",
                          "list_eval_runs", "eval_run_summary", "compare_eval_runs"}
    assert all(t.annotations.read_only_hint for t in tools.values())
    schema = tools["search_notes"].input_schema
    assert schema["required"] == ["query"] and schema["properties"]["top_k"]["maximum"] == 10


def test_search_returns_cited_passages(notes, evals):
    r = call(build_server(notes, evals), "search_notes", {"query": "what are backlinks", "top_k": 2})
    s = r.structured_content
    assert not r.is_error and s["found"] and len(s["passages"]) == 2
    top = s["passages"][0]
    assert top["n"] == 1 and "Backlinks" in top["citation"] and "votes of trust" in top["text"] and top["chunk_id"]


def test_search_in_one_document_and_bad_arguments(notes, evals):
    server = build_server(notes, evals)
    s = call(server, "search_notes", {"query": "metrics", "document": "email"}).structured_content
    assert {p["document"] for p in s["passages"]} == {"email"}
    bad = call(server, "search_notes", {"query": "x", "document": "no such lecture"})
    assert bad.is_error and "no document matches" in bad.content[0].text and "Available" in bad.content[0].text
    assert call(server, "search_notes", {"query": "", "top_k": 2}).is_error
    assert call(server, "search_notes", {"query": "seo", "top_k": 99}).is_error


def test_browse_documents(notes, evals):
    server = build_server(notes, evals)
    docs = call(server, "list_documents").structured_content
    assert docs["count"] == 2 and {d["title"] for d in docs["documents"]} == {"seo", "email"}
    outline = call(server, "document_outline", {"document": "seo"}).structured_content["outline"]
    assert any("Backlinks" in line for line in outline)
    missing = call(server, "get_passage", {"document": "seo", "location": 99})
    assert missing.is_error and "no passage at location 99" in missing.content[0].text


def test_check_answer_flags_unsupported_sentences(notes, evals):
    server = build_server(notes, evals)
    passages = call(server, "search_notes", {"query": "backlinks votes of trust", "top_k": 2}).structured_content["passages"]
    ids = [p["chunk_id"] for p in passages]
    answer = ("Search engines treat quality backlinks from other websites as votes of trust [1]. "
              "Backlinks were invented by Napoleon in the year 1850 [1].")
    r = call(server, "check_answer", {"answer": answer, "chunk_ids": ids}).structured_content
    assert r["sentences"][0]["supported"] and not r["sentences"][1]["supported"]
    assert r["unsupported"] == ["Backlinks were invented by Napoleon in the year 1850."]
    bad = call(server, "check_answer", {"answer": "x y z", "chunk_ids": ["nope"]})
    assert bad.is_error and "unknown chunk_id" in bad.content[0].text


def test_eval_tools_and_path_safety(notes, evals):
    server = build_server(notes, evals)
    assert call(server, "list_eval_runs").structured_content["runs"] == ["demo/v1", "demo/v2"]
    summary = call(server, "eval_run_summary", {"run": "demo/v1"}).structured_content
    assert summary["items"] == 30 and "correct" in summary["metrics"]
    cmp = call(server, "compare_eval_runs", {"base": "demo/v1", "new": "demo/v2"}).structured_content
    d = cmp["metrics"][0]
    assert d["metric"] == "correct" and d["test"] == "McNemar" and d["significant"] and len(d["improved"]) == 10
    for sneaky in ("../secret", "../../etc/passwd", "/etc/passwd"):
        r = call(server, "eval_run_summary", {"run": sneaky})
        assert r.is_error and "password" not in r.content[0].text


def test_resources_and_prompts(notes, evals):
    async def run():
        async with Client(build_server(notes, evals)) as c:
            listing = await c.read_resource("notes://documents")
            text = await c.read_resource("notes://documents/seo")
            prompts = await c.list_prompts()
            quiz = await c.get_prompt("quiz_me", {"topic": "SEO", "questions": "3"})
            return listing, text, prompts, quiz
    listing, text, prompts, quiz = asyncio.run(run())
    assert json.loads(listing.contents[0].text)["count"] == 2
    assert "votes of trust" in text.contents[0].text and text.contents[0].text.startswith("# seo")
    assert {p.name for p in prompts.prompts} == {"explain_from_notes", "quiz_me", "revise_document"}
    assert "3 questions ONE AT A TIME" in quiz.messages[0].content.text


def test_server_picks_up_files_synced_after_it_started(notes, evals, tmp_path):
    server = build_server(notes, evals)
    assert call(server, "list_documents").structured_content["count"] == 2
    new = tmp_path / "docs" / "ppc.md"
    new.write_text("# Pay per click\nPay-per-click advertising charges the advertiser each time an ad is clicked.\n")
    # another process (the StudyRAG app or CLI) syncs the folder: same storage dir, separate Library object
    from studyrag.library import Library
    import os, time
    other = Library(notes.b.settings, notes.b.embedder)
    other.sync_folder(tmp_path / "docs")
    manifest = tmp_path / "storage" / "manifest.json"
    os.utime(manifest, (time.time() + 5, time.time() + 5))      # make the change visible on coarse-mtime filesystems
    assert call(server, "list_documents").structured_content["count"] == 3
    assert "Pay per click" in call(server, "search_notes", {"query": "pay per click ads"}).structured_content["passages"][0]["citation"]
