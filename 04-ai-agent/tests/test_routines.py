"""Routines end to end with a scripted model, a fake internet and a throwaway git repo (with a bare 'origin')."""

import asyncio
import json
import subprocess
from datetime import date

import httpx
import pytest

from conftest import calls, text
from scout.approval import DenyAll, Scripted
from scout.config import Settings
from scout.memory import Memory
from scout.routines import checkup, digest, review
from scout.web import Web

PAGE = "<html><title>MCP 2026-07-28</title><body>" + "<p>The 2026-07-28 MCP revision makes the protocol stateless.</p>" * 10 + "</body></html>"
SEARCH = {"results": [{"title": "MCP changelog", "url": "https://mcp.example/changelog", "content": "stateless"},
                      {"title": "Blog", "url": "https://blog.example/mcp", "content": "discover"}]}
BODY = ("## What happened\nThe MCP spec went stateless 【1†L3-L5】.\n\n## Why it matters for my projects\nMy server must support "
        "server/discover [2].\n\n## Key details\n- No initialize handshake [1]\n- server/discover is required [2]\n\n"
        "## Try it\nProbe my server with server/discover.\n\n## Sources\n1. made-up list that must be dropped")


def sh(cwd, *cmd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def setup(tmp_path):
    origin = tmp_path / "origin.git"
    sh(tmp_path, "git", "init", "-q", "--bare", "-b", "main", str(origin))
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("# Repo\nSee [docs](docs/GONE.md). 15 tests pass.\n")
    sh(root, "git", "init", "-q", "-b", "main")
    sh(root, "git", "config", "user.name", "t")
    sh(root, "git", "config", "user.email", "t@e")
    sh(root, "git", "add", ".")
    sh(root, "git", "commit", "-q", "-m", "init")
    sh(root, "git", "remote", "add", "origin", str(origin))
    sh(root, "git", "push", "-q", "origin", "main")
    settings = Settings(api_key="k", tavily_key="t", workspace=root, journal_dir=root / "learning-log",
                        state_dir=tmp_path / "state", mcp_command=None, topics=["MCP"])
    return settings, root, origin


def fake_web():
    def handler(req):
        url = str(req.url)
        if url.endswith("/search"):
            return httpx.Response(200, json=SEARCH)
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
    return Web("t", transport=httpx.MockTransport(handler), url_check=lambda u: u)


def digest_script(cite_unread_first=False):
    steps = [calls(("web_search", {"query": "MCP stateless revision"})),
             calls(("web_read", {"source": "1", "question": "what changed"}))]
    if cite_unread_first:               # cites [2] before reading it: refused, then reads it and retries
        steps.append(calls(("write_note", {"title": "MCP goes stateless in 2026-07-28", "body": BODY})))
    steps += [calls(("web_read", {"source": "2", "question": "discover"})),
              calls(("write_note", {"title": "MCP goes stateless in 2026-07-28", "body": BODY})),
              text("MCP goes stateless in 2026-07-28")]
    return steps


def test_digest_drafts_a_cited_note_and_commits_after_approval(fake_api, setup):
    settings, root, origin = setup
    api = fake_api(digest_script(cite_unread_first=True))
    r = asyncio.run(digest(settings, Scripted([True]), llm=api.llm(), web=fake_web(), today=date(2026, 11, 25)))
    assert r.status == "applied" and r.title == "MCP goes stateless in 2026-07-28"
    refusal = [e["result"] for e in r.run.events if e["type"] == "tool" and e["name"] == "write_note"][0]
    assert "without reading them" in refusal
    assert "Today is 25 November 2026" in api.requests[0]["messages"][1]["content"] and "after August 2026" in api.requests[0]["messages"][1]["content"]
    note = (root / "learning-log" / "2026-11-25-mcp-goes-stateless-in-2026-07-28.md").read_text()
    assert note.startswith("# MCP goes stateless in 2026-07-28\n\n*2026-11-25 · drafted by Scout")
    assert "made-up list" not in note and "1. [MCP changelog](https://mcp.example/changelog)" in note
    index = (root / "learning-log" / "README.md").read_text()
    assert "- 2026-11-25 · [MCP goes stateless in 2026-07-28](2026-11-25-mcp-goes-stateless-in-2026-07-28.md)" in index
    assert sh(origin, "git", "log", "-1", "--format=%s").strip() == "Learning log: MCP goes stateless in 2026-07-28"
    assert "Co-Authored-By" not in sh(origin, "git", "log", "-1", "--format=%B")
    mem = Memory(settings.memory_path).recent("digest")
    assert mem[-1]["status"] == "applied" and len(mem[-1]["sources"]) == 2


def test_digest_declined_changes_nothing_and_next_digest_avoids_the_topic(fake_api, setup):
    settings, root, origin = setup
    api = fake_api(digest_script())
    r = asyncio.run(digest(settings, DenyAll(), llm=api.llm(), web=fake_web()))
    assert r.status == "declined" and not (root / "learning-log").exists()
    api2 = fake_api([text("nothing new today")])
    r2 = asyncio.run(digest(settings, DenyAll(), llm=api2.llm(), web=fake_web()))
    assert r2.status == "failed" and "MCP goes stateless" in api2.requests[0]["messages"][1]["content"]


def test_prepared_digest_waits_for_review(fake_api, setup):
    settings, root, origin = setup
    r = asyncio.run(digest(settings, DenyAll(), prepare=True, llm=fake_api(digest_script()).llm(), web=fake_web()))
    assert r.status == "prepared" and not (root / "learning-log").exists()
    reviewed = asyncio.run(review(settings, Scripted([True])))
    assert [x.status for x in reviewed] == ["applied"] and (root / "learning-log" / "README.md").exists()
    assert asyncio.run(review(settings, Scripted([True]))) == []          # nothing left pending


def test_checkup_proposes_verified_doc_fixes(fake_api, setup):
    settings, root, origin = setup
    api = fake_api([calls(("read_file", {"path": "README.md"})),
                    calls(("propose_edit", {"path": "README.md", "old": "See [docs](docs/GONE.md). ", "new": "",
                                            "reason": "Remove a link to a file that doesn't exist"})),
                    calls(("propose_edit", {"path": "src/app.py", "old": "x", "new": "y", "reason": "not docs"})),
                    text("Removed one broken link.")])
    r = asyncio.run(checkup(settings, Scripted([True]), llm=api.llm()))
    assert "docs/GONE.md" in api.requests[0]["messages"][1]["content"]       # the deterministic check is given to it
    assert r.status == "applied" and (root / "README.md").read_text() == "# Repo\n15 tests pass.\n"
    assert r.title == "Docs checkup: remove a link to a file that doesn't exist"
    assert r.detail["proposed"] == ["README.md: Remove a link to a file that doesn't exist"]
    assert r.changes.summary.startswith("Proposed edits (from the actual proposals):")
    assert any("not a tracked file" in e["result"] or "limited to" in e["result"] for e in r.run.events if e["type"] == "tool")


def test_checkup_with_nothing_to_fix(fake_api, setup):
    settings, root, origin = setup
    r = asyncio.run(checkup(settings, DenyAll(), llm=fake_api([text("All good.")]).llm()))
    assert r.status == "nothing" and r.detail["report"] == "All good."


def test_note_needs_two_read_sources(fake_api, setup):
    settings, root, origin = setup
    one = BODY.replace("[2]", "[1]")
    api = fake_api([calls(("web_search", {"query": "MCP"})), calls(("web_read", {"source": "1"})),
                    calls(("write_note", {"title": "MCP goes stateless in 2026-07-28", "body": one})), text("gave up")])
    r = asyncio.run(digest(settings, DenyAll(), llm=api.llm(), web=fake_web()))
    assert r.status == "failed" and "at least two different sources" in r.run.events[-3]["result"]


def test_digest_step_limit_is_its_own_setting(fake_api, setup):
    settings, root, origin = setup
    assert Settings().digest_max_steps == 16          # 12 ran out before the note was saved (E4, gpt-oss-20b run 4)
    settings.digest_max_steps = 13
    searching = [calls(("web_search", {"query": f"MCP news {i}"})) for i in range(13)]
    api = fake_api(searching + [text("ran out of steps")])
    r = asyncio.run(digest(settings, DenyAll(), llm=api.llm(), web=fake_web()))
    assert r.status == "failed" and r.run.stopped == "max_steps" and r.run.steps == 14   # 13 + the final answer
