"""Repo tools and change sets, on a throwaway git repo."""

import asyncio
import subprocess

import pytest

from scout.approval import DenyAll, Scripted
from scout.changes import ChangeSet, edit_tools, slugify
from scout.repo import Repo
from scout.tools import ToolError


def sh(cwd, *cmd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "README.md").write_text("# Proj\nSee [setup](docs/SETUP.md) and [gone](docs/MISSING.md).\n"
                                    "```\n[not a link](nowhere.md)\n```\n15 tests pass.\n")
    (root / "docs" / "SETUP.md").write_text("# Setup\npip install proj\n")
    (root / ".gitignore").write_text("secret/\n.env\n")
    (root / "secret").mkdir()
    (root / "secret" / "notes.md").write_text("private lecture notes")
    (root / ".env").write_text("KEY=sk-123")
    sh(root, "git", "init", "-q", "-b", "main")
    sh(root, "git", "-c", "user.name=t", "-c", "user.email=t@e", "add", ".")
    sh(root, "git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "init")
    sh(root, "git", "config", "user.name", "t")
    sh(root, "git", "config", "user.email", "t@e")
    return Repo(root)


def test_only_tracked_files_are_visible(repo):
    assert repo.list_files("*")["files"] == [".gitignore", "README.md", "docs/SETUP.md"]
    for path in ("secret/notes.md", ".env", "../outside.md", "/etc/passwd", "docs/../../x"):
        with pytest.raises(ToolError):
            repo.read_file(path)
    assert not any("private" in m["text"] for m in repo.search("private|sk-")["matches"])


def test_read_search_log_and_links(repo):
    r = repo.read_file("./README.md", 1, 2)
    assert r["lines"] == "1-2 of 6" and "   2  See [setup]" in r["text"]
    assert repo.search("tests pass")["matches"] == [{"path": "README.md", "line": 6, "text": "15 tests pass."}]
    assert repo.git_log(5)["commits"][0].endswith("init")
    links = repo.check_links()
    assert links["count"] == 1 and links["broken"][0]["link"] == "docs/MISSING.md"


def test_propose_edit_drafts_without_writing_then_applies_on_approval(repo):
    cs = ChangeSet(str(repo.root), message="Fix README test count", kind="checkup")
    propose = edit_tools(cs, repo)[0].fn
    propose(path="README.md", old="15 tests", new="16 tests", reason="test count is stale")
    propose(path="README.md", old="[gone](docs/MISSING.md)", new="[setup](docs/SETUP.md)", reason="broken link")
    assert "15 tests" in (repo.root / "README.md").read_text()              # nothing written yet
    assert "-15 tests pass." in cs.diff() and "+16 tests pass." in cs.diff()
    with pytest.raises(ToolError, match="exactly once"):
        propose(path="README.md", old="nonexistent", new="x", reason="nope")
    out = asyncio.run(cs.approve_and_apply(Scripted([True]), push=False))
    assert out["status"] == "applied" and out["written"] == ["README.md"] and "16 tests" in (repo.root / "README.md").read_text()
    assert sh(repo.root, "git", "log", "-1", "--format=%s").strip() == "Fix README test count"


def test_declined_set_changes_nothing(repo):
    cs = ChangeSet(str(repo.root), message="m")
    cs.put("learning-log/2026-11-25-mcp.md", "# MCP\n")
    approver = DenyAll()
    assert asyncio.run(cs.approve_and_apply(approver))["status"] == "declined"
    assert not (repo.root / "learning-log").exists() and "learning-log/2026-11-25-mcp.md" in approver.seen[0].details


def test_stale_proposal_is_refused_and_only_its_paths_are_committed(repo, tmp_path):
    cs = ChangeSet(str(repo.root), message="note")
    cs.put("notes/a.md", "# A\n")
    (repo.root / "docs" / "SETUP.md").write_text("# Setup\nlocal work in progress\n")    # unrelated uncommitted work
    out = cs.apply(push=False)
    assert out["commit"] and sh(repo.root, "git", "show", "--stat", "--format=", "HEAD").count("|") == 1
    assert "SETUP.md" in sh(repo.root, "git", "status", "--short")                  # still uncommitted, untouched
    stale = ChangeSet(str(repo.root), message="x")
    stale.put("README.md", "changed\n")
    (repo.root / "README.md").write_text("someone edited this meanwhile\n")
    with pytest.raises(ToolError, match="changed since"):
        stale.apply(push=False)


def test_save_and_load_round_trip(repo, tmp_path):
    cs = ChangeSet(str(repo.root), message="m", kind="digest", summary="s")
    cs.put("a.md", "x")
    loaded = ChangeSet.load(cs.save(tmp_path / "pending"))
    assert loaded == cs and slugify("What's new in MCP? (2026)") == "what-s-new-in-mcp-2026"
