"""Read-only tools over a git repository, plus deterministic checks (broken links).

Only files **tracked by git** are visible. Anything gitignored (private documents, `.env` keys, model weights)
can't be listed, read or searched, so it is never sent to the model provider. Paths are resolved inside the
repo; `../` tricks and absolute paths are refused.
"""

from __future__ import annotations

import fnmatch
import re
import subprocess
from functools import cached_property
from pathlib import Path

from .tools import Tool, ToolError

TEXT_SUFFIXES = {".md", ".py", ".txt", ".toml", ".yml", ".yaml", ".json", ".cfg", ".ini", ".sh", ".html", ".css", ".js",
                 ".csv", ".rst", ".gitignore"}
_MD_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")


def git(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise ToolError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
    return r.stdout


class Repo:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not (self.root / ".git").exists():
            raise ValueError(f"{self.root} is not a git repository")

    @cached_property
    def files(self) -> list[str]:
        return sorted(f for f in git(self.root, "ls-files", "-z").split("\0") if f)

    def refresh(self) -> None:
        self.__dict__.pop("files", None)

    def resolve(self, path: str) -> Path:
        rel = path.strip().removeprefix("./")
        if rel.startswith("/"):
            raise ToolError("use a path relative to the repository root")
        p = (self.root / rel).resolve()
        if self.root != p and self.root not in p.parents:
            raise ToolError("path must be inside the repository")
        rel = p.relative_to(self.root).as_posix()
        if rel not in self.files:
            close = [f for f in self.files if f.endswith("/" + Path(rel).name) or f == Path(rel).name][:5]
            raise ToolError(f"{rel!r} is not a tracked file" + (f"; did you mean {close}?" if close else ""))
        return p

    # ------------------------------------------------------------------ tools
    def list_files(self, pattern: str = "*", limit: int = 200) -> dict:
        hits = [f for f in self.files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)]
        return {"pattern": pattern, "count": len(hits), "files": hits[:limit],
                **({"note": f"showing {limit} of {len(hits)}; use a narrower pattern"} if len(hits) > limit else {})}

    def read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> dict:
        p = self.resolve(path)
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name not in {"LICENSE", "Makefile", ".gitignore"}:
            raise ToolError(f"{path} isn't a text file this tool reads")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        end_line = min(end_line, start_line + 299, len(lines))
        body = "\n".join(f"{i:>4}  {lines[i - 1]}" for i in range(start_line, end_line + 1))
        return {"path": p.relative_to(self.root).as_posix(), "lines": f"{start_line}-{end_line} of {len(lines)}",
                "text": body}

    def search(self, pattern: str, glob: str = "*", limit: int = 40) -> dict:
        try:
            rx = re.compile(pattern, re.I)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}") from None
        hits = []
        for f in self.files:
            if not (fnmatch.fnmatch(f, glob) or fnmatch.fnmatch(Path(f).name, glob)):
                continue
            p = self.root / f
            if p.suffix.lower() not in TEXT_SUFFIXES or not p.exists() or p.stat().st_size > 1_000_000:
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append({"path": f, "line": i, "text": line.strip()[:200]})
                    if len(hits) >= limit:
                        return {"pattern": pattern, "matches": hits, "note": f"stopped at {limit} matches"}
        return {"pattern": pattern, "matches": hits}

    def git_log(self, n: int = 10, path: str = "") -> dict:
        args = ["log", f"-{n}", "--date=short", "--format=%h %ad %s"]
        if path:
            args += ["--", path]
        return {"commits": git(self.root, *args).splitlines()}

    def check_links(self, glob: str = "*.md") -> dict:
        """Relative links in markdown files that point to files that don't exist (deterministic, no model)."""
        broken = []
        for f in self.files:
            if not (fnmatch.fnmatch(f, glob) or fnmatch.fnmatch(Path(f).name, glob)) or not f.endswith(".md"):
                continue
            text = (self.root / f).read_text(encoding="utf-8", errors="replace")
            text = re.sub(r"```.*?```", "", text, flags=re.S)           # links inside code blocks aren't links
            for i, line in enumerate(text.splitlines(), 1):
                for m in _MD_LINK.finditer(line):
                    target = (m.group(1) or m.group(2)).split("#")[0]
                    if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):    # http:, mailto:, anchors
                        continue
                    dest = ((self.root / f).parent / target).resolve()
                    if not dest.exists():
                        broken.append({"file": f, "line": i, "link": m.group(1) or m.group(2)})
        return {"checked": glob, "broken": broken, "count": len(broken)}

    def tools(self) -> list[Tool]:
        S = {"type": "string"}
        return [
            Tool("list_files", "List tracked files in the repository matching a glob (e.g. '*.md', '01-*/docs/*').",
                 {"type": "object", "properties": {"pattern": {**S, "default": "*"}}}, self.list_files),
            Tool("read_file", "Read lines of a tracked text file (numbered).",
                 {"type": "object", "properties": {"path": S, "start_line": {"type": "integer", "minimum": 1, "default": 1},
                                                   "end_line": {"type": "integer", "minimum": 1, "default": 200}},
                  "required": ["path"]}, self.read_file),
            Tool("search_repo", "Search tracked text files with a regular expression (case-insensitive).",
                 {"type": "object", "properties": {"pattern": {**S, "minLength": 1}, "glob": {**S, "default": "*"}},
                  "required": ["pattern"]}, self.search),
            Tool("git_log", "Recent commits, optionally for one path.",
                 {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                                                   "path": {**S, "default": ""}}}, self.git_log),
            Tool("check_links", "Find relative links in markdown files that point to missing files.",
                 {"type": "object", "properties": {"glob": {**S, "default": "*.md"}}}, self.check_links),
        ]
