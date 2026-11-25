"""Proposed changes: the agent drafts, you approve, then (and only then) files are written, committed and pushed.

During a task the agent can only *add to a ChangeSet* (`propose_edit`, `write_note`); nothing on disk changes.
At the end the whole set is shown as one diff with its commit message, and one approval covers
"write these files, commit them, push". Declined sets are discarded; a prepared set can be saved and reviewed
later (`scout review`), so an unattended daily job never pushes anything by itself.

Safety checks at apply time: every edited file must still have exactly the content the proposal was based on
(otherwise the proposal is stale and refused), and only the proposal's own paths are committed, so unrelated
work in your tree is never swept into the commit.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .approval import Action, Approver
from .repo import Repo, git
from .tools import Tool, ToolError


def _sha(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()[:16]


def slugify(title: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s[:max_len].rstrip("-") or "note"


@dataclass
class Change:
    path: str                  # relative to the repo root
    new_text: str
    base_sha: str | None       # sha of the file when proposed; None = a new file
    reason: str = ""


@dataclass
class ChangeSet:
    repo_root: str
    message: str = ""          # commit message
    kind: str = ""             # "digest" | "checkup" | ...
    summary: str = ""          # the agent's report, shown with the diff
    changes: list[Change] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------------ drafting
    def current(self, path: str) -> str | None:
        """The file's text including earlier proposals in this set (None if it doesn't exist)."""
        for c in reversed(self.changes):
            if c.path == path:
                return c.new_text
        p = Path(self.repo_root) / path
        return p.read_text(encoding="utf-8") if p.exists() else None

    def put(self, path: str, new_text: str, reason: str = "") -> None:
        for c in self.changes:
            if c.path == path:
                c.new_text, c.reason = new_text, "; ".join(filter(None, [c.reason, reason]))
                return
        p = Path(self.repo_root) / path
        base = p.read_text(encoding="utf-8") if p.exists() else None
        self.changes.append(Change(path, new_text, _sha(base), reason))

    # ------------------------------------------------------------------ showing
    def diff(self) -> str:
        out = []
        for c in self.changes:
            p = Path(self.repo_root) / c.path
            old = p.read_text(encoding="utf-8") if p.exists() else ""
            out.extend(difflib.unified_diff(old.splitlines(), c.new_text.splitlines(),
                                            "/dev/null" if c.base_sha is None else f"a/{c.path}", f"b/{c.path}",
                                            lineterm="", n=2))
        return "\n".join(out)

    def action(self, push: bool) -> Action:
        n = len(self.changes)
        where = Path(self.repo_root).name
        summary = f"write {n} file{'s' * (n != 1)} in {where}, commit \"{self.message}\"" + (", and push" if push else "")
        details = (self.summary.strip() + "\n\n" if self.summary.strip() else "") + self.diff()
        return Action("commit", summary, details)

    # ------------------------------------------------------------------ saving for later review
    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.created.replace(':', '')}-{self.kind or 'change'}.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "ChangeSet":
        d = json.loads(path.read_text(encoding="utf-8"))
        d["changes"] = [Change(**c) for c in d["changes"]]
        return cls(**d)

    # ------------------------------------------------------------------ applying
    def apply(self, *, commit: bool = True, push: bool = False, remote: str = "origin") -> dict:
        root = Path(self.repo_root)
        for c in self.changes:                       # all checks first: nothing is written if one is stale
            p = (root / c.path).resolve()
            if root.resolve() not in p.parents:
                raise ToolError(f"{c.path} is outside the repository")
            now = p.read_text(encoding="utf-8") if p.exists() else None
            if _sha(now) != c.base_sha:
                raise ToolError(f"{c.path} changed since this was proposed; run the task again")
        for c in self.changes:
            p = root / c.path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c.new_text if c.new_text.endswith("\n") else c.new_text + "\n", encoding="utf-8")
        result = {"written": [c.path for c in self.changes]}
        if commit:
            paths = [c.path for c in self.changes]
            git(root, "add", "--", *paths)
            git(root, "commit", "-q", "-m", self.message, "--", *paths)
            result["commit"] = git(root, "rev-parse", "--short", "HEAD").strip()
            if push:
                branch = git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
                try:
                    git(root, "push", "-q", remote, branch)
                    result["pushed"] = f"{remote}/{branch}"
                except ToolError as e:
                    result["push_error"] = f"committed locally but the push failed ({e}); run `git push` later"
        return result

    async def approve_and_apply(self, approver: Approver, *, push: bool = True) -> dict:
        if not self.changes:
            return {"status": "nothing to change"}
        if not await approver.approve(self.action(push)):
            return {"status": "declined"}
        return {"status": "applied", **self.apply(push=push)}


# ------------------------------------------------------------------ tools that draft into a ChangeSet
def edit_tools(cs: ChangeSet, repo: Repo, *, allowed: str = "*") -> list[Tool]:
    import fnmatch

    def propose_edit(path: str, old: str, new: str, reason: str) -> dict:
        p = repo.resolve(path)
        rel = p.relative_to(repo.root).as_posix()
        if not fnmatch.fnmatch(rel, allowed):
            raise ToolError(f"edits are limited to {allowed!r} in this task")
        text = cs.current(rel)
        count = text.count(old) if old else 0
        if count != 1:
            raise ToolError(f"`old` must appear exactly once in {rel} (found {count} times); copy it exactly, "
                            "with enough surrounding text to be unique")
        if old == new:
            raise ToolError("old and new are the same")
        cs.put(rel, text.replace(old, new, 1), reason)
        return {"proposed": rel, "note": "not written yet: the user reviews all proposals at the end"}

    return [Tool("propose_edit", "Propose replacing one exact snippet in a tracked file. Nothing is written until the "
                                 "user approves all proposals at the end.",
                 {"type": "object", "properties": {
                     "path": {"type": "string"}, "old": {"type": "string", "minLength": 1,
                                                         "description": "exact current text, unique in the file"},
                     "new": {"type": "string"}, "reason": {"type": "string", "minLength": 3}},
                  "required": ["path", "old", "new", "reason"]}, propose_edit)]
