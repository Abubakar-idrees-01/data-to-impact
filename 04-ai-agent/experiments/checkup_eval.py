"""E3. Does `scout checkup` find real doc problems without "fixing" things that are fine?

A small project ("tidy") with 5 planted problems and 4 decoys (things that look wrong but aren't). The checkup routine
runs on it exactly as on the real repo, with the approver declining, and its proposals are graded:

    planted (should be fixed)                              decoys (must stay unchanged)
    P1 README links docs/USAGE.md; the file is             D1 a #anchor link (not a file)
       docs/usage-guide.md                                 D2 British spelling: colour, normalise
    P2 README says 12 tests; there are 14                  D3 a "link" inside a code block
    P3 SETUP says Python 3.8+; pyproject says >=3.11       D4 "95%" that matches results.json
    P4 typo "recieve"
    P5 SETUP pins tidy==0.3.0; the version is 0.4.0

    python experiments/checkup_eval.py --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent / "02-llm-evals" / "src"))

from scout.approval import DenyAll                   # noqa: E402
from scout.config import Settings                    # noqa: E402
from scout.routines import checkup                   # noqa: E402

FILES = {
    "pyproject.toml": '[project]\nname = "tidy"\nversion = "0.4.0"\nrequires-python = ">=3.11"\n',
    "README.md": ("# tidy\n\nA tiny text-cleaning library.\n\n- Setup: [docs/SETUP.md](docs/SETUP.md)\n"
                  "- Usage: [docs/USAGE.md](docs/USAGE.md)\n- Jump to [installation](#installation)\n\n"
                  "## Installation\n`pip install tidy`\n\n## Tests\n12 tests pass (`pytest`).\n\n"
                  "We use British spelling throughout: colour, normalise.\n"),
    "docs/SETUP.md": ("# Setup\n\nRequires Python 3.8 or newer.\n\n```bash\npip install tidy==0.3.0\n```\n\n"
                      "Then run `tidy --help`.\n"),
    "docs/usage-guide.md": ("# Usage\n\nCall `clean(text)` to recieve a normalised string.\n\n"
                            "```markdown\nSee [example](missing.md) for a sample document.\n```\n"),
    "docs/RESULTS.md": "# Results\n\nAccuracy on the sample set: 95% (see `results.json`).\n",
    "results.json": '{"accuracy": 0.95, "n": 200}\n',
    "tests/test_clean.py": "".join(f"def test_clean_{i}():\n    assert True\n\n\n" for i in range(9)),
    "tests/test_cli.py": "".join(f"def test_cli_{i}():\n    assert True\n\n\n" for i in range(5)),
}

PLANTED = {
    "P1-broken-link": lambda f: "docs/usage-guide.md" in f["README.md"] and "docs/USAGE.md)" not in f["README.md"],
    "P2-test-count": lambda f: "14 tests" in f["README.md"],
    "P3-python-version": lambda f: "3.11" in f["docs/SETUP.md"] and "3.8" not in f["docs/SETUP.md"],
    "P4-typo": lambda f: "receive" in f["docs/usage-guide.md"] and "recieve" not in f["docs/usage-guide.md"],
    "P5-pinned-version": lambda f: "0.3.0" not in f["docs/SETUP.md"],
}
DECOYS = {
    "D1-anchor": lambda f: "[installation](#installation)" in f["README.md"],
    "D2-british": lambda f: "colour, normalise" in f["README.md"] and "normalised string" in f["docs/usage-guide.md"],
    "D3-code-block": lambda f: "[example](missing.md)" in f["docs/usage-guide.md"],
    "D4-matching-number": lambda f: "95%" in f["docs/RESULTS.md"],
}


def make_repo(tmp: Path) -> Path:
    root = tmp / "tidy"
    for name, text in FILES.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    for cmd in (["git", "init", "-q"], ["git", "add", "."],
                ["git", "-c", "user.name=t", "-c", "user.email=t@e", "commit", "-q", "-m", "init"]):
        subprocess.run(cmd, cwd=root, check=True, capture_output=True)
    return root


async def one_run(settings: Settings, i: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_repo(Path(tmp))
        s = Settings(**{**settings.__dict__, "workspace": root, "state_dir": settings.state_dir})
        r = await checkup(s, DenyAll(), max_fixes=5)
        cs = r.changes
        final = {name: (cs.current(name) if cs else text) or "" for name, text in FILES.items()}
        changed = sorted(c.path for c in cs.changes) if cs else []
    planted = {k: bool(v(final)) for k, v in PLANTED.items()}
    decoys_kept = {k: bool(v(final)) for k, v in DECOYS.items()}
    return {"id": f"run{i}", "fixed": sum(planted.values()) / len(planted),
            "decoys_broken": sum(not v for v in decoys_kept.values()),
            **{k: v for k, v in planted.items()}, **{k: v for k, v in decoys_kept.items()},
            "files_changed": changed, "proposals": len(changed), "tokens": r.run.total_tokens,
            "steps": r.run.steps, "seconds": r.run.seconds, "stopped": r.run.stopped,
            "report": (r.detail or {}).get("report", "")[:1500], "trace": r.run.trace_path}


async def main_async(a) -> None:
    s = Settings.from_env()
    if a.model:
        s.api_model = a.model
    s.fallback_model = None
    out = ROOT / "experiments" / "out" / f"checkup-{s.api_model.split('/')[-1]}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = len(out.read_text().splitlines()) if out.exists() else 0
    for i in range(done + 1, a.runs + 1):
        rec = await one_run(s, i)
        if rec["stopped"] == "error":
            print(f"  run{i}: model error, stopping: {rec['report'][:200]}")
            return
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  run{i}: fixed {rec['fixed']:.0%}  decoys broken {rec['decoys_broken']}  "
              f"{rec['tokens']:,} tok {rec['seconds']:.0f}s  " +
              " ".join(k for k in PLANTED if rec[k]), flush=True)
    from evalkit.runs import Run
    items = [json.loads(l) for l in out.read_text().splitlines()]
    run = Run.build("scout-checkup", out.stem.removeprefix("checkup-"), items,
                    metrics=["fixed", "decoys_broken", *PLANTED, "tokens", "seconds", "steps"],
                    config={"model": s.api_model, "planted": list(PLANTED), "decoys": list(DECOYS)})
    print("saved", run.save(ROOT / "results"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
