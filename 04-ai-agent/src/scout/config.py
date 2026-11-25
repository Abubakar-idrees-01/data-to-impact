"""Settings, read from environment variables or the project's `.env` file (gitignored).

    SCOUT_API_BASE_URL=https://api.groq.com/openai/v1
    SCOUT_API_MODEL=openai/gpt-oss-120b
    SCOUT_API_KEY=...
    TAVILY_API_KEY=...
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # 04-ai-agent/
REPO_ROOT = PROJECT_ROOT.parent                          # the data-to-impact monorepo

DEFAULT_TOPICS = [
    "retrieval-augmented generation (RAG)",
    "evaluating LLM applications",
    "Model Context Protocol (MCP)",
    "running small language models on a CPU",
]


def load_dotenv(path: str | Path, override: bool = False) -> None:
    """Read KEY=VALUE lines into os.environ (no python-dotenv dependency)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip().strip("'\"")
        if override or key not in os.environ:
            os.environ[key] = value


def default_mcp_command() -> list[str] | None:
    """The StudyRAG MCP server (project 3): its venv in a source checkout, else an installed `studyrag-mcp`."""
    venv_python = REPO_ROOT / "03-mcp-server" / ".venv" / "bin" / "python"
    if venv_python.exists():
        return [str(venv_python), "-m", "studyrag_mcp"]
    exe = shutil.which("studyrag-mcp")
    return [exe] if exe else None


@dataclass
class Settings:
    api_base_url: str = ""
    api_model: str = ""
    api_key: str = ""
    tavily_key: str = ""
    reasoning_effort: str | None = "low"      # gpt-oss on Groq: low/medium/high; None to omit
    max_steps: int = 8                        # model calls per task before it must answer
    token_budget: int = 40_000                # prompt + completion tokens per task
    max_tokens_per_call: int = 2048
    page_chars: int = 4000                    # how much of a web page the model gets to read
    workspace: Path = REPO_ROOT               # the repo the repo tools may read (and propose changes to)
    journal_dir: Path = REPO_ROOT / "learning-log"
    state_dir: Path = PROJECT_ROOT / ".scout"     # traces and task memory (gitignored)
    mcp_command: list[str] | None = field(default_factory=default_mcp_command)
    topics: list[str] = field(default_factory=lambda: list(DEFAULT_TOPICS))

    @property
    def traces_dir(self) -> Path:
        return self.state_dir / "traces"

    @property
    def memory_path(self) -> Path:
        return self.state_dir / "memory.jsonl"

    @classmethod
    def from_env(cls, dotenv: str | Path | None = PROJECT_ROOT / ".env", **overrides) -> "Settings":
        if dotenv:
            load_dotenv(dotenv)
        env = os.environ.get
        s = cls(
            api_base_url=env("SCOUT_API_BASE_URL", "https://api.groq.com/openai/v1"),
            api_model=env("SCOUT_API_MODEL", "openai/gpt-oss-120b"),
            api_key=env("SCOUT_API_KEY", ""),
            tavily_key=env("TAVILY_API_KEY", ""),
        )
        if env("SCOUT_REASONING_EFFORT") is not None:
            s.reasoning_effort = env("SCOUT_REASONING_EFFORT") or None
        if env("SCOUT_WORKSPACE"):
            s.workspace = Path(env("SCOUT_WORKSPACE")).expanduser()
        if env("SCOUT_JOURNAL_DIR"):
            s.journal_dir = Path(env("SCOUT_JOURNAL_DIR")).expanduser()
        if env("SCOUT_STATE_DIR"):
            s.state_dir = Path(env("SCOUT_STATE_DIR")).expanduser()
        if env("SCOUT_TOPICS"):
            s.topics = [t.strip() for t in env("SCOUT_TOPICS").split(";") if t.strip()]
        if env("SCOUT_MCP_COMMAND") is not None:
            s.mcp_command = env("SCOUT_MCP_COMMAND").split() or None
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
