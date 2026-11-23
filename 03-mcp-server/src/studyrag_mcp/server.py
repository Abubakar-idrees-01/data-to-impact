"""The StudyRAG MCP server (official SDK): your study notes and evaluation results as tools for any AI assistant.

    studyrag-mcp                     # stdio (what Claude Code, Claude Desktop, Cursor... launch)
    studyrag-mcp --http --port 8765  # Streamable HTTP, for clients that connect over the network

Tools never write anything and never call an LLM: the assistant is the language model. Passages a tool returns
are sent to that assistant (and its provider), exactly like text pasted into a chat.
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from typing import Annotated, Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from . import __version__
from .evals import Evals, EvalsError
from .notes import MAX_TOP_K, Notes, NotesError

log = logging.getLogger("studyrag_mcp")

INSTRUCTIONS = """Tools over the user's own study material (lecture slides, PDFs, notes indexed by StudyRAG) and
their evaluation results.
- To answer a question about the user's material: call search_notes, answer ONLY from the returned passages,
  cite them as [n] with their citation (e.g. "Lec 2 — Slide 14"), and if `found` is false say the notes don't
  cover it instead of using general knowledge.
- To double-check an answer you wrote: call check_answer with the chunk_ids you cited, in [n] order.
- list_documents / document_outline / get_passage browse the library; eval tools read evaluation runs."""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def as_tool_errors(fn: Callable) -> Callable:
    """Expected failures (unknown document, bad location...) become ToolError: the model sees the message and
    can correct its call. Anything else is a bug: the SDK hides the details and logs the traceback."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (NotesError, EvalsError) as e:
            raise ToolError(str(e)) from e
    return wrapper


def build_server(notes: Notes | None = None, evals: Evals | None = None) -> MCPServer:
    notes = notes or Notes()
    evals = evals or Evals()
    mcp = MCPServer(name="studyrag", title="StudyRAG notes", version=__version__, instructions=INSTRUCTIONS,
                    description="Search, read and fact-check your own study material; read evaluation runs.")

    # ------------------------------------------------------------------ study-notes tools
    @mcp.tool(title="Search my notes", annotations=READ_ONLY)
    @as_tool_errors
    def search_notes(
        query: Annotated[str, Field(min_length=1, max_length=500,
                                    description="What to look for, in natural language or keywords")],
        top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K, description="How many passages to return")] = 4,
        document: Annotated[str | None, Field(description="Limit to one document (title or part of it)")] = None,
    ) -> dict[str, Any]:
        """Hybrid search (keywords + meaning, then a cross-encoder reranker) over the user's indexed study
        material. Returns numbered passages with citations and relevance, and `found`: false when nothing
        is relevant enough to answer from."""
        return notes.search(query, top_k, document)

    @mcp.tool(title="List my documents", annotations=READ_ONLY)
    @as_tool_errors
    def list_documents() -> dict[str, Any]:
        """All indexed documents with their size (chunks, words) and how many pages needed OCR."""
        return notes.list_documents()

    @mcp.tool(title="Outline a document", annotations=READ_ONLY)
    @as_tool_errors
    def document_outline(
        document: Annotated[str, Field(min_length=1, description="Document title (or part of it) or doc_id")],
    ) -> dict[str, Any]:
        """The slide titles / section headings of one document, in order, with their locations."""
        return notes.outline(document)

    @mcp.tool(title="Read a slide or page", annotations=READ_ONLY)
    @as_tool_errors
    def get_passage(
        document: Annotated[str, Field(min_length=1, description="Document title (or part of it) or doc_id")],
        location: Annotated[int, Field(ge=1, description="Slide or page number")],
    ) -> dict[str, Any]:
        """The full text of the passage at a slide/page number, e.g. to read around a search hit."""
        return notes.get_passage(document, location)

    @mcp.tool(title="Check an answer against its sources", annotations=READ_ONLY)
    @as_tool_errors
    def check_answer(
        answer: Annotated[str, Field(min_length=1, max_length=8000,
                                     description="The answer to check, with [n] citations")],
        chunk_ids: Annotated[list[str], Field(min_length=1, max_length=MAX_TOP_K,
                                              description="chunk_id of source [1], [2], ... in order")],
    ) -> dict[str, Any]:
        """Score every sentence of an answer against the passages it cites (StudyRAG's verifier: word overlap
        plus embedding similarity). Lists unsupported sentences so they can be fixed or removed."""
        return notes.check_answer(answer, chunk_ids)

    # ------------------------------------------------------------------ evaluation tools
    @mcp.tool(title="List evaluation runs", annotations=READ_ONLY)
    @as_tool_errors
    def list_eval_runs() -> dict[str, Any]:
        """Saved evalkit runs (project 2), named 'suite/run'."""
        return evals.list_runs()

    @mcp.tool(title="Summarize an evaluation run", annotations=READ_ONLY)
    @as_tool_errors
    def eval_run_summary(
        run: Annotated[str, Field(min_length=1, description="Run name 'suite/run' from list_eval_runs")],
    ) -> dict[str, Any]:
        """Metrics of one run with 95% confidence intervals, its configuration and size."""
        return evals.summary(run)

    @mcp.tool(title="Compare two evaluation runs", annotations=READ_ONLY)
    @as_tool_errors
    def compare_eval_runs(
        base: Annotated[str, Field(min_length=1, description="Baseline run 'suite/run'")],
        new: Annotated[str, Field(min_length=1, description="New run 'suite/run'")],
        metrics: Annotated[list[str] | None, Field(description="Only these metrics (default: all shared)")] = None,
    ) -> dict[str, Any]:
        """Paired comparison of two runs item by item: change per metric, 95% CI, p-value (McNemar or paired
        bootstrap), and which items got better or worse. Says whether a difference is real or noise."""
        return evals.compare(base, new, metrics)

    # ------------------------------------------------------------------ resources
    @mcp.resource("notes://documents", name="documents", mime_type="application/json",
                  description="The indexed documents")
    def documents_resource() -> dict[str, Any]:
        return notes.list_documents()

    @mcp.resource("notes://documents/{document}", name="document-text", mime_type="text/markdown",
                  description="Full text of one document (title or doc_id), slide by slide")
    def document_resource(document: str) -> str:
        try:
            return notes.document_text(document)
        except NotesError as e:
            raise ResourceNotFoundError(str(e)) from e

    # ------------------------------------------------------------------ prompts
    @mcp.prompt(title="Explain a topic from my notes")
    def explain_from_notes(topic: str) -> str:
        """Explain a topic using only the user's own material, with citations."""
        return (f"Explain \"{topic}\" to me as a student, using ONLY my study notes. First call search_notes "
                f"with a good query (try a second query if the first finds little). Write a clear explanation with "
                f"[n] citations to the passages, then call check_answer and fix any unsupported sentence. If my "
                f"notes don't cover it, say so.")

    @mcp.prompt(title="Quiz me")
    def quiz_me(topic: str, questions: str = "5") -> str:
        """Generate practice questions from the user's notes, one at a time, and grade the answers."""
        return (f"Quiz me on \"{topic}\" from my study notes. Use search_notes (and get_passage to read more) to "
                f"find the material, then ask me {questions} questions ONE AT A TIME, mixing multiple-choice and "
                f"short-answer. After each answer, grade it, explain using the passage, and cite it. Only use "
                f"facts from my notes.")

    @mcp.prompt(title="Revise a lecture")
    def revise_document(document: str) -> str:
        """A structured revision sheet for one lecture/document."""
        return (f"Make a revision sheet for \"{document}\". Call document_outline, then search_notes or get_passage "
                f"for each main heading. Give: key definitions, the main points per section, and 5 likely exam "
                f"questions, all with citations to the slides.")

    return mcp


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="studyrag-mcp", description=__doc__.split("\n")[0])
    p.add_argument("--http", action="store_true", help="serve Streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--version", action="version", version=f"studyrag-mcp {__version__}")
    a = p.parse_args(argv)
    # stdout belongs to the protocol on stdio: every log line goes to stderr
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    server = build_server()
    if a.http:
        server.run("streamable-http", host=a.host, port=a.port)
    else:
        server.run()


if __name__ == "__main__":
    main()
