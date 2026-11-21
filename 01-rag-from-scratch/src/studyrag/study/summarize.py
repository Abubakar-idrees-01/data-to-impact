"""Summaries: of a whole document (map-reduce) or of a topic across documents (retrieve + cite).

Map-reduce exists because a lecture may not fit in the model's context window (and long prompts
are slow on CPU): we summarise consecutive groups of chunks ("map"), then merge the partial
notes into one structured summary ("reduce"). A short document skips straight to one call.
"""

from __future__ import annotations

from typing import Iterator

from ..chunking import Chunk
from ..prompts import (SUMMARY_SYSTEM, format_sources, summary_map_user, summary_reduce_user,
                       topic_summary_user)


def _batches(chunks: list[Chunk], max_words: int) -> list[list[Chunk]]:
    out, cur, words = [], [], 0
    for c in chunks:
        if cur and words + c.word_count > max_words:
            out.append(cur)
            cur, words = [], 0
        cur.append(c)
        words += c.word_count
    if cur:
        out.append(cur)
    return out


def outline(chunks: list[Chunk]) -> list[str]:
    """Instant, LLM-free outline from slide titles / section headings."""
    seen, lines = set(), []
    for c in chunks:
        h = c.heading.strip()
        if h and h.lower() not in seen:
            seen.add(h.lower())
            lines.append(f"{h} ({c.location.split(':')[0]})")
    return lines


def summarize_document(llm, title: str, chunks: list[Chunk], max_words_per_call: int = 900,
                       on_progress=None) -> Iterator[str]:
    """Yields text pieces of the final summary (streamed). on_progress(i, n) reports map steps."""
    batches = _batches(chunks, max_words_per_call)
    sys = {"role": "system", "content": SUMMARY_SYSTEM}
    if len(batches) == 1:
        yield from llm.generate([sys, {"role": "user", "content": summary_map_user(batches[0])}],
                                max_tokens=700, stream=True)
        return
    partials = []
    for i, batch in enumerate(batches, 1):
        if on_progress:
            on_progress(i, len(batches))
        partials.append(llm.generate([sys, {"role": "user", "content": summary_map_user(batch)}],
                                     max_tokens=350))
    yield from llm.generate([sys, {"role": "user", "content": summary_reduce_user(title, partials)}],
                            max_tokens=800, stream=True)


def summarize_topic(llm, retriever, topic: str, doc_ids: list[str] | None = None,
                    k: int = 5) -> tuple[Iterator[str], list[Chunk]]:
    res = retriever.retrieve(topic, top_k=k, doc_ids=doc_ids)
    chunks = [h.chunk for h in res.hits]
    msgs = [{"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": topic_summary_user(topic, chunks)}]
    return llm.generate(msgs, max_tokens=600, stream=True), chunks


__all__ = ["outline", "summarize_document", "summarize_topic", "format_sources"]
