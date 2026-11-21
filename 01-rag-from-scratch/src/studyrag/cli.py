"""Command-line interface.

    python -m studyrag sync data/my_docs          index a folder (new/changed/removed files)
    python -m studyrag list                       show indexed documents
    python -m studyrag search "what is seo"       retrieval only, with the full ranking trace
    python -m studyrag ask "what is seo"          grounded answer with citations
    python -m studyrag mcq -n 5 --doc "Lec 2"     generate MCQs
    python -m studyrag flashcards -n 5 --topic "email marketing"
    python -m studyrag summary --doc "Lec 1"
"""

from __future__ import annotations

import argparse
import sys
import time

from .core import StudyRAG


def _doc_ids(rag: StudyRAG, names: list[str] | None) -> list[str] | None:
    if not names:
        return None
    ids = []
    for n in names:
        d = rag.library.find_doc(n)
        if d is None:
            sys.exit(f"No indexed document matches '{n}'. Try: python -m studyrag list")
        ids.append(d.doc_id)
    return ids


def cmd_sync(rag, a):
    t = time.perf_counter()
    report = rag.library.sync_folder(a.folder)
    for k, v in report.items():
        if v:
            print(f"{k:>9}: {', '.join(v)}")
    print(f"Library: {len(rag.library.docs)} documents, {len(rag.library.chunks)} chunks "
          f"({time.perf_counter() - t:.1f}s)")


def cmd_list(rag, a):
    if not rag.library.docs:
        print("Library is empty. Run: python -m studyrag sync data/my_docs")
    for d in rag.library.docs.values():
        print(f"{d.doc_id}  {d.doc_type:<5} {d.n_chunks:>4} chunks {d.n_words:>6} words  {d.title}")


def cmd_search(rag, a):
    res = rag.retriever.retrieve(a.query, top_k=a.k, doc_ids=_doc_ids(rag, a.doc), mode=a.mode)
    print(f"timings (ms): { {k: round(v, 1) for k, v in res.timings_ms.items()} }\n")
    for i, h in enumerate(res.hits, 1):
        rr = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"
        print(f"#{i}  rerank={rr}  rrf={h.rrf_score:.4f}  bm25_rank={h.bm25_rank}  vector_rank={h.vector_rank}")
        print(f"    {h.chunk.header}")
        print("    " + h.chunk.text[:220].replace("\n", " ") + ("..." if len(h.chunk.text) > 220 else "") + "\n")


def cmd_ask(rag, a):
    qa = rag.qa
    prep = qa.prepare(a.question, _doc_ids(rag, a.doc))
    raw = ""
    if not prep.refused_early:
        for piece in rag.llm.generate(prep.messages, max_tokens=400, temperature=rag.settings.temperature,
                                      stream=True):
            raw += piece
            print(piece, end="", flush=True)
        print()
    ans = qa.finalize(prep, raw)
    if ans.status == "not_found":
        print(ans.text)
    print(f"\nconfidence: {ans.confidence}   ({ans.seconds:.1f}s)")
    for n, h in enumerate(ans.sources, 1):
        mark = "*" if n in ans.cited else " "
        print(f" {mark}[{n}] {h.chunk.header}")
    for c in ans.verification.unsupported:
        print(f"  ! weakly supported: \"{c.sentence[:90]}\" (support {c.support:.2f})")


def cmd_mcq(rag, a):
    from .study.generate import generate_mcqs, select_passages
    passages = select_passages(rag.library, rag.retriever, a.n, _doc_ids(rag, a.doc), a.topic)
    for i, q in enumerate(generate_mcqs(rag.llm, passages, a.difficulty, rag.embedder), 1):
        print(f"\nQ{i}. {q.question}")
        for j, opt in enumerate(q.options):
            print(f"   {'ABCD'[j]}) {opt}")
        print(f"   Answer: {q.answer_letter} — {q.explanation}")
        print(f"   Source: {q.source.header}" + (f"   ⚠ {'; '.join(q.issues)}" if q.issues else ""))


def cmd_flashcards(rag, a):
    from .study.generate import generate_flashcards, select_passages
    passages = select_passages(rag.library, rag.retriever, a.n, _doc_ids(rag, a.doc), a.topic)
    for c in generate_flashcards(rag.llm, passages, rag.embedder):
        print(f"\n[{c.front}]\n  {c.back}\n  ({c.source.header})")


def cmd_summary(rag, a):
    from .study.summarize import outline, summarize_document, summarize_topic
    if a.topic:
        stream, chunks = summarize_topic(rag.llm, rag.retriever, a.topic, _doc_ids(rag, a.doc))
        for piece in stream:
            print(piece, end="", flush=True)
        print("\n\nSources:\n" + "\n".join(f"[{i}] {c.header}" for i, c in enumerate(chunks, 1)))
        return
    if not a.doc:
        sys.exit("Give --doc NAME (whole-document summary) or --topic TEXT")
    doc = rag.library.find_doc(a.doc[0])
    chunks = rag.library.chunks_of(doc.doc_id)
    print("Outline:\n  " + "\n  ".join(outline(chunks)) + "\n\nSummary:\n")
    for piece in summarize_document(rag.llm, doc.title, chunks,
                                    on_progress=lambda i, n: print(f"(reading part {i}/{n}...)")):
        print(piece, end="", flush=True)
    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="studyrag", description="Private study assistant over your own documents.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sync", help="index a folder"); s.add_argument("folder"); s.set_defaults(fn=cmd_sync)
    s = sub.add_parser("list", help="list indexed documents"); s.set_defaults(fn=cmd_list)

    s = sub.add_parser("search", help="retrieval only (debug view)")
    s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    s.add_argument("--doc", action="append"); s.set_defaults(fn=cmd_search)

    s = sub.add_parser("ask", help="ask a question"); s.add_argument("question")
    s.add_argument("--doc", action="append"); s.set_defaults(fn=cmd_ask)

    for name, fn in (("mcq", cmd_mcq), ("flashcards", cmd_flashcards)):
        s = sub.add_parser(name); s.add_argument("-n", type=int, default=5)
        s.add_argument("--doc", action="append"); s.add_argument("--topic")
        s.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
        s.set_defaults(fn=fn)

    s = sub.add_parser("summary"); s.add_argument("--doc", action="append"); s.add_argument("--topic")
    s.set_defaults(fn=cmd_summary)

    a = p.parse_args(argv)
    a.fn(StudyRAG(), a)


if __name__ == "__main__":
    main()
