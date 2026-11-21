"""End-to-end with fake models: library, incremental indexing, retrieval, QA, refusal, verification, MCQs."""

import random

from conftest import FakeLLM

from studyrag.answer import REFUSAL_TEXT, QAEngine
from studyrag.library import Library
from studyrag.retrieval import Retriever
from studyrag.study.generate import generate_mcqs, select_passages, validate_mcq
from studyrag.verify import extract_citations, strip_citations, verify_answer


def _build(settings, embedder, folder):
    lib = Library(settings, embedder)
    report = lib.sync_folder(folder)
    return lib, report


def test_sync_is_incremental_and_persistent(settings, embedder, docs_folder):
    lib, report = _build(settings, embedder, docs_folder)
    assert sorted(report["added"]) == ["email.md", "seo.md"]
    n_chunks = len(lib.chunks)
    assert lib.matrix.shape == (n_chunks, embedder.dim)

    report2 = Library(settings, embedder).sync_folder(docs_folder)       # reload from disk
    assert sorted(report2["unchanged"]) == ["email.md", "seo.md"]

    (docs_folder / "email.md").unlink()
    lib3 = Library(settings, embedder)
    assert lib3.sync_folder(docs_folder)["removed"] == ["email.md"]
    assert all(c.doc_title == "seo" for c in lib3.chunks)
    assert lib3.matrix.shape[0] == len(lib3.chunks)


def test_hybrid_retrieval_finds_the_right_section(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    r = Retriever(lib, embedder, None, settings)
    hit = r.retrieve("what are backlinks?", top_k=1).hits[0]
    assert "Backlinks" in hit.chunk.heading
    assert hit.bm25_rank is not None and hit.vector_rank is not None
    seo_id = next(d.doc_id for d in lib.docs.values() if d.title == "seo")
    only_seo = r.retrieve("open rate metrics", top_k=3, doc_ids=[seo_id]).hits
    assert all(h.chunk.doc_id == seo_id for h in only_seo)


def test_answer_with_citations_and_verification(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    llm = FakeLLM(["Backlinks are links from other websites [1]. Search engines treat quality backlinks as votes of trust [1]."])
    qa = QAEngine(Retriever(lib, embedder, None, settings), llm, embedder, settings)
    ans = qa.ask("What are backlinks?")
    assert ans.status == "answered" and ans.cited == [1]
    assert ans.verification.supported_ratio == 1.0
    assert "Sources:" in llm.calls[0][1]["content"] and "[1] (" in llm.calls[0][1]["content"]


def test_refusal_when_model_says_not_found(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    qa = QAEngine(Retriever(lib, embedder, None, settings), FakeLLM(["NOT_FOUND"]), embedder, settings)
    ans = qa.ask("Who founded Google?")
    assert ans.status == "not_found" and ans.text == REFUSAL_TEXT and ans.confidence == "none"


def test_verifier_flags_invented_claims(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    src = [c for c in lib.chunks if "Backlinks" in c.heading]
    ver = verify_answer("Backlinks are links from other websites [1]. "
                        "Google was founded in 1998 by Larry Page in California [1].", src, embedder, 0.55)
    assert [c.supported for c in ver.checks] == [True, False]


def test_citation_helpers():
    assert extract_citations("A [1]. B [2][3]. C [4, 5].") == [1, 2, 3, 4, 5]
    assert strip_citations("SEO matters [1][2].") == "SEO matters."


def test_mcq_validation_shuffles_and_keeps_correct_answer(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    chunk = next(c for c in lib.chunks if "Backlinks" in c.heading)
    raw = {"question": "What do search engines treat quality backlinks as?",
           "options": {"A": "Votes of trust", "B": "Spam signals", "C": "Paid ads", "D": "Meta tags"},
           "answer": "A", "explanation": "The passage says backlinks are votes of trust."}
    positions = {validate_mcq(raw, chunk, random.Random(s)).answer_index for s in range(20)}
    assert len(positions) > 1                                         # answer no longer always "A"
    q = validate_mcq(raw, chunk, random.Random(0))
    assert q.options[q.answer_index] == "Votes of trust" and q.grounded

    dup = dict(raw, options={"A": "x", "B": "x", "C": "y", "D": "z"})
    assert validate_mcq(dup, chunk, random.Random(0)) is None


def test_mcq_generation_skips_duplicates(settings, embedder, docs_folder):
    lib, _ = _build(settings, embedder, docs_folder)
    passages = select_passages(lib, None, 2, min_words=5)
    same = {"question": "What is email marketing?", "options": {"A": "Sending targeted messages to subscribers",
            "B": "Buying ads", "C": "Printing flyers", "D": "Cold calling"}, "answer": "A", "explanation": "From the passage."}
    llm = FakeLLM(json_outputs=[same, same, same])
    out = list(generate_mcqs(llm, passages, embedder=embedder, retries=1))
    assert len(out) == 1
