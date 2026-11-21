"""StudyRAG web app.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import html
import json
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studyrag.core import StudyRAG  # noqa: E402
from studyrag.ingest import SUPPORTED  # noqa: E402
from studyrag.study.generate import (generate_flashcards, generate_mcqs,  # noqa: E402
                                     generate_short_questions, select_passages)
from studyrag.study.quiz import (flashcards_to_anki_csv, grade_short_answer,  # noqa: E402
                                 mcqs_to_markdown, short_questions_to_markdown)
from studyrag.study.summarize import outline, summarize_document, summarize_topic  # noqa: E402

UPLOAD_DIR = ROOT / "data" / "my_docs"

st.set_page_config(page_title="StudyRAG", page_icon="📚", layout="wide")
st.markdown("""
<style>
.cite {display:inline-block; padding:0 6px; margin:0 1px; border-radius:8px; font-size:0.75em;
       font-weight:600; text-decoration:none !important; background:rgba(99,102,241,.15); color:#6366f1 !important;}
.badge {display:inline-block; padding:2px 10px; border-radius:10px; font-size:0.8em; font-weight:600;}
.badge-high {background:rgba(34,197,94,.18); color:#16a34a;}
.badge-medium {background:rgba(234,179,8,.20); color:#a16207;}
.badge-low, .badge-none {background:rgba(239,68,68,.15); color:#dc2626;}
.src-anchor {scroll-margin-top: 80px;}
.card {border:1px solid rgba(128,128,128,.3); border-radius:12px; padding:14px 16px; margin-bottom:10px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading embedding + reranker models and your library...")
def get_rag() -> StudyRAG:
    return StudyRAG()


rag = get_rag()
lib = rag.library
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("mcqs", [])
ss.setdefault("shortqs", [])
ss.setdefault("cards", [])


# ======================================================================= sidebar
with st.sidebar:
    st.title("📚 StudyRAG")
    st.caption("Answers only from *your* study material — with sources.")

    uploads = st.file_uploader("Add documents", type=[e.lstrip(".") for e in sorted(SUPPORTED)],
                               accept_multiple_files=True)
    if uploads and st.button(f"Index {len(uploads)} file(s)", type="primary", use_container_width=True):
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with st.status("Indexing...", expanded=True) as status:
            for up in uploads:
                path = UPLOAD_DIR / Path(up.name).name
                path.write_bytes(up.getbuffer())
                t = time.perf_counter()
                try:
                    _, how = lib.add_file(path)
                    st.write(f"{up.name}: {how} ({time.perf_counter() - t:.1f}s)")
                except Exception as e:
                    st.write(f"❌ {up.name}: {e}")
            status.update(label="Done", state="complete")

    if st.button("↻ Sync data/my_docs folder", use_container_width=True):
        with st.spinner("Syncing..."):
            rep = lib.sync_folder(UPLOAD_DIR)
        st.toast(", ".join(f"{k}: {len(v)}" for k, v in rep.items() if v) or "Nothing to do")

    st.subheader(f"Library · {len(lib.docs)} docs · {len(lib.chunks)} chunks")
    docs = sorted(lib.docs.values(), key=lambda d: d.title)
    for d in docs:
        c1, c2 = st.columns([5, 1])
        icon = {"pptx": "📊", "pdf": "📕", "docx": "📄", "md": "📝", "txt": "📝"}.get(d.doc_type, "📄")
        ocr_note = f" · 📷 {d.n_ocr} via OCR" if d.n_ocr else ""
        c1.markdown(f"{icon} **{d.title}**  \n<small>{d.n_chunks} chunks · {d.n_words:,} words{ocr_note}</small>",
                    unsafe_allow_html=True)
        for w in d.warnings:
            c1.caption(f"⚠️ {w}")
        if c2.button("✕", key=f"rm-{d.doc_id}", help="Remove from library (the file is kept)"):
            lib.remove(d.doc_id)
            st.rerun()

    titles = {d.title: d.doc_id for d in docs}
    picked = st.multiselect("Search in (empty = all documents)", list(titles))
    scope = [titles[t] for t in picked] or None

    with st.expander("⚙️ Settings"):
        s = rag.settings
        backend = st.radio("Language model", ["llamacpp", "openai"], index=0 if s.llm_backend == "llamacpp" else 1,
                           format_func=lambda b: "Local (offline, private)" if b == "llamacpp" else "API (faster, online)")
        if backend == "openai":
            s.api_base_url = st.text_input("API base URL", s.api_base_url)
            s.api_model = st.text_input("Model", s.api_model)
            s.api_key = st.text_input("API key", s.api_key, type="password")
        if backend != s.llm_backend:
            s.llm_backend = backend
            rag.reload_llm()
        if st.button("Apply API settings"):
            rag.reload_llm()
        s.top_k = st.slider("Passages per answer", 2, 8, s.top_k)
        st.caption(f"Local model: `{s.llm_model_file}` · embeddings: `{s.embedding_model}`")


def need_llm() -> bool:
    ok, msg = rag.llm_ready
    if not ok:
        st.error(msg)
    return ok


def need_docs() -> bool:
    if not lib.chunks:
        st.info("Your library is empty. Upload files in the sidebar or put them in `data/my_docs` and click Sync.")
        return False
    return True


def link_citations(text: str, key: str, hits) -> str:
    """[2] -> a small badge that shows the source on hover and jumps to it on click."""
    def repl(m):
        out = []
        for n in re.split(r"\s*[,;]\s*", m.group(1)):
            i = int(n)
            tip = html.escape(hits[i - 1].chunk.header) if 1 <= i <= len(hits) else "unknown source"
            out.append(f'<a class="cite" href="#src-{key}-{i}" title="{tip}">{i}</a>')
        return "".join(out)
    return re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", repl, text)


def render_answer(entry: dict, key: str):
    ans = entry["answer"]
    if ans.status == "not_found":
        st.markdown(f"🔎 **{ans.text}**")
        if ans.sources:
            # Refusal is decided by a relevance score, so it can occasionally be wrong (e.g. a table the
            # reranker reads poorly). Showing the closest passages lets the student check for themselves.
            st.caption("No passage was relevant enough to answer from. Closest passages, in case they help:")
            for h in ans.sources[:2]:
                rel = "" if h.rerank_score is None else f" · relevance {h.rerank_score:.2f}"
                with st.expander(f"{h.chunk.header}{rel}"):
                    st.text(h.chunk.text)
        return
    st.markdown(link_citations(ans.text, key, ans.sources), unsafe_allow_html=True)
    labels = {"high": "High confidence", "medium": "Medium confidence", "low": "Low confidence"}
    ver = ans.verification
    st.markdown(f'<span class="badge badge-{ans.confidence}">{labels.get(ans.confidence, "")}</span> '
                f'<small>· {ver.supported_ratio:.0%} of statements verified against sources · {ans.seconds:.1f}s</small>',
                unsafe_allow_html=True)
    for c in ver.unsupported:
        st.warning(f"Not clearly supported by the sources: “{c.sentence[:160]}”", icon="⚠️")
    with st.expander(f"Sources ({len(ans.cited)} cited of {len(ans.sources)})"):
        for n, h in enumerate(ans.sources, 1):
            star = "✅ cited" if n in ans.cited else "not cited"
            st.markdown(f'<div id="src-{key}-{n}" class="src-anchor"></div><b>[{n}] {html.escape(h.chunk.header)}</b> · {star}',
                        unsafe_allow_html=True)
            st.text(h.chunk.text)
    with st.expander("🔍 How I found this"):
        r = ans.retrieval
        st.caption("Timings (ms): " + ", ".join(f"{k} {v:.0f}" for k, v in r.timings_ms.items()))
        st.dataframe([{"#": i, "source": h.chunk.header, "BM25 rank": h.bm25_rank, "vector rank": h.vector_rank,
                       "RRF": round(h.rrf_score, 4),
                       "rerank prob": None if h.rerank_score is None else round(h.rerank_score, 3)}
                      for i, h in enumerate(r.hits, 1)], hide_index=True, use_container_width=True)


def source_caption(chunk):
    st.caption(f"📍 {chunk.doc_title} · {chunk.location}")


tab_ask, tab_sum, tab_mcq, tab_short, tab_cards = st.tabs(
    ["💬 Ask", "📝 Summarize", "❓ MCQ Practice", "✍️ Short Answer", "🃏 Flashcards"])

# ======================================================================= Ask
with tab_ask:
    for i, entry in enumerate(ss.chat):
        with st.chat_message("user"):
            st.write(entry["q"])
        with st.chat_message("assistant"):
            render_answer(entry, f"m{i}")
    question = st.chat_input("Ask anything about your documents...")
    if question and need_docs() and need_llm():
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            qa = rag.qa
            prep = qa.prepare(question, scope)
            raw = ""
            if not prep.refused_early:
                with st.spinner("Reading the relevant passages..."):
                    stream = rag.llm.generate(prep.messages, max_tokens=400, temperature=rag.settings.temperature,
                                              stream=True)
                    first = next(stream, "")
                placeholder = st.empty()
                raw = first
                for piece in stream:
                    raw += piece
                    placeholder.markdown(raw + "▌")
                placeholder.empty()
            entry = {"q": question, "answer": qa.finalize(prep, raw)}
            ss.chat.append(entry)
            render_answer(entry, f"m{len(ss.chat) - 1}")
    if ss.chat and st.button("Clear chat"):
        ss.chat = []
        st.rerun()

# ======================================================================= Summarize
with tab_sum:
    if need_docs():
        mode = st.radio("Summarize", ["A whole document", "A topic"], horizontal=True)
        if mode == "A whole document":
            title = st.selectbox("Document", list(titles))
            chunks = lib.chunks_of(titles[title])
            with st.expander("Outline (instant, from slide titles / headings)", expanded=True):
                st.markdown("\n".join(f"- {line}" for line in outline(chunks)) or "_No headings found._")
            if st.button("Generate summary", type="primary") and need_llm():
                prog = st.empty()
                st.write_stream(summarize_document(rag.llm, title, chunks,
                                                   on_progress=lambda i, n: prog.caption(f"Reading part {i} of {n}...")))
                prog.empty()
        else:
            topic = st.text_input("Topic", placeholder="e.g. email marketing metrics")
            if st.button("Summarize topic", type="primary") and topic and need_llm():
                stream, used = summarize_topic(rag.llm, rag.retriever, topic, scope)
                st.write_stream(stream)
                st.caption("Sources: " + " · ".join(f"[{i}] {c.header}" for i, c in enumerate(used, 1)))


def generation_controls(key: str, default_n: int = 5):
    c1, c2, c3 = st.columns([1, 1, 2])
    n = c1.number_input("How many", 1, 20, default_n, key=f"{key}-n")
    diff = c2.selectbox("Difficulty", ["easy", "medium", "hard"], index=1, key=f"{key}-d")
    topic = c3.text_input("Topic (optional — empty = cover the selected documents evenly)", key=f"{key}-t")
    return int(n), diff, topic.strip() or None


# ======================================================================= MCQs
with tab_mcq:
    if need_docs():
        n, diff, topic = generation_controls("mcq")
        st.caption("Generation runs on your CPU: expect about a minute per question with the default local model "
                   "on an older laptop (measured: 66 s on a 2015 i5), much less on newer CPUs or the API backend. "
                   "Each question appears as soon as it is ready.")
        if st.button("Generate MCQs", type="primary") and need_llm():
            ss.mcqs, ss.mcq_answers, ss.mcq_submitted = [], {}, False
            passages = select_passages(lib, rag.retriever, n, scope, topic)
            bar = st.progress(0.0, text="Writing questions...")
            for q in generate_mcqs(rag.llm, passages, diff, rag.embedder):
                ss.mcqs.append(q)
                bar.progress(len(ss.mcqs) / n, text=f"{len(ss.mcqs)}/{n}: {q.question[:70]}")
            bar.empty()
        if ss.mcqs:
            exam = st.toggle("Exam mode (hide answers until you submit)", key="exam")
            ss.setdefault("mcq_answers", {})
            for i, q in enumerate(ss.mcqs):
                with st.container(border=True):
                    st.markdown(f"**Q{i + 1}. {q.question}**")
                    choice = st.radio("Your answer", range(4), index=None, key=f"mcq-{i}",
                                      format_func=lambda j, q=q: f"{'ABCD'[j]}) {q.options[j]}",
                                      label_visibility="collapsed")
                    ss.mcq_answers[i] = choice
                    if (not exam and choice is not None) or (exam and ss.get("mcq_submitted")):
                        if choice == q.answer_index:
                            st.success(f"Correct! {q.explanation}")
                        else:
                            st.error(f"Answer: {q.answer_letter}) {q.options[q.answer_index]} — {q.explanation}")
                        source_caption(q.source)
                        if q.issues:
                            st.caption("⚠️ Auto-check: " + "; ".join(q.issues))
            if exam and st.button("Submit exam", type="primary"):
                ss.mcq_submitted = True
                st.rerun()
            if exam and ss.get("mcq_submitted"):
                score = sum(ss.mcq_answers.get(i) == q.answer_index for i, q in enumerate(ss.mcqs))
                st.metric("Score", f"{score}/{len(ss.mcqs)}", f"{score / len(ss.mcqs):.0%}")
            c1, c2 = st.columns(2)
            c1.download_button("⬇ Markdown", mcqs_to_markdown(ss.mcqs), "mcqs.md")
            c2.download_button("⬇ JSON", json.dumps([q.to_dict() for q in ss.mcqs], indent=2), "mcqs.json")

# ======================================================================= Short answer
with tab_short:
    if need_docs():
        n, diff, topic = generation_controls("short", 3)
        if st.button("Generate questions", type="primary", key="gen-short") and need_llm():
            ss.shortqs = []
            passages = select_passages(lib, rag.retriever, n, scope, topic)
            with st.spinner("Writing questions..."):
                for q in generate_short_questions(rag.llm, passages, diff, rag.embedder):
                    ss.shortqs.append(q)
        for i, q in enumerate(ss.shortqs):
            with st.container(border=True):
                st.markdown(f"**Q{i + 1}. {q.question}**")
                student = st.text_area("Your answer", key=f"sa-{i}", height=80)
                c1, c2 = st.columns([1, 4])
                if c1.button("Grade", key=f"grade-{i}") and need_llm():
                    with st.spinner("Grading..."):
                        g = grade_short_answer(rag.llm, q.question, q.answer, student, q.source)
                    icon = {"correct": "✅", "partially_correct": "🟡", "incorrect": "❌"}[g.verdict]
                    st.markdown(f"{icon} **{g.verdict.replace('_', ' ').title()}** — {g.feedback}")
                with c2.expander("Model answer"):
                    st.write(q.answer)
                    source_caption(q.source)
        if ss.shortqs:
            st.download_button("⬇ Markdown", short_questions_to_markdown(ss.shortqs), "short_questions.md")

# ======================================================================= Flashcards
with tab_cards:
    if need_docs():
        n, _, topic = generation_controls("cards", 4)
        st.caption("Each passage yields 1-3 cards.")
        if st.button("Generate flashcards", type="primary", key="gen-cards") and need_llm():
            ss.cards = []
            passages = select_passages(lib, rag.retriever, n, scope, topic)
            with st.spinner("Writing flashcards..."):
                for c in generate_flashcards(rag.llm, passages, rag.embedder):
                    ss.cards.append(c)
        cols = st.columns(2)
        for i, c in enumerate(ss.cards):
            with cols[i % 2].container(border=True):
                st.markdown(f"#### {c.front}")
                with st.expander("Flip"):
                    st.write(c.back)
                    source_caption(c.source)
        if ss.cards:
            st.download_button("⬇ Anki CSV (front;back)", flashcards_to_anki_csv(ss.cards), "flashcards.csv")
