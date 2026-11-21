"""Loaders keep locations; chunkers keep locations and respect size limits."""

from studyrag.chunking import chunk_fixed, chunk_semantic, chunk_structure, split_sentences
from studyrag.ingest import Block, Document, load_document
from studyrag.ingest.cleaning import clean_text, linearize_table


def test_clean_text_normalises_bullets_and_hyphenation():
    assert clean_text("• first\n▪  second\nmark-\neting") == "- first\n- second\nmarketing"


def test_pptx_loader_keeps_slide_numbers_titles_and_tables(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "What is SEO"
    s1.placeholders[1].text_frame.text = "Improving organic rankings"
    prs.slides.add_slide(prs.slide_layouts[6])                      # blank slide -> skipped
    s3 = prs.slides.add_slide(prs.slide_layouts[5])
    s3.shapes.title.text = "Channel metrics"
    table = s3.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "Channel", "CTR"
    table.cell(1, 0).text, table.cell(1, 1).text = "Email", "2.5%"
    s3.notes_slide.notes_text_frame.text = "Mention benchmarks"
    path = tmp_path / "lec.pptx"
    prs.save(path)

    doc = load_document(path)
    assert doc.doc_type == "pptx" and doc.title == "lec"
    assert [b.location for b in doc.blocks] == [1, 3]
    assert doc.blocks[0].heading == "What is SEO"
    assert "Improving organic rankings" in doc.blocks[0].text
    assert "Email: 2.5%." in doc.blocks[1].text and "Speaker notes: Mention benchmarks" in doc.blocks[1].text


def test_docx_loader_splits_sections_on_headings(tmp_path):
    import docx

    d = docx.Document()
    d.add_paragraph("Course intro text")
    d.add_heading("Week 1: Foundations", level=1)
    d.add_paragraph("Marketing mix and the 4 Ps")
    d.add_paragraph("Product", style="List Bullet")
    d.add_heading("Week 2: SEO", level=1)
    d.add_paragraph("Keywords and backlinks")
    path = tmp_path / "outline.docx"
    d.save(path)

    doc = load_document(path)
    assert [b.heading for b in doc.blocks] == ["", "Week 1: Foundations", "Week 2: SEO"]
    assert "- Product" in doc.blocks[1].text


def _doc(texts, kind="slide"):
    return Document("d1", "Lec", "lec.pptx", "pptx",
                    [Block(t, i, kind, t.split("\n")[0]) for i, t in enumerate(texts, 1)])


def test_fixed_chunks_cover_all_words_with_overlap():
    doc = _doc([" ".join(f"w{i}" for i in range(100))])
    chunks = chunk_fixed(doc, size=40, overlap=10)
    assert [c.word_count for c in chunks] == [40, 40, 40]
    assert chunks[1].text.split()[0] == "w30"                    # 10-word overlap
    assert chunks[-1].text.split()[-1] == "w99"


def test_structure_chunks_merge_thin_slides_and_split_long_ones():
    long = "Big topic\n" + " ".join(f"word{i}." for i in range(130))
    doc = _doc(["Title slide", "Agenda\n- SEO\n- Email", "SEO basics " + "content " * 30, long])
    chunks = chunk_structure(doc, max_words=60, min_words=15)
    assert chunks[0].loc_start == 1 and chunks[0].loc_end >= 2   # thin slides merged with neighbours
    assert all(c.word_count <= 60 + 5 for c in chunks)            # (+ a repeated heading)
    assert {c.loc_start for c in chunks if c.loc_start == 4}      # slide 4 split into several chunks
    assert sum(1 for c in chunks if c.loc_start == 4) >= 2
    assert chunks[0].location.startswith("Slides 1-")


def test_semantic_chunking_cuts_at_topic_shift(embedder):
    sents = ["SEO improves rankings in search.", "SEO uses keywords in search.", "SEO needs backlinks in search.",
             "Email lists need subscribers.", "Email campaigns need subscribers.", "Email open rate for subscribers."]
    doc = _doc(["\n".join(sents)], kind="section")
    chunks = chunk_semantic(doc, embedder.embed_documents, max_words=100, min_words=5, percentile=20)
    assert len(chunks) == 2
    assert chunks[0].text.startswith("SEO") and chunks[1].text.startswith("Email")


def test_split_sentences_handles_bullets_and_periods():
    assert split_sentences("First point. Second point.\n- bullet one\n- bullet two") == [
        "First point.", "Second point.", "- bullet one", "- bullet two"]


def test_linearize_table_repeats_headers_in_every_row():
    rows = [["Assessment", "Weight", "CLO"], ["Midterm", "25%", "1-3"], ["Final exam", "40%", "1-5"]]
    assert linearize_table(rows) == ["Midterm — Weight: 25%; CLO: 1-3.", "Final exam — Weight: 40%; CLO: 1-5."]


def test_linearize_table_key_value_layouts():
    assert linearize_table([["Instructor", "Dr X"], ["Room", "2.12"]]) == ["Instructor: Dr X.", "Room: 2.12."]
    side_by_side = [["Item", "Details", "Item", "Details"], ["Code", "DS-294", "Credits", "2"]]
    assert linearize_table(side_by_side) == ["Code: DS-294. Credits: 2."]
