# 1. Chunking: what is a "searchable unit"?

## Intuition
Imagine searching a textbook by holding the whole book up and asking "is SEO in here?" Useless: the answer
is always yes. Searching single words is also useless, because they have no meaning alone. We need
pieces that are **big enough to carry a complete idea** and **small enough to be about one thing**.
Those pieces are *chunks*.

Chunk size is a trade-off:

| Too small (one bullet) | Too large (whole lecture) |
|---|---|
| Precise, but loses context ("it improves conversion": *what* does?) | Has context, but its embedding is a blurry average of 20 topics |
| More chunks to search | The LLM prompt gets long and slow |

## The three strategies we built ([`chunking.py`](../../src/studyrag/chunking.py))

### Fixed window (baseline)
Take 180 words, then the next 180, overlapping by 30 so a sentence cut at a boundary appears whole in one of them.
It's simple and predictable, but it ignores structure: one chunk can end mid-slide and contain half of two unrelated slides.

### Structure-aware (our default)
Use the document's own units. A slide, a PDF page or a heading section is already "about one thing" because
the author made it so. Two fixes on top of that:
- **Split oversized units recursively**: try paragraph breaks first, then lines, then sentences, then words.
  The biggest natural boundary that makes the piece fit wins.
- **Merge thin neighbours**: a title-only slide ("Lecture 2: SEO") is useless alone, so it is merged with the
  next slide until the chunk has at least `min_chunk_words`.

### Semantic
Embed every sentence and compare each sentence with the next. Where similarity drops sharply (bottom 25% of
adjacent-pair similarities), the topic changed, so cut there. In theory it is the smartest; in practice it
costs one embedding per sentence and can split a list whose items are worded differently.

## Contextual headers
Each chunk is indexed as:
```
Digital Marketing Lec 1 — Slide 4: Online Market space
The market space in marketing is defined as a virtual market place...
```
The header adds the document and slide title to every chunk, so a slide whose body never repeats the topic
name ("It is an integration of numerous areas...") can still be found by searching for that topic.

## What we measured
The retrieval grid in [RESULTS.md](../RESULTS.md) compares all three strategies with the same questions
(hit@k, MRR). See [DECISIONS.md](../DECISIONS.md) for why the default is structure-aware.

## Interview questions
- *Why not just use the whole document?* Embedding dilution and prompt length; see the table above.
- *Why overlap in fixed chunking?* So information on a boundary is fully contained in at least one chunk.
- *How do you pick chunk size?* Measure it: build the index with each setting and compare retrieval metrics
  on a labelled question set. Don't guess.
- *What's the cost of semantic chunking?* One embedding per sentence at index time, and it is sensitive to
  the cut threshold.
