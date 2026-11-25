"""E4's deterministic parts: splitting a note into claims, and the number-changing sensitivity probe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from digest_eval import corrupt, note_claims, sentences, source_text  # noqa: E402

NOTE = """# A release

*2026-11-25 · drafted by Scout (my research agent) from the sources below, reviewed by me*

## What happened
Toolz 2.0 was released on 3 May 2026. [1] It adds streaming to every client. It is stable [2].

## Why it matters for my projects
You could stream answers in your RAG demo.

## Key details
- Streaming works over HTTP and stdio [1]
- The release drops Python 3.9,
  which reached end of life [2]

## Try it
Install it and stream one answer, as the docs show [1].

## Sources
1. [Notes](https://example.org/notes)
"""


def test_claims_keep_sections_and_citations():
    claims = note_claims(NOTE)
    assert [(c["section"].split()[0], c["cites"]) for c in claims] == [
        ("what", [1]), ("what", []), ("key", [1]), ("key", [2]), ("try", [1])]
    assert claims[0]["text"] == "Toolz 2.0 was released on 3 May 2026."
    assert claims[3]["text"] == "The release drops Python 3.9, which reached end of life"


def test_advice_without_citations_is_not_a_claim_and_short_bits_are_skipped():
    texts = [c["text"] for c in note_claims(NOTE)]
    assert not any("RAG demo" in t for t in texts)               # "Why it matters", uncited
    assert not any(t.startswith("It is stable") for t in texts)  # 4 words or fewer


def test_sentences_keep_a_citation_after_the_full_stop_and_ignore_abbreviations():
    assert sentences("Shipped in v1.2. [2] Then tools, e.g. Roots, went away.") == [
        "Shipped in v1.2. [2]", "Then tools, e.g. Roots, went away."]


def test_corrupt_changes_a_year_first_else_the_first_number():
    assert corrupt("Released in May 2026 with 4 models.") == "Released in May 2024 with 4 models."
    assert corrupt("It has 4 models and 1.5 GB weights.") == "It has 12 models and 1.5 GB weights."
    assert corrupt("Version 1.37 of the server.") == "Version 4.37 of the server."
    assert corrupt("No numbers here at all.") is None


def test_source_text_uses_only_what_the_agent_saw():
    ev = [{"n": 1, "url": "u1", "title": "A", "read": True, "snippet": "s1", "passages": ["p1", "p1b"]},
          {"n": 2, "url": "u2", "title": "B", "read": False, "snippet": "s2", "passages": []}]
    assert "p1b" in source_text(ev, [1]) and "u2" not in source_text(ev, [1])
    assert "only the search snippet" in source_text(ev, [2])
    assert "u2" not in source_text(ev, None)                     # uncited claims: all *read* sources
