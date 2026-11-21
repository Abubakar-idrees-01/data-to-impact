"""Quiz grading and export helpers."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from ..prompts import GRADE_SCHEMA, GRADE_SYSTEM, grade_user
from ..text import tokenize


@dataclass
class Grade:
    verdict: str        # correct | partially_correct | incorrect
    feedback: str
    method: str         # "llm" or "keyword" (fallback when the LLM call fails)


def grade_short_answer(llm, question: str, reference: str, student: str, chunk) -> Grade:
    if not student.strip():
        return Grade("incorrect", "No answer given. Model answer: " + reference, "keyword")
    try:
        raw = llm.generate_json([{"role": "system", "content": GRADE_SYSTEM},
                                 {"role": "user", "content": grade_user(question, reference, student, chunk)}],
                                GRADE_SCHEMA, max_tokens=200, temperature=0.0)
        verdict = raw.get("verdict", "")
        if verdict in {"correct", "partially_correct", "incorrect"}:
            return Grade(verdict, raw.get("feedback", "").strip(), "llm")
    except Exception:
        pass
    # fallback: keyword overlap with the model answer
    ref, got = set(tokenize(reference)), set(tokenize(student))
    overlap = len(ref & got) / len(ref) if ref else 0.0
    verdict = "correct" if overlap >= 0.6 else "partially_correct" if overlap >= 0.3 else "incorrect"
    return Grade(verdict, f"Keyword overlap with the model answer: {overlap:.0%}. Model answer: {reference}", "keyword")


def mcqs_to_markdown(mcqs, title: str = "Practice MCQs") -> str:
    lines = [f"# {title}", ""]
    for i, q in enumerate(mcqs, 1):
        lines.append(f"**{i}. {q.question}**\n")
        lines += [f"- {'ABCD'[j]}) {opt}" for j, opt in enumerate(q.options)]
        lines.append(f"\n<details><summary>Answer</summary>\n\n**{q.answer_letter}** — {q.explanation}  \n"
                     f"_Source: {q.source.doc_title}, {q.source.location}_\n</details>\n")
    return "\n".join(lines)


def flashcards_to_anki_csv(cards) -> str:
    """Anki: File > Import, separator ';' — front;back."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    for c in cards:
        w.writerow([c.front, f"{c.back} ({c.source.doc_title}, {c.source.location})"])
    return buf.getvalue()


def short_questions_to_markdown(items, title: str = "Short-answer questions") -> str:
    lines = [f"# {title}", ""]
    for i, q in enumerate(items, 1):
        lines.append(f"**{i}. {q.question}**\n\n<details><summary>Model answer</summary>\n\n{q.answer}  \n"
                     f"_Source: {q.source.doc_title}, {q.source.location}_\n</details>\n")
    return "\n".join(lines)
