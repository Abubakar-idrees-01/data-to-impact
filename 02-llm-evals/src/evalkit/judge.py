"""LLM-as-judge: a strong model grades outputs against a written rubric and returns structured JSON.

A judge is only useful if it agrees with humans, so every rubric here has a name and a version, and its
agreement with human labels is measured (experiments/judge_validation.py, docs/RESULTS.md).

Rubrics:
- faithfulness/claims   lists every unsupported or contradicting statement (with a quote), then a verdict
- faithfulness/direct   verdict only (cheaper, no evidence)
- correctness           compares an answer with a reference answer: correct / partially correct / incorrect
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .llm import APIError, ChatClient, parse_json

FAITHFULNESS_SYSTEM = (
    "You are a meticulous fact-checker. You decide whether a RESPONSE is fully supported by a SOURCE. "
    "You never use outside knowledge: a statement that is true in the real world but is not stated in, or "
    "directly inferable from, the SOURCE counts as unsupported."
)

_FAITHFULNESS_BODY = """TASK the writer was given:
{task}

SOURCE:
<<<
{source}
>>>

RESPONSE to check:
<<<
{response}
>>>

Check every statement in the RESPONSE against the SOURCE. A statement is a problem if it
- contradicts the SOURCE (a different number, name, date, place, negation or relationship), or
- adds information that the SOURCE does not contain and that cannot be directly inferred from it
  (including plausible-sounding details, opinions or background knowledge).
Do NOT flag: faithful paraphrases, faithful summaries, or statements that say the SOURCE lacks some information.
"""

FAITHFULNESS_CLAIMS_USER = _FAITHFULNESS_BODY + """
Reply with JSON only:
{{"problems": [{{"quote": "<exact words from the RESPONSE>", "type": "contradiction" or "unsupported", "why": "<short reason>"}}],
 "verdict": "supported" or "hallucinated"}}
The verdict is "hallucinated" if and only if "problems" is not empty."""

FAITHFULNESS_DIRECT_USER = _FAITHFULNESS_BODY + """
Reply with JSON only: {{"verdict": "supported" or "hallucinated"}}"""

CORRECTNESS_SYSTEM = ("You are a strict but fair teacher grading a student's answer against the reference answer. "
                      "Judge meaning, not wording.")

CORRECTNESS_USER = """QUESTION:
{question}

REFERENCE ANSWER:
{reference}

STUDENT ANSWER:
{response}

Grades:
- "correct": contains the main point(s) of the reference and nothing that contradicts it. Different wording and
  extra details that do not contradict the reference are fine.
- "partially_correct": gets part of the reference right but misses an important point.
- "incorrect": misses the main point, contradicts the reference, or does not answer.

Reply with JSON only:
{{"missing": ["<important points of the reference that the answer lacks>"],
  "contradictions": ["<statements that contradict the reference>"],
  "verdict": "correct" or "partially_correct" or "incorrect"}}"""

CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def _faithfulness_label(d: dict) -> dict:
    verdict = str(d.get("verdict", "")).strip().lower()
    problems = d.get("problems") or []
    if verdict not in ("supported", "hallucinated"):
        if "problems" not in d:
            raise ValueError(f"no verdict in {d}")
        verdict = "hallucinated" if problems else "supported"
    return {"label": verdict == "hallucinated", "problems": problems}


def _correctness_label(d: dict) -> dict:
    verdict = str(d.get("verdict", "")).strip().lower().replace(" ", "_")
    if verdict not in CORRECTNESS_SCORE:
        raise ValueError(f"unknown verdict {verdict!r}")
    return {"label": CORRECTNESS_SCORE[verdict], "verdict": verdict,
            "missing": d.get("missing") or [], "contradictions": d.get("contradictions") or []}


@dataclass(frozen=True)
class Rubric:
    name: str
    version: str
    system: str
    user: str                                   # str.format template
    parse: Callable[[dict], dict]               # JSON -> {"label": ..., ...}
    max_tokens: int = 700

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


RUBRICS = {
    "faithfulness/claims": Rubric("faithfulness/claims", "1", FAITHFULNESS_SYSTEM, FAITHFULNESS_CLAIMS_USER,
                                  _faithfulness_label),
    "faithfulness/direct": Rubric("faithfulness/direct", "1", FAITHFULNESS_SYSTEM, FAITHFULNESS_DIRECT_USER,
                                  _faithfulness_label, max_tokens=400),   # reasoning models think first
    "correctness": Rubric("correctness", "1", CORRECTNESS_SYSTEM, CORRECTNESS_USER, _correctness_label),
}


@dataclass
class Verdict:
    label: Any                                  # bool for faithfulness, 0/0.5/1 for correctness, None on error
    details: dict = field(default_factory=dict)
    raw: str = ""
    cached: bool = False
    error: str | None = None


class Judge:
    def __init__(self, client: ChatClient, rubric: Rubric | str, retries_on_bad_json: int = 1):
        self.client = client
        self.rubric = RUBRICS[rubric] if isinstance(rubric, str) else rubric
        self.retries_on_bad_json = retries_on_bad_json

    @property
    def id(self) -> str:
        return f"{self.rubric.id} · {self.client.model}"

    def __call__(self, **fields: str) -> Verdict:
        messages = [{"role": "system", "content": self.rubric.system},
                    {"role": "user", "content": self.rubric.user.format(**fields)}]
        raw, json_mode = "", True
        for attempt in range(self.retries_on_bad_json + 1):
            try:
                # attempt > 0: ask again without the cache (the cached reply was unparseable)
                res = self.client.chat(messages, temperature=0.0, max_tokens=self.rubric.max_tokens,
                                       json_mode=json_mode, use_cache=attempt == 0)
            except APIError as e:
                # some providers reject output that fails their JSON validation (e.g. cut off while
                # "thinking"): retry once in plain mode and parse the JSON ourselves
                if "json" in str(e).lower() and attempt < self.retries_on_bad_json:
                    json_mode, err = False, str(e)
                    continue
                return Verdict(label=None, error=str(e))
            raw = res.text
            try:
                parsed = self.rubric.parse(parse_json(raw))
                return Verdict(label=parsed.pop("label"), details=parsed, raw=raw, cached=res.cached)
            except (ValueError, json.JSONDecodeError, TypeError, AttributeError) as e:
                err = f"unparseable judge output: {e}"
        return Verdict(label=None, raw=raw, error=err)

    def many(self, items: Iterable[dict], on_progress: Callable[[int, Verdict], None] | None = None) -> list[Verdict]:
        out = []
        for i, fields in enumerate(items, 1):
            v = self(**fields)
            out.append(v)
            if on_progress:
                on_progress(i, v)
        return out
