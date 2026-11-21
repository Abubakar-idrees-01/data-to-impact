"""Every prompt the app sends to the LLM, in one place.

Prompt design notes (small 1-4B models need more structure than big ones):
- Sources are numbered [1]..[k] and the model is told to cite after each sentence.
- A fixed refusal token (NOT_FOUND) is easier for a small model to produce reliably, and for us
  to detect, than a free-form "I don't know".
- Generation tasks (MCQs, flashcards) work on ONE passage at a time: shorter prompts are faster
  on CPU, and every generated item gets an exact source citation for free.
"""

from __future__ import annotations

NOT_FOUND = "NOT_FOUND"

QA_SYSTEM = f"""You are a study assistant. Answer the student's question using ONLY the numbered sources.
Rules:
- After each sentence or bullet point that uses a source, cite it with its number in square brackets, like [1] or [2][3]. Every bullet point must end with a citation.
- Answer the question directly; do not add background the question did not ask for.
- Use only facts stated in the sources. Do not add outside knowledge.
- If the sources do not contain the answer, reply with exactly: {NOT_FOUND}
- Be clear and concise. Use short bullet points for lists."""


def format_sources(chunks) -> str:
    return "\n\n".join(f"[{i}] ({c.header})\n{c.text}" for i, c in enumerate(chunks, 1))


def qa_user(question: str, chunks) -> str:
    return f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}"


# ------------------------------------------------------------------ summaries
SUMMARY_SYSTEM = """You summarise study material for a student revising for an exam.
Use ONLY the given material. Keep the key definitions, lists, examples and numbers."""


def summary_map_user(chunks) -> str:
    material = "\n\n".join(f"({c.location})\n{c.text}" for c in chunks)
    return ("Summarise the key points of this material as concise bullet points. "
            "End each bullet with its location in parentheses, e.g. (Slide 4).\n\n"
            f"Material:\n{material}")


def summary_reduce_user(title: str, partials: list[str]) -> str:
    joined = "\n\n".join(partials)
    return (f"Below are bullet-point notes from consecutive parts of '{title}'. Combine them into one "
            "well-organised study summary: a one-sentence overview, then 3-6 short sections with headings "
            "and bullets. Keep the locations in parentheses. Remove repetition.\n\n"
            f"Notes:\n{joined}")


def topic_summary_user(topic: str, chunks) -> str:
    return (f"Using ONLY the numbered sources, write a study summary about: {topic}\n"
            "Use short sections and bullets, and cite sources like [1] after each point.\n\n"
            f"Sources:\n\n{format_sources(chunks)}")


# ------------------------------------------------------------------ question generation
GEN_SYSTEM = """You are an experienced teacher writing exam questions from course material.
Every question and answer must be fully supported by the given passage. Never use outside facts."""


def _passage(chunk) -> str:
    return f'Passage (from {chunk.header}):\n"""\n{chunk.text}\n"""'


def _avoid(previous: list[str]) -> str:
    if not previous:
        return ""
    recent = "\n".join(f"- {q}" for q in previous[-8:])
    return f"\nDo NOT repeat or rephrase these already-asked questions:\n{recent}\n"


def mcq_user(chunk, difficulty: str, previous: list[str]) -> str:
    return (f"{_passage(chunk)}\n\nWrite ONE {difficulty} multiple-choice question that tests understanding of "
            "an important idea in this passage (not trivia about the slide itself).\n"
            "Rules:\n- Exactly 4 options A-D; exactly one is correct according to the passage.\n"
            "- The 3 wrong options must be plausible but clearly wrong according to the passage.\n"
            "- No 'All of the above' or 'None of the above'.\n"
            "- Put the options only in the options field, not inside the question text.\n"
            "- The explanation says why the answer is correct, using the passage. Refer to options by their "
            "content, not by letter.\n" + _avoid(previous))


def mcq_check_user(mcq) -> str:
    opts = "\n".join(f"{'ABCD'[j]}) {o}" for j, o in enumerate(mcq.options))
    return (f'Passage:\n"""\n{mcq.source.text}\n"""\n\nUsing only the passage, answer this question.\n'
            f"{mcq.question}\n{opts}")


MCQ_CHECK_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string", "enum": list("ABCD")}},
                    "required": ["answer"]}


MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "options": {
            "type": "object",
            "properties": {k: {"type": "string"} for k in "ABCD"},
            "required": list("ABCD"),
        },
        "answer": {"type": "string", "enum": list("ABCD")},
        "explanation": {"type": "string"},
    },
    "required": ["question", "options", "answer", "explanation"],
}


def short_question_user(chunk, difficulty: str, previous: list[str]) -> str:
    return (f"{_passage(chunk)}\n\nWrite ONE {difficulty} short-answer exam question about an important idea "
            "in this passage, plus a model answer of 1-3 sentences taken from the passage.\n" + _avoid(previous))


SHORT_Q_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
}


def flashcards_user(chunk, previous: list[str]) -> str:
    return (f"{_passage(chunk)}\n\nWrite 1 to 3 flashcards for the most important terms, definitions or facts "
            "in this passage. Front: a short term or question. Back: a concise answer from the passage.\n"
            + _avoid(previous))


FLASHCARD_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"front": {"type": "string"}, "back": {"type": "string"}},
                "required": ["front", "back"],
            },
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["cards"],
}


# ------------------------------------------------------------------ grading
GRADE_SYSTEM = """You are a fair examiner. Grade the student's answer against the model answer and the source passage.
Accept answers that are correct in meaning even if worded differently. Be brief and encouraging."""


def grade_user(question: str, reference: str, student: str, chunk) -> str:
    return (f"{_passage(chunk)}\n\nQuestion: {question}\nModel answer: {reference}\n"
            f"Student answer: {student}\n\nGrade it as correct, partially_correct or incorrect and give "
            "one or two sentences of feedback explaining what was right or missing.")


GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "feedback": {"type": "string"},
    },
    "required": ["verdict", "feedback"],
}
