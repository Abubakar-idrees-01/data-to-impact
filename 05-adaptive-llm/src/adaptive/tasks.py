"""Questions with automatically checkable answers, from easy to hard, for evaluating routing between models.

Three families, chosen so that the smallest model gets the easy ones right and only bigger ones get the hard ones:

| family | example | graded by |
|---|---|---|
| `capital` (easy … tricky) | "What is the capital of Australia?" | the city (or an accepted spelling) appears in the answer |
| `arith` (medium) | "What is 37 × 43?" | the last number in the answer equals the result |
| `gsm8k` (hard) | grade-school word problems needing several steps | the number after "Answer:" equals the reference |

No LLM judge is needed, so the whole evaluation runs offline and is exactly reproducible. Capitals include the
commonly confused ones (Canberra not Sydney, Ankara not Istanbul, Naypyidaw not Yangon) so the easy family is not
trivially easy. Arithmetic is generated from a fixed seed; GSM8K uses the first problems of its test set
(OpenAI, MIT licence; `scripts/get_eval_data.sh`).
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GSM8K = ROOT / "data" / "raw" / "gsm8k_test.jsonl"

# country -> accepted answers (first = canonical). Only capitals that are unambiguous today.
CAPITALS = {
    "France": ["Paris"], "Japan": ["Tokyo"], "Germany": ["Berlin"], "Italy": ["Rome"], "Spain": ["Madrid"],
    "Egypt": ["Cairo"], "Russia": ["Moscow"], "China": ["Beijing", "Peking"], "Kenya": ["Nairobi"],
    "Norway": ["Oslo"], "Greece": ["Athens"], "Portugal": ["Lisbon"], "Poland": ["Warsaw"], "Peru": ["Lima"],
    "Argentina": ["Buenos Aires"], "South Korea": ["Seoul"], "Thailand": ["Bangkok"], "Iran": ["Tehran", "Teheran"],
    "Sweden": ["Stockholm"], "Finland": ["Helsinki"], "Iraq": ["Baghdad"], "Saudi Arabia": ["Riyadh"],
    "Chile": ["Santiago"], "Colombia": ["Bogotá", "Bogota"], "Vietnam": ["Hanoi", "Ha Noi"], "Philippines": ["Manila"],
    "India": ["New Delhi"], "Mexico": ["Mexico City", "Ciudad de México"],
    # commonly confused
    "Australia": ["Canberra"], "Canada": ["Ottawa"], "Turkey": ["Ankara"], "Brazil": ["Brasília", "Brasilia"],
    "Nigeria": ["Abuja"], "Pakistan": ["Islamabad"], "Morocco": ["Rabat"], "New Zealand": ["Wellington"],
    "Switzerland": ["Bern", "Berne"], "Kazakhstan": ["Astana", "Nur-Sultan"], "Myanmar": ["Naypyidaw", "Nay Pyi Taw", "Naypyitaw"],
    "Tanzania": ["Dodoma"], "Ivory Coast": ["Yamoussoukro"], "Mongolia": ["Ulaanbaatar", "Ulan Bator"],
    "Bhutan": ["Thimphu"], "Laos": ["Vientiane"], "Uzbekistan": ["Tashkent"], "Ethiopia": ["Addis Ababa"],
    "Ghana": ["Accra"], "Cambodia": ["Phnom Penh"],
}

SYSTEM = "You are a helpful assistant. Answer concisely."
GSM_INSTRUCTION = ("Solve the problem. Think step by step, briefly, then write the final answer on the last line as "
                   "'Answer: <number>'.")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))


def _number(s: str) -> float | None:
    """The last number in a string ('1,234.5' -> 1234.5); None if there is none."""
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


def final_answer(text: str) -> str:
    """The text after the last 'Answer:' (GSM8K format), else the last line."""
    m = re.findall(r"answer\s*[:：]\s*(.+)", text, re.I)
    return m[-1].strip() if m else (text.strip().splitlines() or [""])[-1]


def grade(task: dict, reply: str) -> bool:
    if task["family"] == "capital":
        r = _norm(reply)
        return any(_norm(a) in r for a in task["accept"])
    got = _number(final_answer(reply) if task["family"] == "gsm8k" else reply)
    return got is not None and abs(got - float(task["answer"])) < 1e-6


def capital_tasks() -> list[dict]:
    return [{"id": f"capital-{i:02d}", "family": "capital", "accept": acc, "answer": acc[0],
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"What is the capital of {country}? Reply with the city name only."}]}
            for i, (country, acc) in enumerate(CAPITALS.items())]


def arith_tasks(n: int = 40, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        kind = ["mul", "mul", "add", "sub"][i % 4]                 # half multiplications: the hard part for small models
        if kind == "mul":
            a, b = rng.randint(12, 99), rng.randint(12, 99)
            q, ans = f"What is {a} × {b}?", a * b
        elif kind == "add":
            a, b = rng.randint(100, 999), rng.randint(100, 999)
            q, ans = f"What is {a} + {b}?", a + b
        else:
            a, b = rng.randint(100, 999), rng.randint(10, 99)
            q, ans = f"What is {a} − {b}?", a - b
        out.append({"id": f"arith-{i:02d}", "family": "arith", "kind": kind, "answer": ans,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": q + " Reply with the number only."}]})
    return out


def gsm8k_tasks(n: int = 40) -> list[dict]:
    if not GSM8K.exists():
        raise FileNotFoundError(f"{GSM8K} missing: run bash scripts/get_eval_data.sh")
    out = []
    for i, line in enumerate(GSM8K.read_text().splitlines()[:n]):
        item = json.loads(line)
        ans = item["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k-{i:03d}", "family": "gsm8k", "answer": float(ans),
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": f"{item['question']}\n\n{GSM_INSTRUCTION}"}]})
    return out


def all_tasks(n_arith: int = 40, n_gsm: int = 40) -> list[dict]:
    return capital_tasks() + arith_tasks(n_arith) + gsm8k_tasks(n_gsm)
