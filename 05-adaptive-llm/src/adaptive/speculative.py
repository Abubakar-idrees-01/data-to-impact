"""Speculative decoding: a small model drafts tokens, the big model checks several of them in one pass.

    python -m adaptive.speculative --draft Qwen3.5-0.8B --target Qwen3.5-2B --k 4

**Why it can help on a CPU.** Decoding is memory-bound: one token costs one full read of the weights. Checking
k+1 tokens in a single batch reads the weights once too, so it costs barely more than one token. If a cheap draft
model guesses the next k tokens and the big model agrees with the first j of them, one big-model pass yields j + 1
tokens instead of 1.

**Greedy version (exact).** Both models decode greedily (argmax). Per round, with `last` = the newest emitted token:
1. the draft feeds `last` and produces d1 … dk one at a time;
2. the target feeds [last, d1 … dk] in one batch and gets its own next-token choice after every position;
3. accept drafts while they equal the target's choice; at the first mismatch, emit the target's choice instead
   (a "bonus" token: every round emits at least one token);
4. both models forget the rejected drafts.

The output is exactly what the target alone would produce with greedy decoding (up to tiny floating-point
differences between batched and single-token evaluation); a test and the benchmark check this token by token.

**Forgetting rejected drafts** needs the KV cache to drop its last entries. Transformers can (`kv_cache_seq_rm`).
Qwen3.5's linear-attention layers keep a running state that cannot be rewound (measured: removal returns False), so
for those models we snapshot the state before each round and, on a rejection, restore it and re-feed the accepted
tokens. That costs an extra pass, which is part of what the benchmark measures.

**Expected speed-up** with acceptance rate α, k drafts and draft/target cost ratio c (Leviathan et al., 2023):
tokens per round = (1 − α^(k+1)) / (1 − α); cost per round ≈ 1 + k·c (target passes). Speed-up = the ratio.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT.parent / "01-rag-from-scratch" / "models" / "llm"


def accept(drafts: list[int], target_choice: list[int]) -> tuple[int, int]:
    """(number of drafts accepted, the token to emit after them).

    target_choice[i] = the target's greedy choice after seeing the context + drafts[:i]  (len = len(drafts) + 1).
    """
    j = 0
    while j < len(drafts) and drafts[j] == target_choice[j]:
        j += 1
    return j, target_choice[j]


def expected_speedup(alpha: float, k: int, c: float) -> float:
    tokens = (k + 1) if alpha == 1 else (1 - alpha ** (k + 1)) / (1 - alpha)
    return tokens / (1 + k * c)


class Sequence:
    """One model decoding one sequence, through llama.cpp's low-level batch API.

    Invariant: the model has consumed every emitted token except the newest one (`feed` it next).
    """

    def __init__(self, path: Path, n_ctx: int = 2048, threads: int = 2):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads,
                         n_batch=512, verbose=False)
        self.vocab = self.llm.n_vocab()
        self.rewindable = None                       # set by probe_rewind()
        self.snapshot = None
        self.snapshot_bytes = 0
        self.passes = 0
        self.restores = 0

    @property
    def n(self) -> int:
        return self.llm.n_tokens

    def feed(self, tokens: list[int], all_logits: bool = False) -> np.ndarray:
        """Consume tokens; return the greedy next-token choice after the last one (or after each, if all_logits)."""
        llm, out = self.llm, []
        for i in range(0, len(tokens), llm.n_batch):
            chunk = tokens[i:i + llm.n_batch]
            last_chunk = i + llm.n_batch >= len(tokens)
            llm._batch.set_batch(batch=chunk, n_past=llm.n_tokens, logits_all=all_logits and last_chunk)
            llm._ctx.decode(llm._batch)
            llm.input_ids[llm.n_tokens:llm.n_tokens + len(chunk)] = chunk
            llm.n_tokens += len(chunk)
            self.passes += 1
            if last_chunk:
                rows = len(chunk) if all_logits else 1
                logits = np.ctypeslib.as_array(llm._ctx.get_logits(), shape=(rows * self.vocab,)).reshape(rows, -1)
                out = logits.argmax(-1)
        return out

    def reset(self) -> None:
        """Forget everything. (Llama.reset() only zeroes a counter; the cache entries must be removed too.)"""
        self.llm.n_tokens = 0
        self.llm._ctx.kv_cache_clear()

    def mark(self) -> None:
        """Snapshot this sequence's state (attention cache + recurrent state), for models that cannot rewind.

        Uses llama.cpp's per-sequence state copy, not Llama.save_state(): that one also copies a score row of the
        whole 248k vocabulary for every token so far, and scans a 508 MB array on every restore.
        """
        if self.rewindable is not False:
            return
        import ctypes
        import llama_cpp
        ctx = self.llm._ctx.ctx
        size = llama_cpp.llama_state_seq_get_size(ctx, 0)
        if self.snapshot is None or len(self.snapshot[0]) < size:
            self.snapshot = [(ctypes.c_uint8 * size)(), 0, 0]
        buf = self.snapshot[0]
        self.snapshot[1] = llama_cpp.llama_state_seq_get_data(ctx, buf, size, 0)
        self.snapshot[2] = self.n
        self.snapshot_bytes = self.snapshot[1]

    def rewind(self, keep: int, replay: list[int]) -> None:
        """Forget everything after the first `keep` tokens. `replay` = the tokens between the mark and `keep`."""
        if self.n == keep:
            return
        if self.rewindable:
            if not self.llm._ctx.kv_cache_seq_rm(-1, keep, -1):
                raise RuntimeError("cache removal failed")
            self.llm.n_tokens = keep
            return
        if self.rewindable is None or self.snapshot is None:
            raise RuntimeError("call probe_rewind() before decoding")
        import llama_cpp
        buf, nbytes, at = self.snapshot
        self.llm._ctx.kv_cache_clear()
        if not llama_cpp.llama_state_seq_set_data(self.llm._ctx.ctx, buf, nbytes, 0):
            raise RuntimeError("restoring the sequence state failed")
        self.llm.n_tokens = at
        assert at + len(replay) == keep
        self.restores += 1
        if replay:
            self.feed(replay)

    def probe_rewind(self) -> bool:
        """Can this model drop its last cache entries? (False for recurrent / hybrid models such as Qwen3.5.)"""
        self.reset()
        self.feed(self.llm.tokenize(b"one two three", add_bos=False))
        self.rewindable = bool(self.llm._ctx.kv_cache_seq_rm(-1, 1, -1))
        self.reset()
        return self.rewindable


@dataclass
class Result:
    tokens: list[int]
    seconds: float
    target_passes: int
    rounds: int
    accepted: int
    drafted: int

    @property
    def tok_s(self) -> float:
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    @property
    def acceptance(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "tok_s": round(self.tok_s, 2), "acceptance": round(self.acceptance, 3),
                "tokens": len(self.tokens)}


def greedy(target: Sequence, prompt: list[int], max_new: int, stop: set[int]) -> Result:
    """Baseline: the target alone, one token per pass, through the same code path."""
    t0, p0 = time.perf_counter(), target.passes
    choice = int(target.feed(prompt)[-1])
    out = []
    while len(out) < max_new:
        out.append(choice)
        if choice in stop:
            break
        choice = int(target.feed([choice])[-1])
    return Result(out, time.perf_counter() - t0, target.passes - p0, len(out), 0, 0)


def speculative(draft: Sequence, target: Sequence, prompt: list[int], max_new: int, stop: set[int], k: int = 4) -> Result:
    t0, p0 = time.perf_counter(), target.passes
    # Both models consume the prompt except its last token; that token is `last` for the first round, so the first
    # verification batch also produces the target's first choice.
    if len(prompt) > 1:
        target.feed(prompt[:-1])
        draft.feed(prompt[:-1])
    last = prompt[-1]
    out, rounds, accepted, drafted = [], 0, 0, 0
    while len(out) < max_new:
        base = target.n                              # = draft.n: everything before `last`
        target.mark()
        draft.mark()
        drafts, tok = [], last
        for _ in range(min(k, max_new - len(out))):  # 1. draft k tokens greedily
            tok = int(draft.feed([tok])[-1])
            drafts.append(tok)
            if tok in stop:
                break
        choices = [int(c) for c in target.feed([last] + drafts, all_logits=True)]     # 2. one verification pass
        j, bonus = accept(drafts, choices)          # 3.
        emitted = drafts[:j] + [bonus]
        rounds, accepted, drafted = rounds + 1, accepted + j, drafted + len(drafts)
        keep = base + 1 + j                          # 4. context + last + accepted drafts
        target.rewind(keep, [last] + drafts[:j])
        if draft.n > keep:
            draft.rewind(keep, [last] + drafts[:j])
        elif draft.n < keep:                         # all drafts accepted: the draft never consumed the last one
            draft.feed(drafts[draft.n - keep:])
        for t in emitted:
            out.append(t)
            if t in stop or len(out) >= max_new:
                return Result(out, time.perf_counter() - t0, target.passes - p0, rounds, accepted, drafted)
        last = bonus
    return Result(out, time.perf_counter() - t0, target.passes - p0, rounds, accepted, drafted)


PROMPTS = [
    "Explain in a short paragraph why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
    "List five tips for studying effectively, one line each.",
    "Summarise the water cycle for a ten-year-old in four sentences.",
]


def find(name: str) -> Path:
    hits = sorted(MODELS.glob(f"*{name}*.gguf")) or sorted(MODELS.glob(f"*{name.lower()}*.gguf"))
    if not hits:
        raise FileNotFoundError(f"no GGUF matching {name!r} in {MODELS}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="Qwen3.5-0.8B")
    ap.add_argument("--target", default="Qwen3.5-2B")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    from .runtime import chatml
    draft, target = Sequence(find(a.draft), threads=a.threads), Sequence(find(a.target), threads=a.threads)
    if draft.vocab != target.vocab:
        raise SystemExit(f"draft and target must share a tokenizer (vocab {draft.vocab} vs {target.vocab})")
    print(f"rewind: draft {draft.probe_rewind()}, target {target.probe_rewind()}")
    stop = {target.llm.token_eos(), *target.llm.tokenize(b"<|im_end|>", add_bos=False, special=True)}
    rows = []
    for text in PROMPTS:
        prompt = target.llm.tokenize(chatml([{"role": "user", "content": text}]).encode(), add_bos=False, special=True)
        target.reset()
        base = greedy(target, prompt, a.max_new, stop)
        row = {"prompt": text, "greedy": base.to_dict(), "spec": {}}
        print(f"\n{text[:60]}\n  target alone: {base.tok_s:5.2f} tok/s ({len(base.tokens)} tokens)")
        for k in a.k:
            target.reset()
            draft.reset()
            restores = target.restores + draft.restores
            r = speculative(draft, target, prompt, a.max_new, stop, k)
            same = sum(x == y for x, y in zip(r.tokens, base.tokens)) / max(len(base.tokens), 1)
            row["spec"][k] = {**r.to_dict(), "same_as_greedy": round(same, 3),
                              "restores": target.restores + draft.restores - restores,
                              "snapshot_mb": round(max(target.snapshot_bytes, draft.snapshot_bytes) / 2 ** 20, 2),
                              "speedup": round(r.tok_s / base.tok_s, 3) if base.tok_s else None}
            print(f"  k={k}: {r.tok_s:5.2f} tok/s, acceptance {r.acceptance:.2f}, "
                  f"{len(r.tokens) / r.rounds:.2f} tokens/round, speed-up ×{row['spec'][k]['speedup']}, "
                  f"identical to greedy {same:.0%}")
        rows.append(row)
    out = ROOT / "results" / "speculative" / f"{a.draft}--{a.target}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"draft": a.draft, "target": a.target, "rewind": [draft.rewindable, target.rewindable],
                               "rows": rows}, indent=1))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
