"""Run GGUF models with llama.cpp, stream their output, and time every phase.

A reply has two phases with very different costs:
- **prefill:** the whole prompt goes through the model in batches. Many tokens share each read of the weights, so
  it is compute-bound and fast per token. Its length decides the **time to first token** (TTFT).
- **decode:** one token at a time, each needing a full read of the weights: memory-bound, slower per token.

`LocalModel.generate()` reports both, so the router and the benchmarks can reason about them separately.

**KV-cache reuse:** llama.cpp keeps the keys/values of the previous prompt; when the next prompt starts with the same
tokens (the same chat history plus a new message) only the new part is prefilled. `reuse=False` clears the cache
first, to measure what that saves. Reuse needs llama.cpp to delete the cache *after* the shared prefix; for models
with a recurrent state (Qwen3.5's linear-attention layers) that state cannot be rewound, so llama.cpp may fall back
to prefilling everything. `cached_prefix` is therefore an upper bound; the benchmark measures the real effect on TTFT.

**Weights are read into memory, not memory-mapped** (`use_mmap=False`). On AVX2/NEON, llama.cpp repacks most Q4_K
weights into an interleaved layout, a second copy; with mmap the original file pages also stay resident, so every
repacked weight is in RAM twice. Measured peak RSS: Qwen3.5-2B 1.86 → 1.39 GiB, Qwen2.5-1.5B 1.82 → 1.24 GiB, with
the same load time (2.1-2.3 s).

Thinking models (Qwen3/3.5) would first write a hidden `<think>` block, which at ~5 tok/s adds up to a minute; the
prompt pre-fills an empty one, as the official template does with `enable_thinking=False` (as in project 1).
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator

from .gguf import kv_cache_bytes, model_info
from .hardware import GB

USE_MMAP = False             # see the module docstring: mmap + weight repacking keeps two copies of most weights

CHATML_STOP = ["<|im_end|>", "<|endoftext|>"]
Message = dict          # {"role": "system" | "user" | "assistant", "content": str}


@dataclass
class Generation:
    text: str
    model: str
    prompt_tokens: int
    cached_prefix: int          # leading prompt tokens already in the KV cache (reusable; see the module notes)
    gen_tokens: int
    ttft_s: float               # time to first token = prefill + the first decode step
    decode_tok_s: float         # tokens/s after the first token
    total_s: float
    token_probs: list[float] | None = None   # probability of each generated token (confidence=True)

    @property
    def confidence(self) -> float | None:
        """Geometric mean probability of the generated tokens = exp(mean log-prob): 1.0 = certain of every token."""
        if not self.token_probs:
            return None
        return math.exp(sum(math.log(max(p, 1e-12)) for p in self.token_probs) / len(self.token_probs))

    def to_dict(self) -> dict:
        return asdict(self)


EMPTY_THINK = "<think>\n\n</think>\n\n"


def chatml(messages: list[Message], think: bool = False) -> str:
    """ChatML prompt. With think=False every assistant turn, past and new, starts with an empty think block.

    Past replies keep that block so the history is token-for-token what the model generated: the KV cache then
    matches the conversation so far and only the new user message is prefilled. Smoke test, second turn of a chat:
    16 of 41 prompt tokens matched the cache without it; 26 of 45 with it, i.e. everything up to the end-of-turn
    marker (the stop token is never fed back into the cache).
    """
    prefix = "" if think else EMPTY_THINK
    prompt = "".join(f"<|im_start|>{m['role']}\n{prefix if m['role'] == 'assistant' else ''}{m['content']}<|im_end|>\n"
                     for m in messages)
    return prompt + "<|im_start|>assistant\n" + prefix


class ProbRecorder:
    """A llama.cpp logits processor that records how probable each sampled token was, without changing anything.

    At every step it gets the scores for the whole vocabulary and keeps their softmax; the token chosen at step i is
    only known at step i+1 (it is then the last input id), so each probability is looked up one step late. The final
    step's token never comes back as input; for it the top probability is used (= the chosen token under greedy
    decoding). Cost: one softmax over the vocabulary per token, ~2 ms against ~70 ms per decode step here.
    """

    def __init__(self):
        self.probs: list[float] = []
        self._last = None
        self._n = None

    def __call__(self, input_ids, logits):
        import numpy as np
        if self._last is not None and len(input_ids) > self._n:
            self.probs.append(float(self._last[input_ids[-1]]))
        x = logits - logits.max()
        e = np.exp(x, dtype=np.float32)
        self._last, self._n = e / e.sum(), len(input_ids)
        return logits

    def finish(self) -> list[float]:
        if self._last is not None:
            self.probs.append(float(self._last.max()))
            self._last = None
        return self.probs


def timing(t0: float, t_first: float | None, t_end: float, gen_tokens: int) -> tuple[float, float]:
    """(time to first token, decode tokens/s). Decode speed excludes the first token, which waited for prefill."""
    if t_first is None:                                 # nothing was generated
        return t_end - t0, 0.0
    rest = t_end - t_first
    return t_first - t0, ((gen_tokens - 1) / rest) if gen_tokens > 1 and rest > 0 else 0.0


class LocalModel:
    def __init__(self, path: str | Path, n_ctx: int = 4096, threads: int = 2, llama_factory: Callable | None = None):
        self.path = Path(path)
        self.info = model_info(self.path)
        self.name = self.info["name"]
        self.n_ctx, self.threads = n_ctx, threads
        if llama_factory is None:
            from llama_cpp import Llama
            llama_factory = Llama
        self.llm = llama_factory(model_path=str(self.path), n_ctx=n_ctx, n_threads=threads, n_threads_batch=threads,
                                 n_batch=256, use_mmap=USE_MMAP, verbose=False)
        template = self.llm.metadata.get("tokenizer.chat_template", "")
        if "<|im_start|>" not in template:
            raise ValueError(f"{self.path.name}: only ChatML chat templates (Qwen, many others) are supported")
        self.can_think = "think" in template

    def prompt(self, messages: list[Message]) -> str:
        return chatml(messages, think=not self.can_think)    # thinking models get an empty think block

    def memory_gb(self) -> float:
        """What this model holds in RAM: weights + a full KV cache (buffers excluded)."""
        return (self.info["bytes"] + kv_cache_bytes(self.info, self.n_ctx)) / GB

    def tokens(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def cached_prefix(self, prompt_tokens: list[int]) -> int:
        """How many leading prompt tokens are already in llama.cpp's KV cache."""
        cached = list(self.llm._input_ids) if self.llm.n_tokens else []
        n = 0
        for a, b in zip(cached, prompt_tokens):
            if a != b:
                break
            n += 1
        return min(n, len(prompt_tokens) - 1)          # llama.cpp always re-evaluates at least the last token

    def stream(self, messages: list[Message], max_tokens: int = 256, temperature: float = 0.2,
               logits_processor=None) -> Iterator[str]:
        for piece in self.llm.create_completion(self.prompt(messages), max_tokens=max_tokens, temperature=temperature,
                                                top_p=0.9, repeat_penalty=1.05, stop=CHATML_STOP, stream=True,
                                                logits_processor=logits_processor):
            yield piece["choices"][0].get("text") or ""

    def generate(self, messages: list[Message], max_tokens: int = 256, temperature: float = 0.2,
                 reuse: bool = True, on_text: Callable[[str], None] | None = None,
                 confidence: bool = False) -> Generation:
        if not reuse:
            self.llm.reset()
        prompt_tokens = self.tokens(self.prompt(messages))
        cached = self.cached_prefix(prompt_tokens)
        t0, t_first, parts = time.perf_counter(), None, []
        rec = ProbRecorder() if confidence else None
        for text in self.stream(messages, max_tokens, temperature, [rec] if rec else None):
            if t_first is None:
                t_first = time.perf_counter()
            parts.append(text)
            if on_text:
                on_text(text)
        t_end = time.perf_counter()
        out = "".join(parts)
        n_gen = len(self.tokens(out)) if out else 0
        ttft, rate = timing(t0, t_first, t_end, n_gen)
        return Generation(text=out, model=self.name, prompt_tokens=len(prompt_tokens), cached_prefix=cached,
                          gen_tokens=n_gen, ttft_s=round(ttft, 3), decode_tok_s=round(rate, 2),
                          total_s=round(t_end - t0, 3), token_probs=rec.finish() if rec else None)


class Ladder:
    """The local models from smallest to largest, loaded on first use, within a RAM budget.

    Loading a model costs seconds (the file is memory-mapped, then pages are read from disk as they are touched), so
    loaded models are kept; when the next one would not fit, the least recently used one is unloaded.
    """

    def __init__(self, paths: list[str | Path], budget_gb: float, n_ctx: int = 4096, threads: int = 2,
                 llama_factory: Callable | None = None):
        infos = sorted((model_info(p) for p in paths), key=lambda i: i["params"])
        self.paths = [Path(i["path"]) for i in infos]
        self.names = [i["name"] for i in infos]
        self.budget_gb, self.n_ctx, self.threads, self.factory = budget_gb, n_ctx, threads, llama_factory
        self.loaded: OrderedDict[str, LocalModel] = OrderedDict()
        self.load_seconds: dict[str, float] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, rung: int) -> LocalModel:
        name = self.names[rung]
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]
        need = (model_info(self.paths[rung])["bytes"] + kv_cache_bytes(model_info(self.paths[rung]), self.n_ctx)) / GB
        while self.loaded and sum(m.memory_gb() for m in self.loaded.values()) + need > self.budget_gb:
            _, old = self.loaded.popitem(last=False)
            old.llm.close() if hasattr(old.llm, "close") else None
        t = time.perf_counter()
        model = LocalModel(self.paths[rung], self.n_ctx, self.threads, self.factory)
        self.load_seconds[name] = round(time.perf_counter() - t, 2)
        self.loaded[name] = model
        return model
