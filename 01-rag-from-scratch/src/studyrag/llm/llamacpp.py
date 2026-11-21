"""Local LLM backend: a quantized GGUF model run by llama.cpp on the CPU.

Why llama.cpp + GGUF? A 2B-parameter model in 16-bit floats needs ~4 GB of RAM; quantized to
~4.5 bits per weight (Q4_K_M) it needs ~1.3 GB and runs several times faster on a CPU, with only
a small quality loss. llama.cpp is a C++ inference engine heavily optimised for exactly this
(AVX2 SIMD kernels, memory-mapped weights, KV cache).

Two details matter on weak hardware:
1. Thinking models (Qwen3/3.5) reason in a hidden <think> block by default. At ~5 tokens/s that
   can add a minute before the answer starts, so for ChatML models we build the prompt ourselves
   and pre-fill an empty think block, which is exactly what the official template does for
   enable_thinking=False.
2. JSON output is enforced with a grammar generated from the JSON schema, so the model literally
   cannot produce invalid JSON (constrained decoding). Small models need this.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from .base import Message, parse_json_loose, strip_thinking

CHATML_STOP = ["<|im_end|>", "<|endoftext|>"]


def default_threads() -> int:
    """llama.cpp is usually fastest with one thread per *physical* core."""
    try:
        logical = len(os.sched_getaffinity(0))
    except AttributeError:
        logical = os.cpu_count() or 2
    return max(1, logical // 2) if logical >= 4 else logical


class LlamaCppLLM:
    def __init__(self, model_path: str | Path, n_ctx: int = 4096, n_threads: int = 0):
        from llama_cpp import Llama

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\nRun: python scripts/download_models.py")
        self.name = model_path.stem
        self.n_threads = n_threads or default_threads()
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=self.n_threads,
                          n_threads_batch=self.n_threads, n_batch=256, verbose=False)
        template = self._llm.metadata.get("tokenizer.chat_template", "")
        self._chatml = "<|im_start|>" in template
        self._can_think = "think" in template

    # -------------------------------------------------------------- prompt formatting
    def _chatml_prompt(self, messages: list[Message]) -> str:
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
        prompt = "".join(parts) + "<|im_start|>assistant\n"
        if self._can_think:
            prompt += "<think>\n\n</think>\n\n"      # thinking disabled
        return prompt

    def _complete(self, messages: list[Message], max_tokens: int, temperature: float,
                  stream: bool, grammar=None):
        common = dict(max_tokens=max_tokens, temperature=temperature, top_p=0.9,
                      repeat_penalty=1.05, stream=stream, grammar=grammar)
        if self._chatml:
            return self._llm.create_completion(self._chatml_prompt(messages), stop=CHATML_STOP, **common)
        return self._llm.create_chat_completion(messages=messages, **common)

    @staticmethod
    def _text_of(piece: dict) -> str:
        choice = piece["choices"][0]
        if "text" in choice:
            return choice["text"] or ""
        return (choice.get("delta") or choice.get("message") or {}).get("content") or ""

    # -------------------------------------------------------------- public API
    def generate(self, messages: list[Message], max_tokens: int = 512, temperature: float = 0.2,
                 stream: bool = False) -> str | Iterator[str]:
        out = self._complete(messages, max_tokens, temperature, stream)
        if stream:
            return (self._text_of(p) for p in out)
        return strip_thinking(self._text_of(out))

    def generate_json(self, messages: list[Message], schema: dict, max_tokens: int = 768,
                      temperature: float = 0.3) -> dict:
        from llama_cpp import LlamaGrammar

        grammar = LlamaGrammar.from_json_schema(json.dumps(schema), verbose=False)
        out = self._complete(messages, max_tokens, temperature, stream=False, grammar=grammar)
        return parse_json_loose(self._text_of(out))

    def count_tokens(self, text: str) -> int:
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
