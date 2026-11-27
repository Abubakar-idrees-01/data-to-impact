"""B2 runtime logic, with a fake llama.cpp so no model is needed: prompts, timing, cache-prefix maths, the ladder."""

import numpy as np
import pytest

from adaptive.runtime import EMPTY_THINK, Ladder, LocalModel, chatml, timing
from helpers import write_gguf


class FakeLlama:
    """Mimics the parts of llama_cpp.Llama we use. Tokens = characters; streams the reply one character at a time."""
    loaded = 0

    def __init__(self, model_path, n_ctx, **kw):
        FakeLlama.loaded += 1
        self.metadata = {"tokenizer.chat_template": "<|im_start|>{{ think }}"}
        self.input_ids, self.n_tokens, self.reply, self.closed = np.zeros(n_ctx, np.intc), 0, "Hi there", False

    @property
    def _input_ids(self):
        return self.input_ids[: self.n_tokens]

    def tokenize(self, b, add_bos=False, special=True):
        return list(b)

    def reset(self):
        self.n_tokens = 0

    def close(self):
        self.closed = True

    def create_completion(self, prompt, stream=True, **kw):
        toks = self.tokenize(prompt.encode()) + self.tokenize(self.reply.encode())
        self.input_ids[: len(toks)] = toks                 # like llama.cpp: prompt + generated tokens stay cached
        self.n_tokens = len(toks)
        for ch in self.reply:
            yield {"choices": [{"text": ch}]}


def model_file(tmp_path, name, n_params):
    p = tmp_path / f"{name}.gguf"
    write_gguf(p, {"general.architecture": "qwen2", "general.name": name, "qwen2.block_count": 2,
                   "qwen2.embedding_length": 64, "qwen2.attention.head_count": 4,
                   "qwen2.attention.head_count_kv": 2}, [("w", [n_params])])
    return p


def test_chatml_keeps_the_empty_think_block_in_history():
    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}]
    p = chatml(msgs)
    assert p.count(EMPTY_THINK) == 2 and p.endswith("<|im_start|>assistant\n" + EMPTY_THINK)
    assert f"<|im_start|>assistant\n{EMPTY_THINK}b<|im_end|>" in p
    assert EMPTY_THINK not in chatml(msgs, think=True)


def test_timing_excludes_the_first_token_from_decode_speed():
    assert timing(0.0, 2.0, 4.0, 11) == (2.0, 5.0)        # 10 tokens after the first, in 2 s
    assert timing(0.0, None, 1.0, 0) == (1.0, 0.0)
    assert timing(0.0, 1.0, 1.0, 1) == (1.0, 0.0)


def test_second_turn_reuses_the_whole_history(tmp_path):
    m = LocalModel(model_file(tmp_path, "m", 1000), n_ctx=4096, threads=2, llama_factory=FakeLlama)
    chat = [{"role": "user", "content": "hello"}]
    g1 = m.generate(chat, temperature=0)
    assert (g1.text, g1.cached_prefix, g1.gen_tokens) == ("Hi there", 0, 8)
    chat += [{"role": "assistant", "content": g1.text}, {"role": "user", "content": "more"}]
    g2 = m.generate(chat, temperature=0)
    first_turn = len(m.prompt(chat[:1])) + len("Hi there")     # everything the cache holds
    assert g2.cached_prefix == first_turn
    assert m.generate(chat, reuse=False).cached_prefix == 0


def test_non_chatml_models_are_refused(tmp_path):
    class Plain(FakeLlama):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.metadata = {"tokenizer.chat_template": "[INST] {{ x }} [/INST]"}
    with pytest.raises(ValueError):
        LocalModel(model_file(tmp_path, "m", 10), llama_factory=Plain)


def test_ladder_orders_by_size_and_evicts_least_recently_used(tmp_path):
    paths = [model_file(tmp_path, "big", 3 * 2 ** 28), model_file(tmp_path, "small", 2 ** 28),
             model_file(tmp_path, "mid", 2 * 2 ** 28)]
    ladder = Ladder(paths, budget_gb=1.0, n_ctx=16, llama_factory=FakeLlama)
    assert ladder.names == ["small", "mid", "big"]
    # make every model "weigh" 0.4 GB (sparse files: no disk used) so that only two fit in the 1 GB budget
    for p in paths:
        with open(p, "ab") as f:
            f.truncate(int(0.4 * 2 ** 30))
    a, b = ladder.get(0), ladder.get(1)
    assert list(ladder.loaded) == ["small", "mid"]
    ladder.get(0)                                                      # small is now the most recent
    ladder.get(2)                                                      # big needs room: mid goes, not small
    assert list(ladder.loaded) == ["small", "big"] and b.llm.closed and not a.llm.closed
