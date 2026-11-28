"""The browser demo computes what PyTorch computes: exported int8 weights, the JavaScript forward pass (dense, MoE
top-1 and top-2, early exits) and the JavaScript tokenizer, checked against the Python ones. Needs Node; skipped
without it."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tinygpt.export_web import export, quantize_rows
from tinygpt.model import GPT, GPTConfig
from tinygpt.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def test_int8_rows_round_trip_within_half_a_step():
    w = torch.randn(5, 7) * torch.tensor([[0.01], [1], [10], [0], [3]])
    q, scale = quantize_rows(w)
    back = torch.from_numpy(q).float() * torch.from_numpy(scale)[:, None]
    assert (back - w).abs().max(dim=1).values.le(torch.from_numpy(scale) / 2 + 1e-6).all()   # ≤ half a step per row
    assert q[3].tolist() == [0] * 7                                                           # an all-zero row


CONFIGS = {
    "dense": dict(),
    "moe-top1": dict(n_experts=4, top_k=1, expert_mult=2),
    "moe-top2": dict(n_experts=4, top_k=2, expert_mult=1),
    "exit": dict(exit_weight=1.0),
}


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_matches_pytorch(tmp_path):
    cases = {"models": []}
    for name, extra in CONFIGS.items():
        torch.manual_seed(0)
        c = GPTConfig(vocab_size=96, ctx=24, d=32, n_layer=3, n_head=4, **extra)
        model = GPT(c).eval()
        with torch.no_grad():                                  # bigger, varied weights than the init, so that
            for p in model.parameters():                       # routing and exits actually differ between tokens
                p.add_(torch.randn_like(p) * 0.05)
        q = export(model, tmp_path / name)                     # the int8 model the browser must reproduce
        tokens = torch.randint(c.vocab_size, (20,)).tolist()
        with torch.no_grad():
            idx = torch.tensor([tokens])
            logits = q(idx)[0][0]
            case = {"name": name, "manifest": str(tmp_path / f"{name}.json"), "bin": str(tmp_path / f"{name}.bin"),
                    "tokens": tokens, "logits": logits.tolist(), "argmax": logits.argmax(-1).tolist(),
                    "exit_top_probs": None}
            if name == "exit":
                layers = q.exit_logits(idx)                        # (n_layer) × (1, T, V)
                case["exit_top_probs"] = [[torch.softmax(l[0, t], -1).max().item() for l in layers]
                                          for t in range(len(tokens))]
        cases["models"].append(case)

    text = ("Once upon a time, there was a little girl named Lily. She didn't like the rain! \"Can we play?\" "
            "she asked. They're 2 friends, 3 cats and 100 dots... ") * 20
    tok = BPETokenizer.train(text, vocab_size=330)
    tok.save(tmp_path / "tok.json")
    texts = ["Once upon a time, Lily's cat jumped.", "  two  spaces\tand\na newline  ", "Numbers 12345 and 007!",
             "Ünïcödé, 日本語 and emoji 🙂", "she'll we've I'm they'd it's", "<|endoftext|>Once again<|endoftext|>", ""]
    cases["tokenizer"] = {"path": str(tmp_path / "tok.json"), "texts": texts, "ids": [tok.encode(t) for t in texts]}
    (tmp_path / "cases.json").write_text(json.dumps(cases))

    res = subprocess.run([NODE, str(ROOT / "tests" / "web_check.mjs"), str(ROOT / "site" / "tinygpt.js"),
                          str(tmp_path / "cases.json")], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    for m in out["models"]:
        assert m["max_logit_diff"] < 1e-3, m                  # float32 sums in a different order: ~1e-6 expected
        assert m["argmax_same"] == 1.0, m
        assert m["max_exit_prob_diff"] < 1e-4, m
    assert all(t["same_ids"] for t in out["tokenizer"]), out["tokenizer"]
    assert all(t["round_trip"] for t in out["tokenizer"]), out["tokenizer"]
