"""Samples and validation loss of `moe-4x-top1-bug` under the gate it was trained with.

That run renormalised its single top-1 weight (gate = p / p = 1). The fixed code multiplies by the raw p instead, so
loading the old checkpoint with today's code runs a model it was never trained as. This script restores the old gate
for this one checkpoint and shows both: validation loss under the old and the new gate, and stories under the old one.

    PYTHONPATH=src .venv/bin/python experiments/sample_bug_run.py > results/samples/moe-4x-top1-bug-final.txt
"""

import numpy as np
import torch
import torch.nn.functional as F

from tinygpt import model as M
from tinygpt.sample import ROOT, load
from tinygpt.tokenizer import BPETokenizer
from tinygpt.train import TOKENS, fixed_windows

fixed_forward = M.MoE.forward


def old_forward(self, x):
    """The MoE forward pass as the bug run was trained: top-k weights always renormalised, also for k = 1."""
    B, T, D = x.shape
    flat = x.reshape(-1, D)
    probs = F.softmax(self.router(flat), dim=-1)
    weights, chosen = torch.topk(probs, self.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    out = torch.zeros_like(flat)
    for e, expert in enumerate(self.experts):
        rows, slot = torch.where(chosen == e)
        if rows.numel():
            out.index_add_(0, rows, expert(flat[rows]) * weights[rows, slot, None])
    return out.view(B, T, D)


@torch.no_grad()
def val_loss(model, windows) -> float:
    losses = [model(w[None, :-1], w[None, 1:])[1].item() for w in windows]
    return sum(losses) / len(losses)


def main() -> None:
    torch.set_num_threads(2)
    model = load("moe-4x-top1-bug")
    windows = fixed_windows(np.memmap(TOKENS / "valid.bin", dtype=np.uint16, mode="r"), 256, 64)
    M.MoE.forward = fixed_forward
    new = val_loss(model, windows)
    M.MoE.forward = old_forward
    old = val_loss(model, windows)
    print(f"# moe-4x-top1-bug on 64 validation windows: loss {old:.3f} with the gate it was trained with (p / p = 1), "
          f"{new:.3f} with today's fixed gate (raw p)\n# samples below use the trained gate; prompt 'Once upon a time', "
          f"temperature 0.8, top-k 40, seed 0\n")
    tok = BPETokenizer.load(ROOT / "models" / "tokenizer.json")
    gen = torch.Generator().manual_seed(0)
    for i in range(3):
        idx = torch.tensor([tok.encode("Once upon a time")])
        out = model.generate(idx, 200, 0.8, 40, stop=tok.eot, generator=gen)
        print(f"--- sample {i + 1}\n{tok.decode([t for t in out[0].tolist() if t != tok.eot])}\n")


if __name__ == "__main__":
    main()
