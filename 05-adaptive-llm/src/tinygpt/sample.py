"""Write stories with a trained model.

    python -m tinygpt.sample --name dense-d256-L6 --prompt "Once upon a time" --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import GPT, GPTConfig
from .tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, models: Path = ROOT / "models") -> GPT:
    path = models / name / "model.pt"
    if not path.exists():
        path = models / name / "ckpt.pt"                 # a run still in progress: use its last checkpoint
    ckpt = torch.load(path, weights_only=False)
    m = GPT(GPTConfig(**ckpt["config"]))
    m.load_state_dict(ckpt["model"])
    return m.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    tok = BPETokenizer.load(ROOT / "models" / "tokenizer.json")
    model = load(a.name)
    gen = torch.Generator().manual_seed(a.seed)
    for i in range(a.n):
        idx = torch.tensor([tok.encode(a.prompt)])
        out = model.generate(idx, a.max_new, a.temperature, a.top_k, stop=tok.eot, generator=gen)
        text = tok.decode([t for t in out[0].tolist() if t != tok.eot])
        print(f"--- sample {i + 1}\n{text}\n")


if __name__ == "__main__":
    main()
