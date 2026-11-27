"""How fast does this CPU train each model size? Measure before choosing one.

    python -m tinygpt.bench            # prints a table and writes results/train_throughput.json

For each configuration: a few warm-up steps, then timed full training steps (forward, backward, AdamW) on random
tokens; reports training tokens per second and how long a run of a given number of tokens would take.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]

CANDIDATES = {
    "d128-L4": dict(d=128, n_layer=4, n_head=4),
    "d192-L6": dict(d=192, n_layer=6, n_head=6),
    "d256-L6": dict(d=256, n_layer=6, n_head=8),
    "d256-L8": dict(d=256, n_layer=8, n_head=8),
    "d384-L6": dict(d=384, n_layer=6, n_head=6),
}


def measure(cfg: GPTConfig, batch: int, threads: int, steps: int = 6, warmup: int = 2) -> float:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(cfg.vocab_size, (batch, cfg.ctx))
    for i in range(warmup + steps):
        if i == warmup:
            t0 = time.perf_counter()
        _, loss = m(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return batch * cfg.ctx * steps / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", default="2,4")
    ap.add_argument("--only", default="", help="comma-separated candidate names")
    ap.add_argument("--budget-tokens", type=float, default=20e6, help="for the 'hours for N tokens' column")
    a = ap.parse_args()
    rows = []
    names = [n for n in CANDIDATES if not a.only or n in a.only.split(",")]
    print(f"{'config':<10} {'params':>8} {'threads':>7} {'tok/s':>7} {'hours for ' + f'{a.budget_tokens / 1e6:.0f}M tok':>16}")
    for name in names:
        cfg = GPTConfig(ctx=a.ctx, **CANDIDATES[name])
        params = GPT(cfg).n_params()
        for th in map(int, a.threads.split(",")):
            tps = measure(cfg, a.batch, th)
            rows.append({"config": name, **CANDIDATES[name], "params": params, "threads": th, "tok_per_s": round(tps),
                         "hours_for_budget": round(a.budget_tokens / tps / 3600, 2)})
            print(f"{name:<10} {params / 1e6:>7.2f}M {th:>7} {tps:>7.0f} {a.budget_tokens / tps / 3600:>16.2f}", flush=True)
    out = ROOT / "results" / "train_throughput.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"cpu": platform.processor() or platform.machine(), "torch": torch.__version__,
                               "ctx": a.ctx, "batch": a.batch, "budget_tokens": a.budget_tokens, "rows": rows}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
