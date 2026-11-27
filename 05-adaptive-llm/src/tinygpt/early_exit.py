"""Early exit: how much quality is lost, and how many layers are saved, at each confidence threshold?

    python -m tinygpt.early_exit --name exit-d256-L6            # a model trained with exits
    python -m tinygpt.early_exit --name dense-d256-L6           # baseline: no exit training ("logit lens")

For every position of fixed validation windows we compute the next-token scores after *every* layer (one pass,
teacher-forced). The exit rule for a threshold τ: use the first layer whose top probability is ≥ τ (the last layer
if none is). Reported per τ:

- **loss** (cross-entropy at the chosen exit) and **perplexity**, vs the full model's;
- **avg layers**: mean number of layers run per token (compute ≈ proportional, ignoring embeddings and the head);
- **agree**: how often the chosen exit's top token equals the full model's top token.

Caveat (stated in the docs): positions are evaluated teacher-forced, so a token's deeper layers still see the full
deep states of earlier tokens. In real generation with a KV cache, an exited token has no deep states; methods like
CALM copy its exit state upward. The savings here are what the exit *decision* allows, not a wall-clock speed-up.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .sample import load
from .train import TOKENS, fixed_windows

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]       # 1.01 = never exit early


def per_layer(model, windows: torch.Tensor, batch: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per layer and position: log-prob of the true next token, top probability, top token. Shapes (L, N)."""
    lp, conf, top = [], [], []
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch]
        x, y = w[:, :-1], w[:, 1:]
        outs = model.exit_logits(x)
        logp = torch.stack([F.log_softmax(o, dim=-1) for o in outs])               # (L, B, T, V)
        lp.append(logp.gather(-1, y[None, ..., None].expand(len(outs), -1, -1, 1))[..., 0].flatten(1))
        p_max, t_max = logp.exp().max(-1)
        conf.append(p_max.flatten(1))
        top.append(t_max.flatten(1))
    return torch.cat(lp, 1), torch.cat(conf, 1), torch.cat(top, 1)


def sweep(lp: torch.Tensor, conf: torch.Tensor, top: torch.Tensor, thresholds=THRESHOLDS) -> list[dict]:
    L, N = lp.shape
    rows = []
    for tau in thresholds:
        ok = conf >= tau
        ok[-1] = True                                           # the last layer always answers
        chosen = ok.float().argmax(0)                           # first layer that is confident enough
        idx = torch.arange(N)
        loss = -lp[chosen, idx].mean().item()
        rows.append({"threshold": tau, "loss": round(loss, 4), "ppl": round(math.exp(loss), 2),
                     "avg_layers": round((chosen.float() + 1).mean().item(), 3),
                     "compute_saved": round(1 - (chosen.float() + 1).mean().item() / L, 3),
                     "agree_with_full": round((top[chosen, idx] == top[-1]).float().mean().item(), 4),
                     "exit_share": [round(((chosen == l).float().mean().item()), 3) for l in range(L)]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load(a.name)
    valid = np.memmap(TOKENS / "valid.bin", dtype=np.uint16, mode="r")
    windows = fixed_windows(valid, model.config.ctx, a.windows)
    lp, conf, top = per_layer(model, windows)
    layers = [{"layer": l + 1, "loss": round(-lp[l].mean().item(), 4), "ppl": round(math.exp(-lp[l].mean().item()), 2),
               "agree_with_full": round((top[l] == top[-1]).float().mean().item(), 4),
               "mean_confidence": round(conf[l].mean().item(), 4)} for l in range(len(lp))]
    rows = sweep(lp, conf, top)
    trained = model.exit_norms is not None
    print(f"{a.name}: exits {'trained' if trained else 'NOT trained (logit lens)'}, {lp.shape[1]:,} positions")
    print(f"{'layer':>5} {'loss':>7} {'ppl':>7} {'agree':>6}")
    for r in layers:
        print(f"{r['layer']:>5} {r['loss']:>7.3f} {r['ppl']:>7.2f} {r['agree_with_full']:>6.3f}")
    print(f"\n{'τ':>5} {'loss':>7} {'ppl':>7} {'layers':>7} {'saved':>6} {'agree':>6}")
    for r in rows:
        print(f"{r['threshold']:>5} {r['loss']:>7.3f} {r['ppl']:>7.2f} {r['avg_layers']:>7.2f} "
              f"{r['compute_saved']:>6.1%} {r['agree_with_full']:>6.3f}")
    out = ROOT / "results" / "early_exit" / f"{a.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"name": a.name, "exits_trained": trained, "positions": lp.shape[1],
                               "per_layer": layers, "thresholds": rows}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
