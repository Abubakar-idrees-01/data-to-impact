"""Train a tiny GPT on the tokenized stories, on a CPU, in a way that survives interruptions.

    python -m tinygpt.train --name dense-6L --steps 3000                 # train (resumes if models/dense-6L/ exists)
    python -m tinygpt.train --name dense-6L --steps 3000 --max-minutes 60   # stop after an hour, resume later

Each step: sample `batch` random windows of `ctx + 1` tokens from train.bin; the model predicts token t+1 from
tokens ≤ t at every position (so one window gives ctx training examples); cross-entropy loss; AdamW update.

- **Learning rate:** linear warm-up, then cosine decay to 10% (standard for transformers).
- **Validation:** every `eval_every` steps, mean loss on the same fixed set of validation windows, so the curve is
  comparable across runs. Perplexity = e^loss: "as unsure as choosing among N tokens".
- **Checkpoints:** model + optimizer + step + RNG state, written atomically every eval; rerunning the command resumes.
- **Log:** models/<name>/log.jsonl, one line per eval (step, train/val loss, tokens/s, elapsed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import GPT, MoE, GPTConfig

ROOT = Path(__file__).resolve().parents[2]
TOKENS = ROOT / "data" / "tokens"
MODELS = ROOT / "models"


def batches(data: np.ndarray, ctx: int, batch: int, gen: torch.Generator):
    while True:
        ix = torch.randint(len(data) - ctx - 1, (batch,), generator=gen).tolist()
        chunk = torch.from_numpy(np.stack([data[i:i + ctx + 1].astype(np.int64) for i in ix]))
        yield chunk[:, :-1], chunk[:, 1:]


def fixed_windows(data: np.ndarray, ctx: int, n: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(data) - ctx - 1, (n,), generator=g).tolist()
    return torch.from_numpy(np.stack([data[i:i + ctx + 1].astype(np.int64) for i in ix]))


@torch.no_grad()
def evaluate(model: GPT, windows: torch.Tensor, batch: int) -> float:
    model.eval()
    losses = []
    for i in range(0, len(windows), batch):
        w = windows[i:i + batch]
        _, loss = model(w[:, :-1], w[:, 1:])
        losses.append(loss.item() * len(w))
    model.train()
    return sum(losses) / len(windows)


def lr_at(step: int, steps: int, lr: float, warmup: int) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def save_atomic(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=256)
    ap.add_argument("--threads", type=int, default=os.cpu_count() // 2 or 1, help="default: physical cores")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop (and checkpoint) after this long")
    ap.add_argument("--seed", type=int, default=0)
    for f in GPTConfig.__dataclass_fields__.values():      # model shape: --d 256 --n-layer 6 ...
        if f.name != "vocab_size":
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--data", default=str(TOKENS))
    ap.add_argument("--out", default=str(MODELS))
    a = ap.parse_args(argv)

    torch.set_num_threads(a.threads)
    data_dir, out = Path(a.data), Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data_dir / "meta.json").read_text())
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    valid = np.memmap(data_dir / "valid.bin", dtype=np.uint16, mode="r")

    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        print(f"resuming {a.name} at step {ckpt['step']}")
    else:
        ckpt = None
        config = GPTConfig(vocab_size=meta["vocab_size"],
                           **{k: getattr(a, k) for k in GPTConfig.__dataclass_fields__ if k != "vocab_size"})
    torch.manual_seed(a.seed)
    model = GPT(config)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]       # no weight decay on norms/biases
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(a.seed)
    step, elapsed = 0, 0.0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        gen.set_state(ckpt["gen"])
        step, elapsed = ckpt["step"], ckpt["elapsed"]
    run_info = {"name": a.name, "config": config.to_dict(), "params": model.n_params(),
                "active_params": model.n_active_params(),
                "params_no_embeddings": model.n_params(embeddings=False), "batch": a.batch, "lr": a.lr,
                "steps": a.steps, "threads": a.threads, "tokens_per_step": a.batch * config.ctx}
    (out / "run.json").write_text(json.dumps(run_info, indent=1))
    print(f"{a.name}: {model.n_params() / 1e6:.2f}M parameters, {model.n_active_params() / 1e6:.2f}M active per token "
          f"({model.n_params(embeddings=False) / 1e6:.2f}M without embeddings), {a.threads} threads", flush=True)

    vwin = fixed_windows(valid, config.ctx, a.eval_windows)
    it = batches(train, config.ctx, a.batch, gen)
    log = (out / "log.jsonl").open("a")
    t_start, t_window, tok_window, train_losses, aux_values = time.time(), time.time(), 0, [], []
    stopped = "steps"
    model.train()
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        x, y = next(it)
        _, loss = model(x, y)
        aux = model.aux_loss()
        opt.zero_grad(set_to_none=True)
        (loss + config.aux_coef * aux).backward()        # aux = MoE load balancing (0 for dense)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tok_window += x.numel()
        train_losses.append(loss.item())
        aux_values.append(float(aux.detach()))
        if step % a.eval_every == 0 or step == a.steps:
            now = time.time()
            elapsed += now - t_start
            t_start = now
            val = evaluate(model, vwin, a.batch)
            rec = {"step": step, "tokens": step * a.batch * config.ctx, "train_loss": round(float(np.mean(train_losses)), 4),
                   "val_loss": round(val, 4), "val_ppl": round(math.exp(val), 2),
                   "tok_per_s": round(tok_window / (now - t_window)), "lr": round(opt.param_groups[0]["lr"], 6),
                   "elapsed_min": round(elapsed / 60, 1)}
            moe = [b.ffn for b in model.blocks if isinstance(b.ffn, MoE)]
            if moe:                                         # share of tokens per expert, per layer (last validation batch)
                rec["aux"] = round(float(np.mean(aux_values)), 4)
                rec["expert_share"] = [[round(float(c), 3) for c in f.last_counts / f.last_counts.sum()] for f in moe]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "gen": gen.get_state(), "step": step,
                         "elapsed": elapsed, "config": config.to_dict()}, ckpt_path)
            t_window, tok_window, train_losses, aux_values = time.time(), 0, [], []
            if a.max_minutes and elapsed / 60 >= a.max_minutes and step < a.steps:
                stopped = "time"
                break
    torch.save({"model": model.state_dict(), "config": config.to_dict(), "step": step}, out / "model.pt")
    print(f"done: step {step}/{a.steps} ({stopped}), {elapsed / 60:.1f} min")
    return {"step": step, "stopped": stopped}


if __name__ == "__main__":
    main()
