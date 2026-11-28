"""Export trained models for the in-browser demo (`site/`): int8 weights + a JSON manifest, and the tokenizer.

    python -m tinygpt.export_web                         # the default demo models -> site/models/
    python -m tinygpt.export_web --names dense-d256-L6 --eval-windows 64

**Format.** One `<name>.bin` per model, little-endian, every tensor 4-byte aligned:
- 2-D weight matrices as **int8 with one float32 scale per row** (`w ≈ q × scale[row]`, `scale = max|row| / 127`).
  A row is one output unit of a linear layer (or one token's embedding), so each row keeps its own range;
- the MoE routers and every 1-D tensor (LayerNorm scales and shifts) as float32: they are tiny, and a router
  rounded to int8 could send tokens to different experts.
`<name>.json` lists each tensor's shape, type and byte offsets, plus the config and the measured cost of int8:
validation loss of the original and of the int8 weights on the same windows.

The browser code ([`site/tinygpt.js`](../../site/tinygpt.js)) reproduces the forward pass exactly;
`tests/test_web.py` checks it against PyTorch on these exported weights.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .model import GPT, GPTConfig
from .sample import ROOT, load
from .train import TOKENS, fixed_windows

SITE = ROOT / "site"
DEMO_MODELS = ["dense-d256-L6", "moe-8x-top2", "exit-d256-L6"]
FLOAT_TENSORS = ("router",)            # kept in float32 even if 2-D


def quantize_rows(w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 per row: q = round(w / scale), scale = max|row| / 127 (rows that are all zero get scale 1)."""
    w = w.detach().float()
    scale = w.abs().amax(dim=1) / 127.0
    scale[scale == 0] = 1.0
    q = torch.clamp(torch.round(w / scale[:, None]), -127, 127).to(torch.int8)
    return q.numpy(), scale.numpy().astype(np.float32)


def tensors_of(model: GPT) -> dict[str, torch.Tensor]:
    """State dict without the tied head (it *is* the token embedding)."""
    return {k: v for k, v in model.state_dict().items() if k != "head.weight"}


def export(model: GPT, path: Path, extra: dict | None = None) -> GPT:
    """Write `path`.bin + `path`.json; return a copy of the model with the int8-rounded weights (the reference the
    browser must reproduce)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob, entries, dequant = bytearray(), {}, {}

    def put(arr: np.ndarray) -> int:
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(np.ascontiguousarray(arr).tobytes())
        return off

    for name, t in tensors_of(model).items():
        if t.dim() == 2 and not any(f in name for f in FLOAT_TENSORS):
            q, scale = quantize_rows(t)
            entries[name] = {"shape": list(t.shape), "dtype": "int8", "offset": put(q), "scale_offset": put(scale)}
            dequant[name] = torch.from_numpy(q.astype(np.float32) * scale[:, None])
        else:
            arr = t.detach().float().numpy().astype("<f4")
            entries[name] = {"shape": list(t.shape), "dtype": "float32", "offset": put(arr)}
            dequant[name] = t.detach().float().clone()
    path.with_suffix(".bin").write_bytes(bytes(blob))
    manifest = {"config": model.config.to_dict(), "tensors": entries, "bytes": len(blob), **(extra or {})}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
    q_model = GPT(GPTConfig(**model.config.to_dict()))
    q_model.load_state_dict({**dequant, "head.weight": dequant["tok.weight"]})
    return q_model.eval()


@torch.no_grad()
def val_loss(model: GPT, windows: torch.Tensor, batch: int = 16) -> float:
    losses = [model(w[:, :-1], w[:, 1:])[1].item() * len(w) for w in windows.split(batch)]
    return sum(losses) / len(windows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=DEMO_MODELS)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    windows = fixed_windows(np.memmap(TOKENS / "valid.bin", dtype=np.uint16, mode="r"), 256, a.eval_windows)
    out = SITE / "models"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "models" / "tokenizer.json", SITE / "tokenizer.json")
    index = []
    for name in a.names:
        model = load(name)
        run = json.loads((ROOT / "models" / name / "run.json").read_text())
        full = val_loss(model, windows)
        q_model = export(model, out / name)
        quant = val_loss(q_model, windows)
        info = {"name": name, "params": model.n_params(), "active_params": model.n_active_params(),
                "val_loss_float32": round(full, 4), "val_loss_int8": round(quant, 4),
                "eval_windows": a.eval_windows, "final_val_loss": run.get("final_val_loss")}
        manifest = json.loads((out / f"{name}.json").read_text())
        (out / f"{name}.json").write_text(json.dumps({**manifest, **info}, indent=1))
        index.append({**info, "bytes": manifest["bytes"]})
        print(f"{name:16s} {manifest['bytes'] / 2**20:5.1f} MiB  val loss float32 {full:.4f} -> int8 {quant:.4f} "
              f"(+{quant - full:.4f})")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
