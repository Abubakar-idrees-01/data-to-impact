"""Pick a model, quantization, context size and thread count for this machine.

    python -m adaptive.plan                       # profile this machine, plan over the local GGUF models
    python -m adaptive.plan --models DIR --min-tok-s 5 --ctx 4096

Two rules, both from how CPU inference works:

1. **Memory:** a model needs its whole file in RAM (every token may use any weight), plus its attention cache
   (grows with context), plus working buffers. It must fit in *available* RAM minus a reserve, or the OS swaps and
   speed collapses by 10-100×.
2. **Speed:** generating one token reads every weight once, so on a CPU tokens/s ≈ efficiency × bandwidth / file
   size. Bigger or less-quantized models are slower in direct proportion. `efficiency` is calibrated against measured
   llama.cpp speeds (B2); until then it is 1.0 and the estimate is a ceiling.

Among the models that fit, take the largest (more parameters = better answers) that is still at least
`min_tok_s` fast; a reply slower than reading speed (~5 tokens/s) feels broken.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .gguf import kv_cache_bytes, model_info
from .hardware import GB, Hardware, profile, slug

ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODELS = ROOT.parent / "01-rag-from-scratch" / "models" / "llm"
CALIBRATION = ROOT / "results" / "hardware" / "calibration.json"
OVERHEAD_GB = 0.35            # llama.cpp compute buffers + the Python process; checked against measured RSS in B2
RESERVE_GB = 1.0              # left free for the OS and other apps
READING_SPEED = 5.0           # tokens/s: about 250 words per minute

# Approximate bits per weight of llama.cpp quantizations (base formats are exact: Q4_K stores 256 weights in 144
# bytes = 4.5 bits; the _M mixes keep some tensors at a higher type). Small models measure higher still, because
# their large embedding matrix stays at higher precision: our three Q4_K_M files are 5.0-5.7 bits.
QUANT_BITS = {"Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.7, "Q4_K_M": 4.9, "Q3_K_M": 3.9, "Q2_K": 3.0}
MIN_GOOD_QUANT = "Q4_K_M"     # below 4 bits quality drops fast; prefer a smaller model at Q4 to a bigger one at Q2


def efficiency() -> float:
    if CALIBRATION.exists():
        return json.loads(CALIBRATION.read_text()).get("efficiency", 1.0)
    return 1.0


def estimate(hw: Hardware, file_bytes: int, kv_bytes: int, eff: float | None = None) -> dict:
    eff = efficiency() if eff is None else eff
    need = (file_bytes + kv_bytes) / GB + OVERHEAD_GB
    budget = hw.ram_available_gb - RESERVE_GB
    tok_s = eff * (hw.bandwidth_gbs or 0) * 1e9 / file_bytes if hw.bandwidth_gbs else None
    return {"need_gb": round(need, 2), "fits": need <= budget, "tok_s": round(tok_s, 1) if tok_s else None}


def plan_local(hw: Hardware, models: list[dict], n_ctx: int = 4096, min_tok_s: float = READING_SPEED,
               eff: float | None = None) -> dict:
    """Rank the given GGUF models for this machine and choose one."""
    rows = []
    for m in models:
        kv = kv_cache_bytes(m, n_ctx)
        rows.append({"file": m["file"], "params_b": round(m["params"] / 1e9, 2), "quant": m["quant"],
                     "file_gb": round(m["bytes"] / GB, 2), "kv_gb": round(kv / GB, 3),
                     **estimate(hw, m["bytes"], kv, eff)})
    rows.sort(key=lambda r: r["params_b"])
    fitting = [r for r in rows if r["fits"]]
    fast = [r for r in fitting if r["tok_s"] is None or r["tok_s"] >= min_tok_s]
    if fast:
        choice, why = fast[-1], f"largest model that fits in RAM and runs at ≥ {min_tok_s} tok/s"
    elif fitting:
        choice, why = max(fitting, key=lambda r: r["tok_s"] or 0), f"nothing reaches {min_tok_s} tok/s; fastest that fits"
    else:
        choice, why = None, "no model fits in available RAM; use a smaller model or a lower quantization"
    return {"choice": choice and choice["file"], "why": why, "threads": hw.physical_cores, "n_ctx": n_ctx,
            "gpu_layers": gpu_layers(hw), "models": rows}


def gpu_layers(hw: Hardware) -> int:
    """llama.cpp n_gpu_layers: all layers on Apple Silicon (unified memory); 0 otherwise.

    NVIDIA offload (as many layers as fit in VRAM) is not implemented: no GPU here to test it on.
    """
    return -1 if any("Metal" in g for g in hw.gpus) else 0


def quant_grid(hw: Hardware, sizes_b=(0.5, 1, 2, 4, 8, 14), kv_gb_per_b: float = 0.03,
               eff: float | None = None) -> list[dict]:
    """For a model of N billion parameters at each quantization: memory needed, estimated speed, and the best pick.

    kv_gb_per_b: a rough attention cache of 0.03 GB per billion parameters at 4k context (Qwen2.5-1.5B: 0.11 GB).
    """
    out = []
    for size in sizes_b:
        cells = {}
        for q, bits in QUANT_BITS.items():
            file_bytes = int(size * 1e9 * bits / 8)
            cells[q] = estimate(hw, file_bytes, int(size * kv_gb_per_b * GB), eff)
        order = list(QUANT_BITS)
        good = [q for q in order[:order.index(MIN_GOOD_QUANT) + 1]
                if cells[q]["fits"] and (cells[q]["tok_s"] is None or cells[q]["tok_s"] >= READING_SPEED)]
        out.append({"params_b": size, "best": good[0] if good else None, "cells": cells})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(LOCAL_MODELS))
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--min-tok-s", type=float, default=READING_SPEED)
    ap.add_argument("--no-bandwidth", action="store_true", help="skip the 2-second bandwidth measurement")
    a = ap.parse_args()
    hw = profile(bandwidth=not a.no_bandwidth)
    print(f"{hw.cpu} | {hw.physical_cores} cores / {hw.logical_cores} threads | {', '.join(hw.flags) or 'no SIMD flags'}")
    print(f"RAM {hw.ram_available_gb:.1f} of {hw.ram_total_gb:.1f} GB available | GPU: {', '.join(hw.gpus) or 'none'} | "
          f"bandwidth {hw.bandwidth_gbs} GB/s | efficiency {efficiency()}")
    if "avx2" not in hw.flags and "neon" not in hw.flags:
        print("warning: no AVX2/NEON, llama.cpp will be several times slower")

    models = [model_info(p) for p in sorted(Path(a.models).glob("*.gguf"))]
    p = plan_local(hw, models, a.ctx, a.min_tok_s)
    print(f"\n{'model':<36} {'params':>6} {'quant':>7} {'file GB':>7} {'KV GB':>6} {'need':>5} {'fits':>4} {'tok/s':>6}")
    for r in p["models"]:
        print(f"{r['file']:<36} {r['params_b']:>5}B {r['quant']:>7} {r['file_gb']:>7} {r['kv_gb']:>6} "
              f"{r['need_gb']:>5} {'yes' if r['fits'] else 'NO':>4} {r['tok_s'] or '?':>6}")
    print(f"\nchoice: {p['choice']} ({p['why']}); threads={p['threads']}, n_ctx={p['n_ctx']}, n_gpu_layers={p['gpu_layers']}")

    grid = quant_grid(hw)
    print(f"\nIf downloading: GB needed / est. tok/s per size and quantization ('*' = best pick, '-' = doesn't fit)")
    print(f"{'size':>5} " + " ".join(f"{q:>12}" for q in QUANT_BITS))
    for row in grid:
        cells = [(f"{c['need_gb']}/{c['tok_s']}" if c["fits"] else "-") + ("*" if q == row["best"] else " ")
                 for q, c in row["cells"].items()]
        print(f"{row['params_b']:>4}B " + " ".join(f"{c:>12}" for c in cells))

    out = ROOT / "results" / "hardware" / f"{slug(hw.cpu)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hardware": hw.to_dict(), "plan": p, "grid": grid}, indent=1))
    print("saved", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
