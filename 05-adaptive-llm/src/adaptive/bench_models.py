"""Measure every local GGUF model on this machine, and calibrate the planner's speed estimate.

    python -m adaptive.bench_models                     # all models in the default folder (~10 min on the i5)
    python -m adaptive.bench_models --threads 1 2 4 --models DIR

Each model runs in its own subprocess, so its peak memory (RSS) is its own. Per model:
- **load** time;
- **prefill** and **decode** speed on a fixed ~300-token prompt (temperature 0, 96 new tokens), for each thread count;
- **KV-cache reuse:** the second turn of a chat, timed with the cache kept and with it cleared;
- **peak RSS**, compared with the planner's memory estimate (file + KV cache + buffers).

Then, per architecture, **efficiency = measured decode tok/s ÷ (bandwidth / file size)**: the share of the memory
bandwidth ceiling llama.cpp reaches. Saved to results/hardware/calibration.json, where the planner picks it up.
Close other programs first: a busy CPU makes every number worse.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

from .hardware import GB, profile, slug

ROOT = Path(__file__).resolve().parents[2]

PROMPT = ("Read the text and answer the question.\n\nText: " + " ".join([
    "The river town grew around a stone bridge built in the twelfth century.",
    "Merchants crossed it with wool and salt, and a market formed on the eastern bank.",
    "In 1847 a flood destroyed two of the five arches; the town rebuilt them in brick, which is why the bridge",
    "is now half grey and half red. A railway arrived in 1871 and the market moved to the station square.",
    "Today the old market hall is a library, and the bridge is closed to cars but open to walkers and cyclists.",
]) * 3 + "\n\nQuestion: Why is the bridge two colours, and what happened to the market? Answer in a short paragraph.")
FOLLOW_UP = "Now summarise the whole history of the town in exactly three bullet points."


def rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / GB if platform.system() == "Darwin" else peak * 1024 / GB      # macOS: bytes; Linux: KB


def one_model(path: str, threads: list[int], n_ctx: int, max_tokens: int) -> dict:
    """Runs inside the subprocess."""
    from .runtime import LocalModel
    user = [{"role": "user", "content": PROMPT}]
    runs, load_s = [], None
    for th in threads:                                   # a fresh model per thread count (the file stays in page cache)
        t = time.perf_counter()
        m = LocalModel(path, n_ctx=n_ctx, threads=th)
        load_s = load_s or time.perf_counter() - t       # the first load, from disk (or the OS file cache)
        m.generate(user, max_tokens=8, temperature=0)    # warm-up: touch every weight page once
        g = m.generate(user, max_tokens=max_tokens, temperature=0, reuse=False)
        runs.append({"threads": th, **g.to_dict(), "prefill_tok_s": round(g.prompt_tokens / g.ttft_s, 1)})
        m.llm.close()
    th = max(runs, key=lambda r: r["decode_tok_s"])["threads"]
    m = LocalModel(path, n_ctx=n_ctx, threads=th)
    first = m.generate(user, max_tokens=max_tokens, temperature=0, reuse=False)
    chat = user + [{"role": "assistant", "content": first.text}, {"role": "user", "content": FOLLOW_UP}]
    kept = m.generate(chat, max_tokens=48, temperature=0, reuse=True)
    cleared = m.generate(chat, max_tokens=48, temperature=0, reuse=False)
    return {"file": Path(path).name, "name": m.name, "arch": m.info["arch"], "bytes": m.info["bytes"],
            "load_s": round(load_s, 2), "runs": runs, "best_threads": th,
            "reuse": {"prompt_tokens": kept.prompt_tokens, "cached_prefix": kept.cached_prefix,
                      "ttft_kept_s": kept.ttft_s, "ttft_cleared_s": cleared.ttft_s},
            "sample": first.text[:300], "peak_rss_gb": round(rss_gb(), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT.parent / "01-rag-from-scratch" / "models" / "llm"))
    ap.add_argument("--threads", type=int, nargs="+", default=None, help="default: physical cores, then 1 and logical")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)       # internal: run one model and print JSON
    a = ap.parse_args()
    if a.worker:
        print("RESULT " + json.dumps(one_model(a.worker, a.threads, a.ctx, a.max_tokens)))
        return

    hw = profile()
    threads = a.threads or sorted({hw.physical_cores, 1, hw.logical_cores})
    print(f"{hw.cpu}: {hw.physical_cores} cores / {hw.logical_cores} threads, read bandwidth {hw.bandwidth_gbs} GB/s")
    from .gguf import kv_cache_bytes, model_info
    from .plan import OVERHEAD_GB
    results = []
    for path in sorted(Path(a.models).glob("*.gguf")):
        print(f"\n{path.name} ...", flush=True)
        out = subprocess.run([sys.executable, "-m", "adaptive.bench_models", "--worker", str(path), "--ctx", str(a.ctx),
                              "--max-tokens", str(a.max_tokens), "--threads", *map(str, threads)],
                             capture_output=True, text=True, cwd=ROOT,
                             env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        line = next((l for l in out.stdout.splitlines() if l.startswith("RESULT ")), None)
        if not line:
            print("  failed:", out.stderr[-800:])
            continue
        r = json.loads(line[7:])
        info = model_info(path)
        r["estimated_need_gb"] = round((info["bytes"] + kv_cache_bytes(info, a.ctx)) / GB + OVERHEAD_GB, 3)
        best = next(x for x in r["runs"] if x["threads"] == r["best_threads"])
        r["ceiling_tok_s"] = round(hw.bandwidth_gbs * 1e9 / r["bytes"], 1)
        r["efficiency"] = round(best["decode_tok_s"] / r["ceiling_tok_s"], 3)
        results.append(r)
        for x in r["runs"]:
            print(f"  {x['threads']} threads: prefill {x['prefill_tok_s']:>6} tok/s (TTFT {x['ttft_s']} s for "
                  f"{x['prompt_tokens']} tokens), decode {x['decode_tok_s']:>5} tok/s")
        print(f"  load {r['load_s']} s | peak RSS {r['peak_rss_gb']} GB (estimate {r['estimated_need_gb']}) | "
              f"ceiling {r['ceiling_tok_s']} tok/s -> efficiency {r['efficiency']}")
        print(f"  2nd turn TTFT: {r['reuse']['ttft_kept_s']} s with cache, {r['reuse']['ttft_cleared_s']} s without "
              f"({r['reuse']['cached_prefix']} of {r['reuse']['prompt_tokens']} tokens reusable)")

    name = slug(hw.cpu)
    out_dir = ROOT / "results" / "hardware"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}-models.json").write_text(json.dumps({"hardware": hw.to_dict(), "ctx": a.ctx,
                                                             "models": results}, indent=1))
    by_arch: dict[str, list[float]] = {}
    for r in results:
        by_arch.setdefault(r["arch"], []).append(r["efficiency"])
    calib = {"measured_on": name, "bandwidth_gbs": hw.bandwidth_gbs,
             "efficiency": {k: round(sum(v) / len(v), 3) for k, v in by_arch.items()},
             "default": round(min(e for v in by_arch.values() for e in v), 3) if by_arch else 1.0}
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1))
    print(f"\ncalibration: {calib}")


if __name__ == "__main__":
    main()
