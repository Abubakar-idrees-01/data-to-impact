# Results

Every number here was measured on the owner's laptop: Intel i5-5200U (2 cores / 4 threads, AVX2, no GPU), 15 GB
RAM, PyTorch CPU, 2 threads. How each was measured is in the linked concept note.

## Part A: a tiny GPT trained from scratch

Every run: TinyStories V2, our 4,096-token BPE tokenizer, d = 256, 6 layers, 8 heads, context 256, 1,500 steps of
32 windows = **12.3M training tokens**, the same 256 validation windows (65,536 positions).

| run | total params | active per token | val loss | ppl | training tok/s | wall clock |
|---|---|---|---|---|---|---|
| `dense-d256-L6` (baseline) | 5.84M | 5.84M | 2.235 | 9.34 | 2,265 | 96 min |
| `moe-4x-top1` (fixed gate) | 15.28M | 5.85M | 2.181 | 8.85 | 2,130 (−6%) | 98 min |
| **`moe-8x-top2`** | 15.28M | 5.85M | **2.120** | **8.33** | 1,985 (−12%) | 105 min |
| `moe-4x-top1-bug` (router got no task gradient) | 15.28M | 5.85M | 2.345 | 10.44 | 2,120 (−6%) | 108 min* |
| `exit-d256-L6` (exit after every layer) | 5.84M | 5.84M | 2.281 | 9.79 | 1,240 (−45%) | 168 min |

\* its first two eval windows overlapped with other jobs; tokens/s are from the clean windows.

### Mixture-of-Experts ([concept 3](concepts/03-mixture-of-experts.md))
- At equal compute per token, **2.6× the parameters bought 0.05–0.115 nats**: perplexity 9.34 → 8.85 (4 experts,
  top-1) and → 8.33 (8 half-width experts, top-2). The fine-grained shape wins.
- **On a CPU the same arithmetic runs 6–12% slower** (grouping tokens by expert, smaller matrix multiplies), and all
  experts must be in memory.
- **One line of gating code decided the result.** Renormalising a single top-1 weight gives `p / p = 1`, so the
  router got no gradient from the task (measured 4 × 10⁻⁹ vs 0.016). Fixed, the same run went from 0.11 worse than
  dense to 0.05 better.

### Early exit ([concept 4](concepts/04-early-exit.md))
- Training an exit after every layer took layer 1 from perplexity **130 → 13.8**, at a cost of 0.05 nats at the
  final layer and 1.8× the training time.
- **Counting the confidence check honestly reverses the usual conclusion.** Each check is a 256 × 4,096 projection,
  more than a layer's compute. With a check per layer, τ = 0.6 skips 27% of layers but costs 1.35× the full model.
  Only τ ≤ 0.4 breaks even (ppl 11.1).
- **A fixed exit at layer 4 is the better trade**: 0.72× the compute at ppl 10.1. Exits are bimodal (layer 1 or the
  end), so one check after layer 1 would capture most of what the dynamic rule can save.

### Samples (prompt "Once upon a time", temperature 0.8, top-k 40, seed 0)
[`results/samples/`](../results/samples/). All models write grammatical stories with dialogue. All still lose track
of objects and names within a few sentences (Tim's "boat" makes "his boat go fast"). At 12M tokens, consistency
is what none of them has learned yet, which matches the loss still falling at the end of every run.
The buggy MoE checkpoint must be sampled with the gate it was trained with
([`experiments/sample_bug_run.py`](../experiments/sample_bug_run.py)): loss 2.390 under its own gate, 4.406 under
the fixed one.

## Part B: the adaptive engine on real models
Qwen3.5-0.8B, Qwen2.5-1.5B and Qwen3.5-2B (Q4_K_M GGUF) through llama.cpp, 2 threads, idle CPU unless stated.

### The machine and the models ([concept 5](concepts/05-hardware-aware-model-choice.md))
| | Qwen3.5-0.8B | Qwen2.5-1.5B | Qwen3.5-2B |
|---|---|---|---|
| decode tok/s (2 threads) | 18.5 | 12.3–13.4 | 8.3 |
| share of the bandwidth ceiling (21 GB/s) | 0.47 | 0.72 | 0.51 |
| peak RAM, mmap (default) → without mmap | 0.89 → 0.74 GiB | 1.87 → 1.24–1.29 GiB | 1.92 → 1.39–1.44 GiB |
| 2nd chat turn, time to first token: cache kept / cleared | 0.49 / 6.47 s | 0.84 / 12.20 s | 1.04 / 13.74 s |
| accuracy on the 128 questions | 0.664 | 0.820 | 0.852 |

- Keeping the conversation in the KV cache makes the next reply start **13× sooner**.
- **Weights were in RAM twice.** llama.cpp repacks Q4_K weights for AVX2, and with mmap the original pages stay
  resident too. Loading without mmap saves 17–32% at the same load time; the engine now does.
- **The planner's first rule, "largest model that fits", picked the 2B.** It now picks the fastest model that is
  not significantly less accurate than the best (McNemar on the 128 questions): **Qwen2.5-1.5B** (vs 2B: p = 0.50;
  the 0.8B vs 2B: p < 0.001).

### Routing and cascades ([concept 6](concepts/06-routing-and-cascades.md))
Held-out half (64 questions), thresholds chosen on the other half:

| policy | accuracy [95% CI] | s / question |
|---|---|---|
| **always Qwen2.5-1.5B** | 0.812 [0.70, 0.89] | **5.13** |
| always Qwen3.5-2B | 0.797 [0.68, 0.88] | 8.33 |
| rule router | 0.719 [0.60, 0.81] | 7.88 |
| cascade through all three (τ = 0.95) | 0.828 [0.72, 0.90] | 7.95 |
| oracle (smallest correct model, all 128) | 0.922 | 4.82 |

- No policy beats always using the 1.5B (evalkit McNemar: p ≥ 0.15 for every comparison), and the close ones are
  40–60% slower.
- **Confidence works for short answers and fails after reasoning.** The AUROC of answer-token confidence for right vs
  wrong is 0.85–1.00 on capitals and arithmetic and 0.51–0.66 on GSM8K, where 42 of 54 wrong answers had
  confidence ≥ 0.9.

### Speculative decoding ([concept 7](concepts/07-speculative-decoding.md))
- **Slower in all 24 settings** (2 pairs × 4 prompts × k = 2, 4, 6): ×0.24–0.64 of the target alone.
- **Why:** on 2 cores a pass over 5 tokens costs 3.4–3.8 single-token passes (compute-bound), not ≈ 1 as on a GPU.
  With the measured pass costs the speed-up formula predicts the Qwen2.5 pair within 0.05. Even 100% acceptance
  would give 0.76–0.97×.
- **Batched and single-token llama.cpp passes differ** by up to 1.7 in the logits. So near-ties flip, and the
  output matches the target's one-token-at-a-time greedy output only up to the first such flip.
