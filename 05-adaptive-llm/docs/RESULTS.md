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

## Part B: the adaptive engine on real models
*(added when `experiments/part_b.sh` finishes: model speeds and memory, calibration, routing headline, speculative
decoding)*
