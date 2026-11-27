# How it works

Two parts that share one idea: **use only as much model as each token or request needs.**

## Part A: a tiny GPT, trained on a laptop CPU (`src/tinygpt/`)

```
TinyStories V2 (first 200 MB) ──► tokenizer.py: byte-level BPE, 4,096 tokens, trained from scratch
        │                                    │
        └──► data.py: encode every story ──► data/tokens/train.bin, valid.bin   (uint16, memory-mapped)
                                                     │
                                  train.py: random 257-token windows, AdamW, warm-up + cosine,
                                  fixed validation windows, atomic checkpoints, exact resume
                                                     │
              ┌──────────────────────────────────────┼──────────────────────────────────────┐
              ▼                                      ▼                                      ▼
     dense-d256-L6 (5.84M)              moe-4x-top1 / moe-8x-top2               exit-d256-L6
     every parameter, every token       15.3M params, 5.85M active:             an exit after every layer,
                                        a router picks experts per token        trained with the mean exit loss
              │                                      │                                      │
              └──────────── same 12.3M tokens, same steps, same validation windows ─────────┘
                                                     │
                       results: loss curves (models/*/log.jsonl), samples, early_exit.py sweeps
```

| file | role |
|---|---|
| [`tokenizer.py`](../src/tinygpt/tokenizer.py) | BPE training (incremental pair counts + heap), encode/decode, save/load |
| [`data.py`](../src/tinygpt/data.py) | stories → tokens on disk |
| [`model.py`](../src/tinygpt/model.py) | GPT: pre-norm blocks, causal attention, tied embeddings; `MoE` layer; early exits |
| [`train.py`](../src/tinygpt/train.py) | resumable CPU training loop; logs loss, speed, expert use |
| [`bench.py`](../src/tinygpt/bench.py) | training throughput by model size and thread count |
| [`early_exit.py`](../src/tinygpt/early_exit.py) | per-layer quality and the confidence-threshold sweep |
| [`sample.py`](../src/tinygpt/sample.py) | write stories with a trained model |

## Part B: an engine that fits real models to the machine (`src/adaptive/`)

```
                 ┌──────────────────────────── once per machine ────────────────────────────┐
                 │ hardware.py   cores, SIMD flags, available RAM, GPU, measured bandwidth   │
                 │ gguf.py       each model file's size, layers, KV-cache size per token     │
                 │ plan.py       fits? (file + KV + buffers ≤ RAM − reserve)                 │
                 │               fast enough? (efficiency × bandwidth / size ≥ 5 tok/s)      │
                 │ bench_models  measured speeds → per-architecture efficiency (calibration) │
                 └───────────────────────────────────────┬──────────────────────────────────┘
                                                         ▼
request ──► router.py: prompt features ──► start rung ──► runtime.py: LocalModel on the Ladder
                (numbers, maths words,                     (llama.cpp, streamed, timed; models
                 reasoning words, code, length)             loaded on demand, LRU-unloaded)
                                                                       │
                                                  confidence = geometric mean probability
                                                  of the answer tokens (logits hook)
                                                                       │
                                          ≥ τ ──► answer        < τ ──► next rung up (cascade)

speculative.py: for one model's reply, a small same-family model drafts k tokens; the big one checks them in one pass
```

| file | role |
|---|---|
| [`hardware.py`](../src/adaptive/hardware.py) | profile (Linux `/proc`, macOS `sysctl`, Windows `kernel32`); read-bandwidth test |
| [`gguf.py`](../src/adaptive/gguf.py) | GGUF metadata + tensor table, stdlib only; KV-cache size |
| [`plan.py`](../src/adaptive/plan.py) | choose model / context / threads; quantization grid for downloads |
| [`runtime.py`](../src/adaptive/runtime.py) | `LocalModel` (ChatML, streaming, TTFT/decode timing, KV reuse, confidence), `Ladder` |
| [`bench_models.py`](../src/adaptive/bench_models.py) | per-model speed, threads, peak RSS, cache reuse; writes the calibration |
| [`tasks.py`](../src/adaptive/tasks.py) | 128 auto-graded questions: capitals, arithmetic, GSM8K |
| [`router.py`](../src/adaptive/router.py) | rule router, cascade, policy evaluation, oracle |
| [`route_eval.py`](../src/adaptive/route_eval.py) | run every model on every question once; simulate every policy |
| [`speculative.py`](../src/adaptive/speculative.py) | greedy speculative decoding on llama.cpp's batch API |
| [`server.py`](../src/adaptive/server.py) | OpenAI-compatible `/v1/chat/completions` over the ladder; renders each model's own template, parses Qwen tool calls |

### One request, end to end
1. **Plan (startup).** The profiler and planner decide which models the ladder may use, the context size and the
   thread count (physical cores).
2. **Route.** The router reads the last user message and picks a starting rung.
3. **Generate.** `LocalModel` builds a ChatML prompt (with an empty think block for Qwen3.5, also kept in past turns
   so the KV cache matches the whole history), streams tokens and records each token's probability.
4. **Escalate?** If the answer's confidence is under τ, the same messages go to the next rung; its answer is used.
5. **Next turn.** The conversation so far is still in the model's KV cache; only the new message is prefilled.

### Two things llama.cpp's Python wrapper does not do for us
- **Per-token confidence** without `logits_all=True` (which would allocate n_ctx × vocab floats: 4 GB for Qwen3.5's
  248k vocabulary at 4k context): a logits-processor hook sees each step's scores and records the chosen token's
  probability.
- **Taking back tokens on hybrid models.** Qwen3.5's linear-attention layers cannot drop their last entries, so
  speculative decoding snapshots each sequence's state (`llama_state_seq_get_data`, 19.5 MB) and restores it on a
  rejected draft. `Llama.save_state()` is unusable for this: it copies a 248k-float score row for every token.
