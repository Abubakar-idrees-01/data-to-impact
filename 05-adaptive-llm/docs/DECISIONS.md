# Decisions

Why X over Y, with the measurement behind each choice. Numbers are from the i5-5200U laptop (2 cores / 4 threads,
AVX2, 15 GB RAM, no GPU).

## Part A

**D1. Byte-level BPE written from scratch, not the `tokenizers` library.** The point of Part A is to understand
every piece. Bytes as the base alphabet mean no unknown tokens ever. A test checks the fast implementation
(incremental pair counts + heap) learns exactly the same merges as a naive one. Result: 4,096 tokens, 3.95
characters per token on TinyStories.

**D2. TinyStories V2, first 200 MB only.** Simple English that a 5M-parameter model can learn to write coherently
(Eldan & Li, 2023). A run sees 12.3M tokens of the 52M training tokens, so more data would not be used anyway.

**D3. 2 training threads, not 4.** Measured at every model size: 2 threads are 12–20% faster. The 4 "CPUs" are 2
cores with hyper-threading; dense matrix maths gains nothing from a core's second thread, only contention.

**D4. d = 256, 6 layers, 1,500 steps × 8,192 tokens.** Four comparable runs had to fit in a night. The throughput
table (concept 2) put d256-L6 at ~1,600 tok/s → ~2.1 h per run of 12.3M tokens; measured: 96 min (2,265 tok/s once
warm). Under-trained by research standards (loss still falling), but every variant gets the identical budget.

**D5. MoE compared at equal *active* compute, in two shapes.** A bigger model is expected to be better, so "MoE
beats dense" alone says little. Both MoE runs use 5.85M active parameters per token (the dense model: 5.84M) and
15.3M in total: 4 experts of full width, top-1; and 8 half-width experts, top-2 (the fine-grained variant).
Measured: 8×top-2 beat dense by 0.115 val loss (ppl 8.33 vs 9.34) at 12% lower CPU throughput and 2.6× the weights;
4×top-1 (fixed gate) by 0.054 (ppl 8.85) at 6% lower throughput.

**D5b. Top-1 gate = raw router probability, not renormalised (Switch), top-2 renormalised (Mixtral).** Found by a
result, not by review: the first top-1 run lost to dense, and the cause was `p / p = 1`, a constant gate that gave the
router zero gradient from the task (4e-9 vs 0.016 measured). A regression test covers both; the rerun went from
val 2.345 to 2.181 (0.11 worse than dense → 0.05 better).

**D6. Early exits share the output layer; trained with the mean exit loss at weight 1.0.** 512 parameters per exit
instead of a full 1M-parameter head each. The dense model's logit lens is the untrained baseline.
Measured: layer-1 exit ppl 13.8 (logit lens 130), final layer 9.79 vs dense 9.34, 1.8× training time. The sweep
reports compute *including* the confidence check (one 256 × 4,096 projection per check, 1.2 layers' worth), because
leaving it out makes every τ look like a saving; counted, only τ ≤ 0.4 breaks even and a fixed layer-4 exit wins.

## Part B

**D7. Reuse project 1's models and llama.cpp build.** Three Q4_K_M GGUF models (0.8B, 1.5B, 2B) are already on
disk; compiling llama-cpp-python takes ~10 minutes of both cores. The compiled package is relocatable (its
libraries have `RUNPATH $ORIGIN`), so it was copied into this venv and `pip check` passes. One extra download:
Qwen2.5-0.5B (0.49 GB, `models/llm/`, gitignored) as a same-tokenizer draft for Qwen2.5-1.5B.

**D8. Profiler with the standard library only (no psutil).** psutil has no CPU-flag information anyway, and
`/proc`, `sysctl` and `kernel32` give everything in ~100 lines. A fixture test caught one parser bug: on ARM
(Raspberry Pi style `/proc/cpuinfo`) the trailing board block was counted as a fifth CPU.

**D9. Multi-threaded *read* bandwidth, not a single-threaded copy.** Generating a token only reads weights. The
first version (one-thread `np.copyto`) predicted a ceiling of 15.9 tok/s for Qwen3.5-0.8B, which project 1 had
measured at 18.9: a ceiling below the measurement is wrong. One thread cannot saturate the memory bus.

**D10. Speed calibration per architecture, not one factor per machine.** Qwen3.5-2B's file is 15% bigger than
Qwen2.5-1.5B's, yet project 1 measured it 2.5× slower (5.7 vs 14.2 tok/s): its hybrid linear-attention layers have
less optimised CPU kernels. `bench_models` measures efficiency = measured ÷ bandwidth ceiling for each architecture.

**D11. Keep the empty think block in past assistant turns.** The official Qwen3.5 template drops it from history,
so the history no longer matches the tokens in the KV cache. Measured on the second turn of a chat: 16 of 41 prompt
tokens matched the cache without it, 26 of 45 (everything but the end-of-turn marker) with it. It matters more for
Qwen3.5 than usual: its cache cannot drop trailing entries (`kv_cache_seq_rm` returns False), so llama.cpp can only
reuse the cache when the new prompt extends *all* of it; any mismatch means prefilling the whole conversation again
(read from llama-cpp-python's prefix-matching code; `bench_models` measures the effect on time to first token).

**D12. Confidence from a logits-processor hook, not `logits_all=True`.** `logits_all` keeps scores for every
position: n_ctx × 248,320 × 4 bytes = 4 GB at a 4k context. The hook sees each step's scores once (~2 ms softmax per
token). Checked: output identical with and without the hook; 0.8B "Paris" 0.88 (right), "37 × 43 = 1041" 0.39
(wrong), an invented Tour de France winner 0.42.

**D13. Routing evaluated on auto-graded questions, policies simulated offline.** Capitals, seeded arithmetic and
GSM8K have exact answers, so no LLM judge (and no API quota) is needed. Every model answers every question once;
any policy (fixed model, router, cascade at any τ, oracle) is then computed exactly from those results instead of
re-running models per policy and threshold.

**D14. Router rules fixed before seeing results.** Otherwise the rules would be fitted to the test set. Known
limit: on this question set the three families differ on the surface (the GSM8K prompt even says "step by step"),
so the router effectively recognises the family; difficulty *within* a family (Paris vs Naypyidaw) is invisible to
it. That is what the cascade is for.

**D15. Speculative decoding on llama.cpp's low-level batch API, greedy.** llama-cpp-python supports only
prompt-lookup drafting, not a draft model. Greedy acceptance makes the output exactly the target's own greedy
output, which is testable token by token (first check: identical). Rollback: cache removal where possible,
per-sequence state snapshots (`llama_state_seq_*`, 19.5 MB for Qwen3.5-2B) where not; `Llama.save_state()` was
rejected because it copies a vocabulary-sized score row per token and scans a 508 MB array on restore.
Measured: ×0.24–0.64 for every pair, prompt and k, because a pass over 5 tokens costs 3.4–3.8 single steps on this
CPU; **so the engine does not use speculative decoding on this machine.** `pass_cost()` measures V(n) on any machine;
the technique is worth enabling only where V(k + 1) + k·c is clearly below the expected tokens per round.
