# Timeline

What happened, in order, including the mistakes. Dates are 2026.

## Nov 28: Part A built and training; Part B written while the CPU was busy

| when | what | what we learned |
|---|---|---|
| A1 | TinyStories (first 200 MB via HTTP range) + byte-level BPE from scratch | a naive-merge test proves the fast trainer learns identical merges; 3.95 chars/token at 4,096 tokens |
| A2 | GPT, resumable training loop, throughput benchmark | **2 threads beat 4 at every size** (hyper-threads only add contention); sized the model so four runs fit in a night |
| A2 | first launch looked stuck (empty log) | not a bug: the first print wasn't flushed and fewer minutes had passed than assumed; added `flush=True` |
| A2 | `.view` crash on non-contiguous targets | use `.reshape` |
| A3 | Mixture-of-Experts layer + load-balancing loss | a slow per-token loop is the test oracle for the grouped implementation |
| A4 | early exits + threshold sweep | evaluation is teacher-forced; the KV-cache caveat is in the docs, not hidden |
| A2 done | dense baseline: val loss 2.235 (ppl 9.3) in 96 min | loss still falling: under-trained by design; samples went from word salad (step 250) to coherent stories |
| B1 | profiler + GGUF reader + planner | the GGUF metadata showed Qwen3.5 is a hybrid (6 of 24 layers cache keys/values): 12 KB/token vs Qwen2.5's 28 |
| B1 | first bandwidth test (one-thread copy) predicted 15.9 tok/s for a model measured at 18.9 | a ceiling below a measurement is wrong: switched to a multi-threaded *read* test |
| B1 | ARM fixture test failed | real bug: the board block at the end of a Pi's `/proc/cpuinfo` was counted as a fifth CPU |
| B2 | reused project 1's compiled llama.cpp by copying it | its libraries use `RUNPATH $ORIGIN`, so the copy is relocatable; `pip check` clean |
| B2 | second chat turn reused only 16 of 41 cached tokens | the empty think block was dropped from history; keeping it: everything but the end-of-turn marker is reused |
| B2 | read llama-cpp-python's cache code | Qwen3.5 cannot drop trailing cache entries (`kv_cache_seq_rm` returns False), so reuse only works when the new prompt extends the whole cache, which the fix above makes true |
| B3 | confidence for the cascade | `logits_all` would allocate 4 GB; a logits hook gives each token's probability for ~2 ms. 0.8B: "Paris" 0.88 ✓, "37 × 43 = 1041" 0.39 ✗; but a wrong GSM8K answer at 0.99 |
| B3 | smoke run of the routing eval | crash with one model (cascade pairs out of range): fixed |
| B4 | speculative decoding | `Llama.save_state()` would copy a 248k-float row per token: used per-sequence state copies (19.5 MB) instead; first check: output identical to the 2B's own greedy output |
| B4 | found while reading the source | `Llama.reset()` only zeroes a counter; low-level decoding must clear the cache itself, or stale entries corrupt attention |
| A3 | MoE run's first eval window overlapped with llama.cpp smoke tests | stopped running models while training: speed comparisons use the clean windows |
| B5 | routing headline on a held-out half | thresholds chosen on the other half, so the cascade is not tuned on its own test set |
| A3 | moe-4x-top1: val 2.345 (ppl 10.4) vs dense 2.235 | the MoE lost; first guess: "each top-1 expert trains on only a quarter of the tokens" |
| A3 | moe-8x-top2: val **2.120** (ppl 8.3), 12% slower than dense | **the guess was wrong**: top-2 of 8 also gives each expert a quarter of the tokens, and it won |
| A3 | read the gate code | **bug**: top-1 renormalised its one weight to p/p = 1, so the router got no gradient from the task (4e-9 vs 0.016); fixed as in Switch (gate = raw p), regression test fails on the old code, run redone after Part A (`part_a_rerun.sh`) |
| A4 | exit-d256-L6: final ppl 9.79 (dense 9.34), 1.8× slower to train; layer-1 exit ppl 13.8 (logit lens: 130) | exits work, but checking confidence = a 256 × 4,096 projection, **more than a layer**: with a check per layer, every τ > 0.4 costs more than the full model; a fixed layer-4 exit (0.72×, ppl 10.1) is the better trade |
| A3 rerun | moe-4x-top1 with the fixed gate: val **2.181** (ppl 8.85), was 2.345 | one line of gating code: from 0.11 worse than dense to 0.05 better; the learning router uses experts *less* evenly (0.15–0.40): it specialises |
| A5 | samples of the buggy run came out as garbage | not a model result: sampling loaded the old checkpoint with the *fixed* gate, which scales outputs by p; deleted, to be regenerated with the old gate |
