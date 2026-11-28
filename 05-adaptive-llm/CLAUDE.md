# 05 — Adaptive LLM (own small "ChatGPT", optimized for any hardware)

## Goal
A chat assistant that runs well on weak, GPU-less machines by using only as much model as each
request needs, and adapts automatically to whatever hardware it runs on.

## Key technical reality (agreed framing)
- Training a ChatGPT-quality model from scratch is impossible on CPU. So: build a tiny model from
  scratch to *demonstrate* the ideas, and build an adaptive inference engine around open models
  for real-world use.
- MoE ("use part of the model") saves **compute**, not memory: all experts still need to be loaded.
  On low-RAM machines the bigger wins are quantization, routing to smaller models, and speculative decoding.

## Part A — Tiny GPT from scratch (PyTorch, CPU-trainable, ~10–30M params)
- Own tokenizer (BPE), transformer, training loop; small dataset (e.g. TinyStories)
- **Mixture-of-Experts** layers: router activates top-k experts per token
- **Early exit**: stop at a shallower layer when confident (fewer layers on easy tokens)
- Measure quality vs compute for dense vs MoE vs early-exit

## Part B — Adaptive inference engine (real open models, GGUF via llama.cpp)
- Hardware profiler: RAM, cores, CPU flags (AVX2/AVX-512), GPU → picks model size + quant level
- Complexity router: easy prompts → tiny model, hard → larger model; cascade/escalate on low confidence
- Speculative decoding: small draft model + larger verifier
- KV-cache reuse, streaming, context management
- Benchmarks across machines: tokens/sec, RAM, quality (via 02-llm-evals)

## Decisions (2026-11-28)
- Package `tinygpt` in `src/tinygpt` (Part A) and `adaptive` in `src/adaptive` (Part B). PyTorch CPU wheel
  (download.pytorch.org/whl/cpu) is the only heavy dependency; numpy, pytest.
- **Root drive has ~3 GB free:** pip must use TMPDIR / PIP_CACHE_DIR on this drive (`.tmp/`, gitignored).
- Data: **TinyStories V2 (GPT-4 stories)**, CDLA-Sharing-1.0, Hugging Face `roneneldan/TinyStories`. Only the first
  ~200 MB of the 2.2 GB train file is downloaded (HTTP Range), plus the 22 MB validation file. Raw data in `data/raw/`
  (gitignored), rebuilt by `scripts/get_data.sh`. Checkpoints in `models/` (gitignored).
- Tokenizer: byte-level BPE written from scratch (no `tokenizers` library), trained on a sample of the train text.
- Part B reuses project 1's GGUF models (`../01-rag-from-scratch/models/llm/`: Qwen3.5-0.8B, Qwen2.5-1.5B,
  Qwen3.5-2B, Q4_K_M) and its compiled llama-cpp-python, rather than downloading or compiling again.
- Hardware: i5-5200U (2 cores / 4 threads, AVX2 + FMA + F16C, no AVX-512, no GPU), 15 GB RAM.

## Plan (phases; docs written with each phase)
A1. Data + byte-level BPE tokenizer (train, encode/decode, tests), tokenized dataset on disk
A2. Dense GPT baseline: model, training loop with checkpoints/resume, validation loss, sampling; measure CPU
    throughput first and size the model to what trains in a few hours
A3. Mixture-of-Experts: top-k router, load-balancing loss; dense vs MoE at equal *active* compute
A4. Early exit: per-layer heads, confidence threshold; quality vs average layers used
A5. Part A results + docs
B1. Hardware profiler; B2. llama.cpp runtime + model ladder; B3. complexity router + cascade;
B4. speculative decoding; B5. benchmarks via evalkit; Scout (project 4) offline backend demo; docs

## Status
- 2026-11-28: A1 done; A2 baseline done (val 2.235, ppl 9.34, 96 min). A3/A4 code done; `experiments/part_a.sh`
  queue training moe-4x-top1 → moe-8x-top2 → exit-d256-L6 → early_exit evals (log: `logs/part_a.log`).
- B1–B5 code written and unit-tested (49 tests) while the CPU trained; **measurements need an idle CPU**, so
  `experiments/part_b.sh` (detached, log `logs/part_b.log`) waits for "PART-A DONE", then runs bench_models
  (calibration), plan, route_eval run + export, speculative (Qwen3.5 0.8B→2B, Qwen2.5 0.5B→1.5B).
- Don't run llama.cpp or other heavy jobs while part_a.sh trains: it skews the tokens/s comparison.
- llama-cpp-python 0.3.37 was copied from project 1's venv (relocatable); Qwen2.5-0.5B draft in `models/llm/`.
- MoE: 8x-top2 val 2.120 beats dense 2.235. Top-1 gating bug (p/p = 1, no router gradient) fixed; buggy run kept as
  `models/moe-4x-top1-bug`. `experiments/part_a_rerun.sh` (detached) waits for PART-A DONE, retrains moe-4x-top1,
  writes samples for all models, then execs part_b.sh (the old waiting part_b.sh was stopped).
- After both queues: fill Results in concepts 3, 4, 5, 6, 7; RESULTS.md; README numbers; then the Scout offline
  demo (OpenAI-compatible server over the engine, Qwen `<tool_call>` parsing).
