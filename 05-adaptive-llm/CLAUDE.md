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

## Status
Not started.
