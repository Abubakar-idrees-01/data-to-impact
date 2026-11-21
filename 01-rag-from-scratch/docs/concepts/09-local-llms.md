# 9. Running an LLM on a laptop CPU

## The constraint
Our development machine is a 2015 Intel i5-5200U (2 cores / 4 threads, no GPU, 15 GB RAM). The goal is a
study assistant that works on machines like this, not just on cloud GPUs.

## Quantization
A model is billions of numbers (weights). Stored as 16-bit floats, a 2B-parameter model needs ~4 GB.
**Quantization** stores them with fewer bits:

| Format | Bits per weight | Size of a 2B model | Quality |
|---|---|---|---|
| FP16 | 16 | ~4 GB | reference |
| Q8_0 | ~8.5 | ~2.1 GB | almost identical |
| **Q4_K_M** (ours) | ~4.8 | ~1.3 GB | small loss, the usual sweet spot |
| Q3_K_M | ~3.9 | ~1.1 GB | noticeable loss |

"K-quants" quantize weights in small blocks, each with its own scale factor, so outliers in one block don't
ruin precision elsewhere. "_M" keeps some sensitive layers at higher precision.

Smaller weights help twice over. They need less RAM, and they are faster, because CPU inference is limited
mostly by **memory bandwidth**: every generated token reads every weight once.

## Two speeds you must know
1. **Prompt processing (prefill)**: reading your prompt. It runs in parallel over all prompt tokens, so it is
   faster per token, but prompts are long (instructions plus 4 passages ≈ 800–1200 tokens).
2. **Generation (decode)**: writing the answer, one token at a time, each needing a full pass over the weights.

Time to first token ≈ prompt tokens ÷ prefill speed. That's why we keep prompts short (top 4 chunks), and
why MCQs are generated from one passage at a time. Measured numbers for our laptop are in
[RESULTS.md](../RESULTS.md) (`python eval/run_eval.py speed`).

## llama.cpp
A C/C++ inference engine built for exactly this: SIMD kernels (AVX2 on our CPU), memory-mapped weights
(the model loads almost instantly and shares RAM pages), a KV cache, and **grammar-constrained decoding**.
We use it through `llama-cpp-python` ([`llm/llamacpp.py`](../../src/studyrag/llm/llamacpp.py)).

### Grammar-constrained JSON
For MCQs we need valid JSON every time. llama.cpp converts our JSON schema into a grammar and, at each step,
**masks out every token that would break the grammar**. The model can't produce invalid JSON, even a tiny one.

### Disabling "thinking"
Qwen3/3.5 models first write hidden reasoning in a `<think>` block. That improves hard reasoning, but at
~5 tokens/s it can add a minute before the answer starts. For retrieval-grounded answers the reasoning is mostly
unnecessary, so we pre-fill an empty `<think></think>` block. This is exactly what the official chat template
does when `enable_thinking=False`.

## Threads
llama.cpp is usually fastest with one thread per **physical** core. Hyper-threads share a core's execution
units and memory bandwidth, so extra threads often don't help. We default to physical cores and measure 2 vs 4.

## Model size does not predict speed
We measured three models on the same laptop ([RESULTS.md §6](../RESULTS.md)):

| model | size | generation | time to first token |
|---|---|---|---|
| Qwen3.5-0.8B | 0.53 GB | 18.9 tok/s | 6.3 s |
| Qwen2.5-1.5B | 1.12 GB | 14.2 tok/s | 13.4 s |
| Qwen3.5-2B | 1.28 GB | 5.7 tok/s | 19.4 s |

If memory bandwidth were the only cost, the 2B model would be about 15% slower than the 1.5B (1.28 vs 1.12 GB read
per token). Instead it is **2.5× slower**. Qwen3.5 mixes classic attention with a newer linear-attention design, and
llama.cpp's CPU kernels for it are younger and less optimized. Architecture and runtime support matter as much as size.

We still chose the 2B as the default, because on the real lecture slides its citations were far more reliable
(0.98 vs 0.80 citation support). Speed is a trade-off you measure, not a number you guess.

## The API escape hatch
If a machine is too slow, the same app can call any OpenAI-compatible API
([`llm/openai_compat.py`](../../src/studyrag/llm/openai_compat.py)). It's faster and stronger, but no longer
private, and the UI makes that trade-off explicit.

## Interview questions
- *Why is generation slower than prompt processing?* Decode is sequential and memory-bound (one full weight
  read per token); prefill batches many tokens per weight read.
- *How does quantization affect quality?* Measure it on your task. Q4_K_M is usually within a few percent of FP16 on benchmarks.
- *What is the KV cache?* Stored attention keys/values for previous tokens, so each new token doesn't recompute the whole prefix.
