# 5. Hardware-aware model choice: what can this machine run, and how fast?

## Intuition
The same chat app might run on a 2015 laptop, a gaming PC or a MacBook. A fixed model choice is wrong on most of
them: too slow on the laptop, needlessly weak on the gaming PC. So the engine first asks *what machine is this?*,
then picks the biggest model that fits in memory and still answers at reading speed.

Two facts about running a language model on a CPU make that a calculation instead of a guess:

1. **Memory.** The whole model must be in RAM, because any token may use any weight. If it doesn't fit, the OS
   swaps to disk and generation slows 10–100×. Add the **attention (KV) cache**, which grows with the context
   length, and some working buffers.
2. **Speed is limited by memory bandwidth.** Generating one token multiplies the current vector by *every* weight
   matrix once. For a single sequence that is little arithmetic per byte loaded, so the CPU mostly waits for RAM:
   ```
   tokens/s  ≤  memory bandwidth (GB/s)  /  model size (GB)
   ```
   A 1 GB model on a machine that reads 20 GB/s can produce at most 20 tokens/s. That is why quantization (fewer
   bits per weight = fewer bytes to read) speeds things up as well as saving memory.

## What the profiler measures ([`hardware.py`](../../src/adaptive/hardware.py))
Standard library only, with one code path per OS (Linux `/proc`, macOS `sysctl`, Windows `kernel32` via ctypes):

| fact | why it matters |
|---|---|
| physical cores | thread count. Hyper-threads share a core's arithmetic units; Part A measured 2 threads beating 4 on this 2-core CPU |
| AVX2 / AVX-512 / FMA / F16C (x86), NEON / dot-product (ARM) | which fast kernels llama.cpp can use; without AVX2 an x86 CPU is several times slower |
| *available* RAM (not total, not "free") | what a model can use without pushing other programs into swap. Linux's `MemAvailable` counts reclaimable file cache; `MemFree` doesn't |
| GPU | Apple Silicon: offload everything (unified memory). NVIDIA: not handled, since there is none here to test on |
| memory bandwidth, **measured** | a large array (far bigger than the CPU caches) summed by one thread per core, best of 5 runs |

Bandwidth is measured, not looked up: the same "DDR3L-1600" laptop gets different numbers depending on whether one or
both memory slots are filled, and a spec sheet's peak is never reached. A *read* test matches the workload (weights
are only read), and it must use every core: one thread cannot keep the memory bus busy.

## Reading the model file ([`gguf.py`](../../src/adaptive/gguf.py))
llama.cpp's GGUF format starts with a header, a list of key-value metadata, and a table of tensors (name, shape,
type). A 100-line reader gets what the planner needs without loading any weights:

| model | params | quant | file | effective bits/weight | attention layers | KV cache per token |
|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.75B | Q4_K_M | 0.50 GB | 5.66 | 6 of 24 | 12 KB |
| Qwen2.5-1.5B | 1.78B | Q4_K_M | 1.04 GB | 5.03 | 28 of 28 | 28 KB |
| Qwen3.5-2B | 1.88B | Q4_K_M | 1.19 GB | 5.45 | 6 of 24 | 12 KB |

Two things the metadata taught us:
- **"4-bit" models are 5–5.7 bits per weight.** Q4_K_M keeps some tensors at higher precision, and in small models
  the embedding matrix is a big share of all weights: Qwen3.5's vocabulary has 248,320 tokens, so in the 0.8B model
  the embeddings alone are 248,320 × 1,024 = 254M weights, a third of the model.
- **Qwen3.5 is a hybrid.** Only every 4th layer is full attention (`full_attention_interval = 4`); the others are
  linear-attention layers with a fixed-size state. So its KV cache is **12 KB per token** against Qwen2.5's 28 KB,
  even though it is the bigger model: at a 32k context that is 0.38 GB against 0.9 GB.

KV cache = 2 (keys and values) × attention layers × KV heads × head size × context × 2 bytes (f16).

## The plan ([`plan.py`](../../src/adaptive/plan.py))
```
need   = file size + KV cache(context) + 0.35 GB buffers
budget = available RAM − 1 GB reserve for everything else
speed  ≈ efficiency × bandwidth / file size        (efficiency calibrated against real llama.cpp speeds in B2)
choice = the largest model with need ≤ budget and speed ≥ 5 tokens/s (about reading speed)
```
If nothing is fast enough, it takes the fastest model that fits; if nothing fits, it says so. Threads = physical
cores. For downloads it also prints a grid of model size × quantization (Q8_0 … Q2_K) and never recommends below
Q4_K_M: under 4 bits quality drops quickly, so a smaller model at 4 bits usually beats a bigger one at 2 bits.

## Results
*(added when measured on an idle CPU, after the Part A training runs)*

Already known from project 1's measurements on this laptop: Qwen3.5-0.8B 18.9 tok/s, Qwen2.5-1.5B 14.2,
Qwen3.5-2B 5.7. The 2B is 2.5× slower than the 1.5B although its file is only 15% bigger, so bandwidth alone does
not predict it. Hybrid attention has younger, slower CPU kernels in llama.cpp. The planner therefore needs a
measured efficiency **per architecture**, not one number for the machine.

## Interview questions
- *Why is LLM generation on a CPU memory-bound?* For each new token, every weight is read once and used in one
  multiply-add; the CPU can do far more arithmetic per second than RAM can deliver bytes.
- *Why does quantization make generation faster, not just smaller?* Fewer bytes per weight means fewer bytes to read
  per token, and reading is the bottleneck.
- *Why does a long context need more memory?* The KV cache stores keys and values for every past token in every
  attention layer.
- *Why "available" and not "free" memory?* Free memory excludes the file cache the OS would happily give up; available
  memory includes it, so it is what a new program can actually get.
