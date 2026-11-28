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
Measured by [`bench_models.py`](../../src/adaptive/bench_models.py) on the idle laptop (after Part A), context 4,096,
a fixed ~370-token prompt, 96 generated tokens, each model in its own process
([`results/hardware/`](../../results/hardware/)).

**The profile:** i5-5200U, 2 cores / 4 threads, AVX2 + FMA + F16C, 10.7 of 15.5 GB available, no GPU, measured read
bandwidth **21.0 GB/s**.

### Speed by thread count (decode tok/s; prefill tok/s in brackets)
| model | 1 thread | 2 threads | 4 threads | bandwidth ceiling | efficiency |
|---|---|---|---|---|---|
| Qwen3.5-0.8B (0.50 GiB) | 12.0 (43) | **18.5** (78) | 17.7 (79) | 39.4 | 0.47 |
| Qwen3.5-2B (1.19 GiB) | 5.3 (16) | **8.3** (31) | 7.6 (29) | 16.4 | 0.51 |
| Qwen2.5-1.5B (1.04 GiB) | 8.2 (20) | 10.3 (23) | **13.5** (37) | 18.8 | 0.72 |

- **Real decoding reaches 47–72% of the bandwidth ceiling**, so the ceiling alone over-promises by 1.4–2×. That is
  why the planner multiplies by a measured efficiency, and why it is kept per architecture: the hybrid Qwen3.5
  reaches 0.49, the plain-attention Qwen2.5 0.72. The calibration file stores `{"qwen35": 0.487, "qwen2": 0.719}`
  and uses the lowest (0.469) for an architecture it has never measured.
- **Two threads (= physical cores) is best for both Qwen3.5 models**, as in Part A's training benchmark. Qwen2.5-1.5B
  looked like the exception (4 threads 31% faster), but its 2-thread prefill (23 tok/s) barely beat 1 thread (20),
  the sign of a disturbed run. Re-checked: the speculative-decoding benchmark, later on the idle CPU, decoded the same
  model at **12.3–13.4 tok/s on 2 threads**, as fast as the "4-thread" figure. So the rule stays: threads = physical
  cores. (A full re-run of this table was spoiled by desktop load: bandwidth measured 12 GB/s instead of 21, every
  model slowed down, and it was discarded: [`contended/`](../../results/hardware/contended/).)

### Memory: estimate vs peak RSS (GiB = 2³⁰ bytes, as in the planner)
| model | file + KV cache (4k) | estimate (+0.35 GB buffers) | peak RSS | error |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.50 + 0.05 GB | 0.89 GB | 0.89 GB | 0.00 |
| Qwen3.5-2B | 1.19 + 0.05 GB | 1.59 GB | 1.92 GB | −0.33 |
| Qwen2.5-1.5B | 1.04 + 0.11 GB | 1.50 GB | 1.87 GB | −0.37 |

The fixed 0.35 GB for buffers fits the small model and **under-estimates the two larger ones by about a fifth**.

**Where the extra memory came from.** llama.cpp's load log and `/proc/self/status` (resident memory split into file
pages and anonymous memory) explain it. On AVX2 most Q4_K weight matrices are **repacked** into an interleaved 8×8
layout that the matrix kernels read faster. The repacked copy is new, anonymous memory: 161 MiB for the 0.8B,
491 MiB for the 2B and 597 MiB for the 1.5B. With the file memory-mapped (llama.cpp's default), the original pages of
those weights also stay resident, though they are never read again:

| model | resident file pages | anonymous (repacked copy + KV + buffers) | RSS |
|---|---|---|---|
| Qwen3.5-0.8B | 0.51 GiB = the whole file | 0.34 GiB | 0.84 GiB |
| Qwen3.5-2B | 1.20 GiB = the whole file | 0.66 GiB | 1.86 GiB |
| Qwen2.5-1.5B | 1.06 GiB = the whole file | 0.77 GiB | 1.82 GiB |

So every repacked weight was in RAM twice. **The fix: load with `use_mmap=False`**, which reads each weight
straight into its final buffer:

| model | peak RSS with mmap | without | saved | load time without |
|---|---|---|---|---|
| Qwen3.5-0.8B | 0.89 GiB | 0.74 GiB | −17% | 2.0 s |
| Qwen3.5-2B | 1.92 GiB | 1.39–1.44 GiB | −25% | 2.3–3.3 s |
| Qwen2.5-1.5B | 1.87 GiB | 1.24–1.29 GiB | −32% | 2.1–2.8 s |

The engine now loads this way (`runtime.USE_MMAP = False`). The planner's estimate then sits 0.1–0.2 GiB *above* the
measured peak, which is the safe side. Without mmap the pages could not be shared between two processes running the
same model, which this engine never does.

### The plan and the cached second turn
- **First plan:** Qwen3.5-2B, 2 threads, 4k context: "the largest model that fits and runs at ≥ 5 tok/s". It
  predicted 7.9 tok/s and the model measured 8.3. That prediction uses the efficiency measured on these same models,
  so the real test of the calibration is a different machine.
- **The rule was wrong, measured by the routing eval ([concept 6](06-routing-and-cascades.md)).** The 2B answered no
  better than the 1.5B and was 1.6× slower. The planner now reads the routing eval's answers when they exist and takes
  **the fastest model that is not significantly less accurate than the most accurate one** (exact McNemar test on
  the same 128 questions, p < 0.05); "largest" is only the fallback without measurements:

  ```
  model                         tok/s    acc
  Qwen3.5-0.8B                   19.2  0.664     significantly worse than the 2B
  qwen2.5-1.5b-instruct          13.5  0.82      vs the 2B: McNemar p = 0.50  → choice
  Qwen3.5-2B                      8.0  0.852     most accurate
  ```
  The threshold is a standard significance level, not a margin tuned until the answer came out "right".
- **KV-cache reuse on the second chat turn** (the first turn stays cached; only the new message is prefilled):

| model | 2nd-turn time to first token, cache kept | cache cleared | reused tokens |
|---|---|---|---|
| Qwen3.5-0.8B | **0.49 s** | 6.47 s | 469 of 499 |
| Qwen3.5-2B | **1.04 s** | 13.74 s | 438 of 467 |
| Qwen2.5-1.5B | **0.84 s** | 12.20 s | 417 of 442 |

  **13× faster replies** from keeping the conversation in the cache. On a CPU, prefill (30–80 tok/s) is the slow part
  of a long chat, not decoding.
- **Quantization grid** for downloads on this machine: the best pick is Q8_0 at 0.5–1B, Q6_K at 2B and nothing above
  4B (Q4_K_M 4B: 4.0 tok/s, under the 5 tok/s floor). The grid never recommends below Q4.

## Interview questions
- *Why is LLM generation on a CPU memory-bound?* For each new token, every weight is read once and used in one
  multiply-add; the CPU can do far more arithmetic per second than RAM can deliver bytes.
- *Why does quantization make generation faster, not just smaller?* Fewer bytes per weight means fewer bytes to read
  per token, and reading is the bottleneck.
- *Why does a long context need more memory?* The KV cache stores keys and values for every past token in every
  attention layer.
- *Why "available" and not "free" memory?* Free memory excludes the file cache the OS would happily give up; available
  memory includes it, so it is what a new program can actually get.
