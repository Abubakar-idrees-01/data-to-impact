# 7. Speculative decoding: let a small model guess, let the big model check

## Intuition
Producing one token means reading every weight once (concept 5). Checking 5 tokens in one batch also reads every
weight once, so *if memory is the only bottleneck* it costs little more than producing one. Then a small, fast model
could guess the next few tokens and the big model could confirm several of them for the price of one step. That
"if" is the whole question on a CPU, and on this laptop the answer is no (see Results).

Many tokens are easy to guess: "The capital of France is" → "Paris"; the second half of a word; closing brackets
in code. A 0.8B model guesses those just as well as a 2B one.

## The algorithm (greedy version, [`speculative.py`](../../src/adaptive/speculative.py))
Each round, with `last` = the newest token so far:
1. **Draft:** the small model produces d₁ … d_k, one at a time (cheap).
2. **Verify:** the big model reads [last, d₁ … d_k] in *one* batch, giving its own choice after every position.
3. **Accept** drafts while they equal the big model's choice; at the first mismatch, use the big model's choice
   instead. Every round therefore yields between 1 and k + 1 tokens.
4. **Take back** the rejected drafts in both models' caches.

Because a draft is accepted only when it *is* the big model's greedy choice, the output is exactly what the big
model would write alone (up to rounding differences between batched and single-token maths); the benchmark checks
this token by token. Sampling with temperature needs the probabilistic acceptance rule of Leviathan et al. (2023)
and Chen et al. (2023), which keeps the output *distribution* identical instead.

**Expected speed-up.** With acceptance rate α, k drafts, draft/target cost per token c, and V(n) = the cost of one
target pass over n new tokens (in single-token steps):
```
tokens per round = (1 − α^(k+1)) / (1 − α)          cost per round = k·c + V(k + 1)
```
The textbook version assumes V(n) ≈ 1 (true on a GPU, where one token leaves most of the compute idle): for α = 0.7,
k = 4, c = 0.3 that is 2.77 tokens for 2.2 steps = 1.26×. **On this CPU we measured V(5) = 3.4–3.8**, and the
speed-up becomes 2.77 / 4.6 = 0.6×.

## Taking back tokens: easy for transformers, not for Qwen3.5
Step 4 needs the KV cache to forget its last entries. Transformers can (`kv_cache_seq_rm`, measured: works on
Qwen2.5). Qwen3.5 interleaves linear-attention layers that keep a single running state per sequence: there is no
per-token entry to delete, and llama.cpp refuses (measured: returns False). The fix: before each round, copy the
sequence's state (`llama_state_seq_get_data`: 19.5 MB for Qwen3.5-2B with a short prompt); on a rejection,
restore it and re-feed the accepted tokens, which costs an extra pass. `Llama.save_state()` would also work but
copies a 248k-float score row for every token so far, which is far too slow.

Two pairs are measured: Qwen3.5-0.8B → Qwen3.5-2B (snapshots) and Qwen2.5-0.5B → Qwen2.5-1.5B (cache removal).
Draft and target must share a tokenizer, since the target checks the draft's token ids.

## Results
Four prompts, 128 new tokens at most, greedy, 2 threads, idle CPU ([`results/speculative/`](../../results/speculative/)).
The first run is kept in `first-run/`: it gave Qwen2.5 Qwen3.5's empty think block as plain text, fixed here.

### Speculative decoding is slower here, for every pair, prompt and k
| pair | target alone | k = 2 | k = 4 | k = 6 | acceptance (k = 2 → 6) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B → 2B | 9.4–9.6 tok/s | ×0.42–0.64 | ×0.30–0.61 | ×0.24–0.51 | 0.50–0.87 → 0.24–0.68 |
| Qwen2.5-0.5B → 1.5B | 12.3–13.4 tok/s | ×0.46–0.63 | ×0.43–0.59 | ×0.31–0.53 | 0.43–0.82 → 0.26–0.61 |

Code (the Fibonacci prompt) is the best case, as expected: 0.87 acceptance, 2.7 tokens per round at k = 2. Still a
slowdown (×0.64).

### Why: on 2 cores, checking n tokens costs almost n steps
One forward pass over n new tokens after a 300-token context (median of 5), relative to one token:

| new tokens | 1 | 2 | 3 | 5 | 7 |
|---|---|---|---|---|---|
| Qwen3.5-2B | 113 ms | ×2.26 | ×2.47 | ×3.43 | ×4.87 |
| Qwen2.5-1.5B | 67 ms | ×1.89 | ×2.77 | ×3.83 | ×5.50 |

Draft cost per token: c = 0.43 (0.8B vs 2B) and 0.59 (0.5B vs 1.5B).

For one token, the CPU is bandwidth-bound: it waits for weights. With a few tokens per pass it becomes
**compute-bound**, because two cores of AVX2 have no spare arithmetic. Each extra token costs 0.5–0.8 of a full step
instead of nearly nothing. A GPU, where one token uses a small fraction of the arithmetic units, is where the
technique pays.

The cost model above with the *measured* V(n) and c predicts the Qwen2.5 pair within ±0.05 in all 12 cases (e.g.
Fibonacci k = 4: predicted ×0.59, measured ×0.59). For Qwen3.5 the measured speed-up is about 72% of the prediction:
the rest is the 12–86 state restores per prompt that its non-rewindable cache forces.

**Even if every draft were accepted (α = 1),** the best case here is ×0.76–0.81 for the Qwen2.5 pair and ×0.90–0.97
for Qwen3.5. Speculative decoding cannot speed up these pairs on this machine. It would need a draft far cheaper than
c ≈ 0.5, or hardware with idle compute.

### "Identical to greedy": not exactly, and why that is expected
In 6 of 24 runs the output matched the target's own greedy output token for token. The others matched up to a
point (token 4 to 111) and then took another path. That is not an acceptance bug. **llama.cpp is not batch-invariant
on this CPU.** We generated 60 tokens one at a time, then re-scored the same 60 tokens in one batch:

| | logit difference, median / max | top choice flipped at |
|---|---|---|
| Qwen3.5-2B | 0.62 / 1.03 | 1 position (top-2 margin 0.065) |
| Qwen2.5-1.5B | 0.92 / 1.72 | 3 positions (margins 0.12, 0.04, 0.009) |

Batched and single-token passes use different kernels and accumulate in a different order. Where the top two tokens
are nearly tied, the winner can change, and from there the two outputs are different but equally "greedy" texts.
Every token the speculative decoder emits is still the target's top choice in the pass that checked it.

**For an interview:** "I implemented greedy speculative decoding on llama.cpp's batch API, with state snapshots for
Qwen3.5's recurrent layers. On my 2-core laptop it was 0.24–0.64× the target's speed for every setting. I measured
why: verifying 5 tokens cost 3.4–3.8 single-token steps, because the CPU becomes compute-bound as soon as a pass has
more than one token. With the measured pass costs the speed-up formula predicted the Qwen2.5 results within 0.05.
Even 100% acceptance would top out at 0.8–0.97×. It's a GPU technique."

## Interview questions
- *Why does verifying k tokens cost about the same as generating one?* On a GPU, single-token decode is memory-bound:
  the weights are read once per batch, and k + 1 positions are mostly extra arithmetic, which the GPU has spare. On
  this 2-core CPU it does not: 5 tokens cost 3.4–3.8 steps (measured), because the arithmetic is the bottleneck too.
- *Does speculative decoding change the output?* Not with the right acceptance rule: greedy acceptance gives exactly
  the target's greedy output; the probabilistic rule gives exactly its sampling distribution.
- *When does it not help?* When the draft is not much cheaper than the target, when they rarely agree (creative,
  high-temperature text), when rollback is expensive, and when the hardware has no spare compute for the
  verification batch. This laptop has the first and the last problem, and Qwen3.5 the third as well.
- *Why must draft and target share a tokenizer?* The target verifies token ids; different vocabularies would need
  re-tokenisation of every draft.
