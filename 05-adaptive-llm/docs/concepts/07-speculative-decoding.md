# 7. Speculative decoding: let a small model guess, let the big model check

## Intuition
Decoding on a CPU is memory-bound: producing one token means reading every weight once (concept 5). Checking
5 tokens in one batch also reads every weight once, so it costs little more than producing one. If a small, fast
model could guess the next few tokens, the big model could confirm several of them for the price of one step.

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

**Expected speed-up.** With acceptance rate α, k drafts and draft/target cost ratio c:
```
tokens per round = (1 − α^(k+1)) / (1 − α)          cost per round ≈ 1 + k·c   (in big-model steps)
```
For α = 0.7, k = 4, c = 0.3 (0.8B vs 2B on this laptop): 2.77 tokens for 2.2 steps = 1.26×. The draft must be much
cheaper *and* agree often; on a CPU, where the small model is only ~3× faster, the margin is thin.

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
*(added when measured on an idle CPU, after the Part A training queue)*

## Interview questions
- *Why does verifying k tokens cost about the same as generating one?* Decode is memory-bound; the weights are read
  once per batch, and k + 1 positions in a batch are mostly extra arithmetic, which the CPU has spare.
- *Does speculative decoding change the output?* Not with the right acceptance rule: greedy acceptance gives exactly
  the target's greedy output; the probabilistic rule gives exactly its sampling distribution.
- *When does it not help?* When the draft is not much cheaper than the target, when they rarely agree (creative,
  high-temperature text), or when rollback is expensive.
- *Why must draft and target share a tokenizer?* The target verifies token ids; different vocabularies would need
  re-tokenisation of every draft.
