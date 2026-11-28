# 3. Mixture-of-Experts: more parameters, same compute per token

## Intuition
A dense model runs **every** parameter for **every** token. But "the" and "dinosaur" probably don't need the same
knowledge. A Mixture-of-Experts (MoE) layer replaces the single feed-forward network with several smaller-or-equal
"experts" and a **router** that sends each token to only one or two of them. The model has more parameters (more
places to store knowledge), but each token still pays for only one feed-forward. That is how models like Mixtral and
DeepSeek grow large without running all of themselves per token.

**What MoE does not save: memory.** Every expert must be loaded, because the next token may need any of them. On a
low-RAM laptop that is the wrong trade, which is why Part B saves memory a different way: quantization, and routing
between whole models.

## The algorithm ([`model.py`](../../src/tinygpt/model.py), class `MoE`)
For each token vector `x` (of size `d` = 256):
1. **Route:** `p = softmax(x · W_r)`, one probability per expert. `W_r` is a learned `d × n_experts` matrix, so the
   router costs almost nothing (1,024 parameters for 4 experts).
2. **Choose:** keep the `top_k` largest probabilities. With `top_k` ≥ 2, renormalise them to sum to 1 (as Mixtral
   does). With `top_k` = 1, keep the raw probability as the weight (as the Switch Transformer does), because
   renormalising a single weight gives `p / p = 1`: a constant, through which no gradient reaches the router. We
   learned this the hard way (see Results).
3. **Mix:** `output = Σ weight_e · expert_e(x)` over the chosen experts only. Each expert is an ordinary feed-forward
   network (expand, GELU, project back).

The code groups tokens by expert (`torch.where(chosen == e)`) so each expert runs once on its batch of tokens, then
scatters the results back with `index_add_`. A test compares this against a slow loop over individual tokens.

### The collapse problem, and the load-balancing loss
Left alone, routing collapses: whichever expert is slightly better early gets more tokens, so it trains more, so it
gets even more tokens. The other experts become dead weight. The fix used by the Switch Transformer is an extra loss:

```
f_e = share of tokens whose first choice is expert e        (what actually happened)
P_e = mean router probability given to expert e              (what the router wants, and is differentiable)
aux = n_experts · Σ_e f_e · P_e
```
`aux` = 1.0 when routing is perfectly even (`f_e = P_e = 1/E`) and up to `E` when one expert takes everything
(both tests). Training minimises `loss + 0.01 · aux`: a gentle push towards even use that doesn't override what the
router learns. `f_e` is logged per layer as `expert_share` in `log.jsonl`.

## Our comparison: equal *active* compute
All three runs use the same attention, the same 12.3M training tokens and the same steps. Only the feed-forward differs:

| run | feed-forward per layer | total params | active params per token |
|---|---|---|---|
| `dense-d256-L6` | 1 network, 1,024 wide | 5.84M | 5.84M |
| `moe-4x-top1` | 4 experts, 1,024 wide each, top-1 | 15.28M | 5.85M |
| `moe-8x-top2` | 8 experts, 512 wide each, top-2 | 15.28M | 5.85M |

The two MoE variants have the same parameters and the same active compute (one 1,024-wide expert, or two 512-wide
ones). They test whether *finer* experts, combined two at a time, beat coarse ones used alone (the idea behind
DeepSeek-MoE's "fine-grained experts").

The question is not "is the MoE better?" (it has 2.6× the parameters, so it is expected to be; see Results for what happened), but **how much better per
unit of compute**, and what it costs in wall-clock speed and memory on a CPU.

## Results

All runs: 1,500 steps × 32 × 256 = 12.3M tokens, the same validation windows, 2 threads on the i5-5200U.

### Validation loss (perplexity) during training

| step | tokens | dense | moe-8x-top2 | moe-4x-top1 (bug) | moe-4x-top1 (fixed) |
|---|---|---|---|---|---|
| 250 | 2.0M | 3.454 (31.6) | 3.363 (28.9) | 3.554 (34.9) | 3.459 (31.8) |
| 500 | 4.1M | 2.867 (17.6) | 2.744 (15.5) | 2.983 (19.7) | 2.851 (17.3) |
| 750 | 6.1M | 2.570 (13.1) | 2.454 (11.6) | 2.684 (14.6) | 2.545 (12.7) |
| 1000 | 8.2M | 2.401 (11.0) | 2.284 (9.8) | 2.516 (12.4) | 2.358 (10.6) |
| 1250 | 10.2M | 2.291 (9.9) | 2.178 (8.8) | 2.407 (11.1) | 2.241 (9.4) |
| 1500 | 12.3M | **2.235 (9.34)** | **2.120 (8.33)** | **2.345 (10.44)** | **2.181 (8.85)** |

| | dense | moe-8x-top2 | moe-4x-top1 (fixed) | moe-4x-top1 (bug) |
|---|---|---|---|---|
| training speed (clean windows) | 2,262–2,267 tok/s | 1,980–1,996 tok/s (**12% slower**) | 2,099–2,138 tok/s (6% slower) | 2,103–2,132 tok/s (6% slower) |
| wall clock, 1,500 steps | 96.4 min | 104.6 min | 97.7 min | 108.1 min |
| weights in memory (fp32) | 23 MB | 61 MB (2.6×) | 61 MB (2.6×) | 61 MB (2.6×) |
| load-balancing `aux` at the end (6.0 = perfectly even) | n/a | 6.007 | 6.022 | 6.004 |
| expert shares at the end, any layer (even = 1/E) | n/a | 0.086–0.152 (even 0.125) | 0.15–0.40 (even 0.25) | 0.22–0.28 (even 0.25) |

Steps 250 and 500 of the buggy run overlapped with llama.cpp tests on the same CPU; their speeds are not used.

### What this shows
- **The 8-expert top-2 MoE beats the dense model at equal active compute:** validation loss 0.115 lower (perplexity
  8.3 vs 9.3), ahead by about the same margin at every checkpoint. It wins at equal *time* too: after 87 minutes it
  was at 2.178, already better than the dense model's final 2.235 after 96 minutes.
- **It costs speed and memory on a CPU.** Same arithmetic per token, but grouping tokens by expert, smaller matrix
  multiplies (eight 512-wide experts instead of one 1,024-wide network) and scattering the results back cost 12% of
  throughput, and every expert's weights must be in memory: 2.6× the weights for 0.115 nats.
- **Both MoE shapes beat dense once the router learns.** Fixed top-1 ends 0.054 below dense (ppl 8.85 vs 9.34), and
  its lead grows with training (0.00 at step 250, 0.05 at step 1500). Fine-grained top-2 of 8 is better still
  (−0.115) and ahead from the first checkpoint, at twice the slowdown (12% vs 6%). Same parameters, same active
  compute: mixing two smaller experts per token beats one big one, as DeepSeek-MoE argues.
- **The load-balancing loss worked** in every run: no expert collapsed. A router that learns from the task uses its
  experts *less* evenly (0.15–0.40 with the fix, 0.22–0.28 with the bug): it specialises, and the small balancing
  term only stops it from collapsing.

### The bug the first top-1 run found
The first `moe-4x-top1` run *lost* to the dense model by 0.11, and my first explanation was "each expert only trains
on a quarter of the tokens". The 8-expert run disproved that: with top-2 of 8, each expert also sees about a quarter
of the tokens, and it won.

The real cause was in the code. Top-1 renormalised its single chosen weight, `w = p / p = 1`. That is a constant, so
the gradient of the language-model loss with respect to the router was zero; measured on a test layer it was
**4 × 10⁻⁹ (rounding noise) instead of 0.016**. The router learned only from the load-balancing loss, which asks
"spread tokens evenly" and nothing about *which* expert suits a token. So routing was balanced but blind: each expert
got a quarter of the tokens without any specialisation, which is worse than one network seeing them all. Top-2 was
not affected, because `p₁ / (p₁ + p₂)` still depends on both probabilities.

The fix follows the Switch Transformer: for top-1 the output is `p · expert(x)`, so a token's loss tells the router
whether that expert was a good choice. A regression test checks that the router gets a real gradient from the output
alone, for both top-1 and top-2 (it fails on the old code). The buggy run is kept as `models/moe-4x-top1-bug` and the
fixed one is retrained with identical settings. **Fixing that one line moved the final loss from 2.345 to 2.181:**
from 0.11 worse than dense to 0.05 better.

A side effect worth knowing: the fixed code changes what the *old* checkpoint computes (outputs are now scaled by p),
so it must be run with the gate it was trained with. On 64 validation windows it scores 2.390 with its own gate and
4.406 with the fixed one, and its samples are only coherent with its own
([`experiments/sample_bug_run.py`](../../experiments/sample_bug_run.py)).

### For an interview
"At equal active compute and 12M training tokens, my 8-expert top-2 MoE reached perplexity 8.3 against the dense
model's 9.3, 12% slower on a CPU and with 2.6× the weights in memory; 4-expert top-1 reached 8.85 at 6% slower. My
first top-1 run lost to dense, and the reason was a gating bug: renormalising a single top-1 weight makes it the
constant 1, so the router got no gradient from the task. Fixing that one line took it from 0.11 worse than dense to
0.05 better. That's why Switch multiplies by the raw gate probability."

## Interview questions
- *Why doesn't MoE reduce memory?* Every expert must be in memory, because routing is decided per token at run time.
- *Why is a load-balancing loss needed?* Routing is self-reinforcing: a favoured expert trains more and is favoured
  more. The loss rewards spreading tokens evenly.
- *Why multiply `f_e` (not differentiable) by `P_e` (differentiable)?* The gradient flows through `P_e`, and it is
  largest for the experts that already get the most tokens, so the router is pushed to lower exactly those.
- *Top-1 vs top-2?* Top-2 mixes two experts per token (smoother, more robust routing); at equal compute each expert
  must then be half the size.
