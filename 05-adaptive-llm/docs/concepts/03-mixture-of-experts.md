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
2. **Choose:** keep the `top_k` largest probabilities; renormalise them to sum to 1.
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

### `moe-4x-top1` vs the dense baseline (same tokens, same steps, same validation windows)

| step | tokens | dense val loss (ppl) | moe-4x-top1 val loss (ppl) | gap |
|---|---|---|---|---|
| 250 | 2.0M | 3.454 (31.6) | 3.554 (34.9) | +0.10 |
| 500 | 4.1M | 2.867 (17.6) | 2.983 (19.7) | +0.12 |
| 750 | 6.1M | 2.570 (13.1) | 2.684 (14.6) | +0.11 |
| 1000 | 8.2M | 2.401 (11.0) | 2.516 (12.4) | +0.12 |
| 1250 | 10.2M | 2.291 (9.9) | 2.407 (11.1) | +0.12 |
| 1500 | 12.3M | **2.235 (9.34)** | **2.345 (10.44)** | +0.11 |

| | dense | moe-4x-top1 |
|---|---|---|
| training speed (clean windows, steps 750–1500) | 2,262–2,267 tok/s | 2,103–2,132 tok/s (**6% slower**) |
| wall clock, 1,500 steps | 96.4 min | 108.1 min |
| weights in memory (fp32) | 23 MB | 61 MB (2.6×) |
| load-balancing `aux` (6 layers, 6.0 = perfectly even) | n/a | 6.06 → 6.004 |
| busiest / idlest expert share, any layer | n/a | 0.31 / 0.16 at step 250 → 0.28 / 0.22 at the end |

(Steps 250 and 500 of the MoE run overlapped with llama.cpp tests on the same CPU, so their speeds, 1,553 and
1,651 tok/s, are not used.)

**What this shows.**
- **At this budget the MoE is worse, not better:** 2.6× the parameters at the same compute per token gave a
  validation loss 0.11 *higher* (perplexity 10.4 vs 9.3). The gap is about the same from step 250 to step 1500; it
  does not close as training goes on.
- **It is also slower on a CPU.** Same arithmetic per token, but grouping tokens by expert, the smaller matrix
  multiplies and scattering results back cost 6% of throughput. The memory cost is the 2.6× in weights.
- **The load-balancing loss worked.** No expert collapsed; by the end every expert in every layer handled 22–28% of
  the tokens (25% is perfectly even).

**Why, most likely.** Both models are under-trained (the dense loss is still falling, see concept 2). In that regime
the limit is how many gradient updates each weight gets, not how much the model can store. With top-1 routing each
expert sees only about a quarter of the tokens, so each expert's weights are trained on a quarter of the data. Extra
capacity only pays once a model has learned what a smaller one can hold, which is why MoE gains in the Switch
Transformer and Mixtral papers are reported at far larger token budgets. We did not run longer to confirm this: on
this laptop that is many hours per run.

**For an interview:** "At equal active compute and a small fixed token budget, my 4-expert MoE lost to the dense
model by 0.11 nats and trained 6% slower on CPU, while load balancing kept all experts at 22–28% use. MoE trades
memory for capacity; it only wins when the model is capacity-limited, and at 12M tokens mine was data-limited."

### `moe-8x-top2`
*(training; added when it finishes)*

## Interview questions
- *Why doesn't MoE reduce memory?* Every expert must be in memory, because routing is decided per token at run time.
- *Why is a load-balancing loss needed?* Routing is self-reinforcing: a favoured expert trains more and is favoured
  more. The loss rewards spreading tokens evenly.
- *Why multiply `f_e` (not differentiable) by `P_e` (differentiable)?* The gradient flows through `P_e`, and it is
  largest for the experts that already get the most tokens, so the router is pushed to lower exactly those.
- *Top-1 vs top-2?* Top-2 mixes two experts per token (smoother, more robust routing); at equal compute each expert
  must then be half the size.
