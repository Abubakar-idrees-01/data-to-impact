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

The question is not "is the MoE better?" (it has 2.6× the parameters, so it should be), but **how much better per
unit of compute**, and what it costs in wall-clock speed and memory on a CPU.

## Results
*(added when the MoE runs finish)*

## Interview questions
- *Why doesn't MoE reduce memory?* Every expert must be in memory, because routing is decided per token at run time.
- *Why is a load-balancing loss needed?* Routing is self-reinforcing: a favoured expert trains more and is favoured
  more. The loss rewards spreading tokens evenly.
- *Why multiply `f_e` (not differentiable) by `P_e` (differentiable)?* The gradient flows through `P_e`, and it is
  largest for the experts that already get the most tokens, so the router is pushed to lower exactly those.
- *Top-1 vs top-2?* Top-2 mixes two experts per token (smoother, more robust routing); at equal compute each expert
  must then be half the size.
