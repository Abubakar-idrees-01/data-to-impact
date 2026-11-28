# 4. Early exit: stop thinking when the answer is obvious

## Intuition
After "Once upon a" the next word is "time": a model doesn't need all 6 layers to know that. After "Tim gave the
ball to" it might. **Early exit** lets each token leave the network at the first layer where the model is already
confident, so easy tokens cost 1–2 layers and only hard ones pay for all 6.

Mixture-of-Experts ([concept 3](03-mixture-of-experts.md)) saves compute *across* the width of a layer; early exit
saves it *across* depth. Both run only part of the model per token.

## How an exit works ([`model.py`](../../src/tinygpt/model.py))
Any layer's output is a vector of the same size as the final one, so the same output layer can turn it into
next-token scores. Each layer *i* gets its own small LayerNorm, and all exits share the (tied) output matrix:
```
x₁ = block₁(embeddings)      exit₁ = head(norm₁(x₁))      "after 1 layer, I'd say …"
x₂ = block₂(x₁)              exit₂ = head(norm₂(x₂))
…
x₆ = block₆(x₅)              final = head(norm(x₆))        the normal output
```
Each exit adds only 512 parameters (a norm's scale and shift), so the 6-layer model grows from 5.84M to 5.84M + 2.6k.

**Training the exits.** A normal model never asks layer 2 for an answer, so its intermediate states aren't shaped
to be read out. With `--exit-weight 1.0` the loss becomes
`loss = CE(final) + 1.0 × mean(CE(exit₁) … CE(exit₅))`, so every layer learns to produce a usable prediction. The
price: early layers now serve two masters (a prediction of their own, and good input for the next layer), which
can cost the final output a little quality. Training is also slower: each exit is an extra 256 × 4,096 projection
and softmax for every token, nearly doubling the compute of a model this small.

**Without training** we can still read intermediate layers with the final norm (the "logit lens",
nostalgebraist 2020). The dense baseline is evaluated that way, as the "untrained exits" comparison.

## The exit rule ([`early_exit.py`](../../src/tinygpt/early_exit.py))
For a confidence threshold τ: run layers one by one; after each, if the exit's top probability is ≥ τ, stop and
emit that prediction. The last layer always answers. Sweeping τ traces a curve:

- τ = 1.01: never exits early = the full model (6 layers, full quality);
- lower τ: more tokens leave early, fewer average layers, some quality lost.

Reported per τ on the same 256 validation windows (65k positions): loss and perplexity at the chosen exits, average
layers per token, share of compute saved, and how often the early answer equals the full model's.

**Caveat, stated plainly:** the evaluation is teacher-forced: every layer still has the deep states of *earlier*
tokens to attend to. In real generation with a KV cache, a token that exited at layer 2 has no layer-3..6 keys and
values for later tokens to attend to. Methods like CALM (Schuster et al., 2022) copy the exit state upward to fill
them. So the "compute saved" here is what the exit decisions *allow*, not a measured wall-clock speed-up.

## Results
Same budget as every Part A run (1,500 steps, 12.3M tokens). Evaluated on 256 validation windows = 65,536 positions
([`results/early_exit/`](../../results/early_exit/)).

### 1. What training the exits costs
| | dense (no exit training) | exit-d256-L6 (`--exit-weight 1.0`) |
|---|---|---|
| final-layer val loss (ppl) | **2.235 (9.34)** | 2.281 (9.79) |
| training speed | 2,265 tok/s | 1,240 tok/s (**1.8× slower**) |
| wall clock | 96 min | 168 min |

The concept's prediction held: early layers that must also predict cost the full model 0.05 nats, and five extra
256 × 4,096 projections nearly double the training compute.

### 2. Every layer as an exit
| layer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| dense, logit lens (untrained): ppl | 130.0 | 79.9 | 60.6 | 31.3 | 15.7 | 9.34 |
| dense, agrees with its final top token | 24% | 30% | 37% | 50% | 68% | 100% |
| **trained exits: ppl** | **13.8** | 11.7 | 10.7 | 10.1 | 9.9 | 9.79 |
| trained exits: agree with the final top token | 69% | 78% | 85% | 90% | 94% | 100% |

Exit training turns layer 1 from useless (ppl 130) into a usable model (ppl 13.8, the dense model's quality at about
step 900). The model now contains six models of increasing depth that share their weights.

### 3. The confidence rule
| τ | ppl | avg layers | layer-only "saved" | exits at layer 1 / 2–5 / 6 | **real compute vs full model** |
|---|---|---|---|---|---|
| 0.3 | 11.78 | 2.54 | 58% | 58% / 17% / 25% | 0.78× |
| 0.4 | 11.08 | 3.25 | 46% | 43% / 18% / 38% | 1.00× |
| 0.5 | 10.59 | 3.87 | 36% | 32% / 16% / 51% | 1.19× |
| 0.6 | 10.28 | 4.37 | 27% | 23% / 15% / 62% | 1.35× |
| 0.8 | 9.91 | 5.15 | 14% | 12% / 9% / 80% | 1.59× |
| 0.95 | 9.81 | 5.69 | 5% | 4% / 4% / 92% | 1.76× |
| never | 9.79 | 6.00 | 0% | 0 / 0 / 100% | 1.85× |

The dense model with logit-lens exits is far worse: at τ = 0.9 it saves 8% of layers at ppl 11.1.

**The column that matters is the last one.** The sweep's "saved" counts layers only, and so does most writing on
early exit. But deciding whether to exit means computing that exit's confidence. Here that is the full projection onto
4,096 words: 256 × 4,096 = **1.05M multiply-adds, more than a whole layer** (12 × 256² weights + attention ≈ 0.85M).
Counting both, a token that leaves at layer *i* costs *i* × (layer + check) against the full model's 6 layers + 1 head
= 6.16M. Every threshold above 0.4 then costs **more** than not exiting at all.

**What to take from this**
- **A fixed shallow exit beats the dynamic rule here.** Always answering from layer 4 costs 0.72× the full model at
  ppl 10.1; the dynamic rule at τ = 0.6 gives ppl 10.3 at 1.35×. One trained model offers several sizes (6 layers for
  quality, 3–4 for speed), and choosing the size per *machine* or per *request* is the idea Part B builds on.
- **Dynamic exits need a cheap confidence check.** That is why CALM trains a tiny classifier on the hidden state
  instead of reading the full softmax. It also helps to check only where exits happen: here almost all early exits
  are at layer 1, so one check after layer 1 would capture most of the saving.
- **The balance changes with scale.** The check costs `d × V` and a layer about `12 d²`, so the check is relatively
  cheaper in wide models: at d = 4,096 and V = 32k it is about 0.7 of a layer instead of 1.2. It is still not free.
- The caveat above stands: these are teacher-forced decisions, not a measured generation speed-up.

**For an interview:** "Training an exit after every layer took my 6-layer model's first layer from perplexity 130 to
13.8, for 0.05 nats at the final layer and 1.8× the training time. At τ = 0.6 the rule skips 27% of layers, but my
vocabulary projection costs more than a layer, so checking confidence after every layer made it 1.35× *more*
expensive. A fixed exit at layer 4 was the better trade: 0.72× the compute at perplexity 10.1."

## Interview questions
- *Why can every layer share the output matrix?* All layers write into the same residual stream, a vector space
  the output layer already knows how to read; a per-layer norm adjusts the scale.
- *Why do untrained exits work at all?* Residual connections mean each layer adds a correction to the same vector,
  so intermediate states are already rough drafts of the final one; the logit lens shows how rough.
- *What is the KV-cache problem with early exit?* Later tokens attend to the keys/values of every layer of earlier
  tokens; an exited token never computed its deeper layers.
- *MoE vs early exit?* MoE: same depth, part of the width, decided by a learned router. Early exit: full width,
  part of the depth, decided by confidence. MoE adds parameters; early exit adds almost none.
