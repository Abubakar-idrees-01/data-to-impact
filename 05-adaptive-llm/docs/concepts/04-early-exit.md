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
*(added when `exit-d256-L6` and the sweeps finish)*

## Interview questions
- *Why can every layer share the output matrix?* All layers write into the same residual stream, a vector space
  the output layer already knows how to read; a per-layer norm adjusts the scale.
- *Why do untrained exits work at all?* Residual connections mean each layer adds a correction to the same vector,
  so intermediate states are already rough drafts of the final one; the logit lens shows how rough.
- *What is the KV-cache problem with early exit?* Later tokens attend to the keys/values of every layer of earlier
  tokens; an exited token never computed its deeper layers.
- *MoE vs early exit?* MoE: same depth, part of the width, decided by a learned router. Early exit: full width,
  part of the depth, decided by confidence. MoE adds parameters; early exit adds almost none.
