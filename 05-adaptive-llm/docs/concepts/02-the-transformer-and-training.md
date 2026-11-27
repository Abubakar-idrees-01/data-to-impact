# 2. A GPT from scratch: the transformer, and training it on a CPU

## Intuition
A language model is a machine for one question: *given these tokens, what comes next?* Ask it repeatedly, feeding
each answer back in, and it writes. Everything else (attention, layers, training) is how it gets good at that one
question.

## The architecture ([`model.py`](../../src/tinygpt/model.py))
```
tokens ─► token embedding + position embedding ─► [block] × 6 ─► norm ─► output layer ─► a score per vocabulary token
block:  x = x + attention(norm(x))       mix information *between* tokens (each looks only at earlier ones)
        x = x + feedforward(norm(x))     transform each token's vector on its own
```
- **Embeddings.** Each token id becomes a learned vector of `d` = 256 numbers; a second learned vector per position
  says *where* it is.
- **Attention.** Each token makes a query `q`, a key `k` and a value `v`. Its output is a weighted average of the
  values of earlier tokens, weighted by `softmax(q·k / √d_head)`: it "looks up" the tokens relevant to it. 8 heads do
  this in parallel on 32-dimensional slices, so one can track the subject while another tracks the last word.
  **Causal mask:** token *t* may only look at tokens ≤ *t*, or training would let it read the answer. A test changes
  later tokens and checks that no earlier prediction moves.
- **Feed-forward.** Per token: expand to 4 × 256 = 1,024 numbers, GELU, project back. Most of the parameters, and
  where the model stores "facts" (the slot that Mixture-of-Experts replaces in [concept 3](03-mixture-of-experts.md)).
- **Residual connections** (`x = x + ...`) let each layer add a correction instead of rewriting everything, and give
  gradients a direct path back through 6 layers. **Pre-norm** (LayerNorm before each sub-layer) keeps training stable.
- **Weight tying.** The output layer reuses the embedding matrix (scores = vector · each token's embedding): saves
  1M of our 5.8M parameters and usually helps small models.

## Training ([`train.py`](../../src/tinygpt/train.py))
1. Take 32 random windows of 257 tokens from the 52M training tokens (memory-mapped, never loaded whole).
2. The model predicts token *t+1* at every position *t*: one window gives 256 training examples.
3. **Loss** = cross-entropy = −log(probability given to the true next token), averaged. Random guessing over 4,096
   tokens gives ln 4096 = 8.3; **perplexity** = e^loss reads as "as unsure as choosing among N tokens".
4. Backpropagation computes how each of the 5.8M parameters should change; **AdamW** applies the change, with weight
   decay pulling unused weights towards zero.
5. **Learning rate:** linear warm-up over 100 steps (large early updates on random weights are destructive), then
   cosine decay to 10%.
6. Every 250 steps: loss on the *same* 256 validation windows, so curves are comparable between runs, and a
   checkpoint (model, optimizer, step, random-number state) written atomically. A test checks that stopping and
   resuming gives bit-for-bit the same model as training straight through.

## Sizing a model to a 2015 laptop
Measured before choosing ([`bench.py`](../../src/tinygpt/bench.py), full training steps, batch 32 × 256 tokens):

| model | parameters | 2 threads | 4 threads | hours for 20M tokens (2 threads) |
|---|---|---|---|---|
| d128, 4 layers | 1.35M | 4,909 tok/s | 4,350 | 1.1 |
| d192, 6 layers | 3.49M | 2,333 | 1,994 | 2.4 |
| **d256, 6 layers** | **5.84M** | **1,594** | 1,336 | 3.5 |
| d256, 8 layers | 7.41M | 1,243 | 1,045 | 4.5 |
| d384, 6 layers | 12.30M | 907 | 775 | 6.1 |

- **2 threads beat 4 at every size** (by 12–20%): the CPU has 2 physical cores; the other 2 "threads" are
  hyper-threads sharing the same arithmetic units, and only add contention. Project 1 found the same with llama.cpp.
- Part A needs four comparable runs, so each must fit in about two hours: **d256 × 6 layers, 1,500 steps × 8,192
  tokens = 12.3M tokens** (~2.1 h). Small and under-trained by research standards; the same budget for every
  variant makes the comparisons fair.

## Results
*(added when the baseline finishes)*

## Interview questions
- *Why does attention need a mask?* Training predicts every position at once; without the mask, position *t* could
  look at token *t+1*, the answer.
- *Why divide by √d_head?* Dot products of long vectors grow with their length; without scaling, softmax saturates and
  gradients vanish.
- *What is perplexity?* e^loss: the effective number of tokens the model is choosing between. 1 = certain; 4,096 =
  random over our vocabulary.
- *Why are more threads slower here?* Hyper-threads share a core's arithmetic units; for dense matrix maths they
  compete instead of helping.
