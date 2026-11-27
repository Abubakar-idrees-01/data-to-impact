# 1. Tokenization: byte-pair encoding (BPE)

## Intuition
A neural network does arithmetic on numbers, so text must become a list of integers first. The question is what
each integer stands for:

| unit | vocabulary | a 100-character sentence becomes | problem |
|---|---|---|---|
| characters / bytes | 256 | ~100 tokens | sequences are long: the model must learn spelling, and attention cost grows with length |
| whole words | 100,000+ | ~20 tokens | huge embedding table; any new word ("Zentrite", a typo) is unknown |
| **subwords (BPE)** | a few thousand | ~25 tokens | none of the above: common words are one token, rare ones a few pieces |

BPE learns the subwords from data: *whatever appears together often becomes one token*.

## The algorithm
**Training** (once, on a sample of the text):
1. Split the text into pre-tokens: words with their leading space (`" happy"`), numbers, punctuation runs. Merges
   never cross these boundaries.
2. Write every pre-token as its UTF-8 bytes. The vocabulary starts as the 256 possible bytes, so any text at all can
   be encoded.
3. Count every pair of neighbouring tokens, weighted by how often each word occurs.
4. Merge the most frequent pair into a new token (`" t" + "he"` → `" the"`) and record the merge.
5. Repeat until the vocabulary has the size you chose.

**Encoding** a new text: split it into pre-tokens, write each as bytes, then apply the learned merges *in the order
they were learned* (always merge the earliest-learned pair present, until none applies). **Decoding**: concatenate each
token's bytes and read them as UTF-8.

**Special tokens** such as `<|endoftext|>` are added on top and never produced by merges: the model learns that this
id means "the story is over, the next one is unrelated".

## Making training fast
The naive version recounts every pair in the corpus after every merge: 3,839 merges × the whole corpus. Ours keeps,
for each pair, its count and the set of words containing it, so a merge only rewrites the words that contain that
pair and adjusts their counts. A heap gives the most frequent pair; entries whose count has changed since they were
pushed are skipped when popped. Encoding caches each pre-token's result: a story corpus repeats the same few thousand
words millions of times.

A test trains both versions on the same text and checks they learn **exactly the same merges**.

## Our code
- [`src/tinygpt/tokenizer.py`](../../src/tinygpt/tokenizer.py): `BPETokenizer.train / encode / decode / save / load`
- [`src/tinygpt/data.py`](../../src/tinygpt/data.py): trains on the first 20 MB of stories, encodes everything into
  `uint16` files the training loop memory-maps
- [`tests/test_tokenizer.py`](../../tests/test_tokenizer.py): round trip on any text (emoji, Chinese, tabs), the naive
  comparison, special tokens

## Results (TinyStories, vocabulary 4,096)
| | |
|---|---|
| tokenizer training | 24,834 stories (20 MB), 3,839 merges, **4 seconds** |
| encoding | 256,006 train + 27,629 validation stories → **52.0M + 5.6M tokens** in 47 s |
| compression | **3.95 characters per token** |

What it learned:
```
first merges:  ' t'  'he'  ' a'  ' s'  ' w'  ' the'  'nd'  'ed'  ' b'  ' to'  ' and' ...
last merges:   ' sunglasses'  ' basketball'  ' gorilla'  ' whenever'  ' realised' ...

"Once upon a time, Lily found a shiny red ball in the garden. She was very happy!"
  → Once| upon| a| time|,| Lily| found| a| shiny| red| ball| in| the| garden|.| She| was| very| happy|!   (80 chars, 20 tokens)

"The quantum chromodynamics lecture was postponed."
  → The| qu|ant|um| ch|rom|o|dy|nam|ic|s| l|ec|ture| was| post|p|on|ed|.
```
The vocabulary mirrors its data: every word of a children's story is one token, while physics vocabulary falls apart
into pieces. A model trained on these tokens can still *spell* "quantum" (bytes are always available) but has never
seen it as a unit.

**Why 4,096?** The model's first and last layers have one row per token: at 256 dimensions, 4,096 tokens cost 1M
parameters, a sensible share of a 10–30M model. GPT-2's 50,257 would cost 12.9M, most of the model, for words that
never appear in TinyStories.

## Interview questions
- *Why not characters or words?* Characters make sequences ~4× longer; words make the vocabulary huge and can't handle
  unseen words. Subwords balance the two.
- *Why bytes as the base?* Every string is a sequence of bytes, so there is never an unknown token.
- *Why does the merge order matter when encoding?* Merges were learned on top of each other (`" the"` needs `"he"`
  first). Applying them in a different order gives different tokens than the model saw in training.
- *What does tokenization have to do with model quality?* The model can only predict tokens it has; a tokenizer
  trained on other text splits your domain's words into many pieces, making sequences longer and patterns harder.
