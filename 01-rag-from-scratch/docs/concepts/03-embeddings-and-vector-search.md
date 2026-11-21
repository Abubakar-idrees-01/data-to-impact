# 3. Embeddings and vector search: searching by meaning

## Intuition
An **embedding model** turns text into a list of numbers (a vector) such that texts with similar
*meaning* get vectors pointing in similar *directions*. "How do I get more website visitors?" and
"techniques to increase site traffic" share almost no words, but their vectors are close.

We use `BAAI/bge-small-en-v1.5`: 33M parameters, 384-dimension vectors, run on CPU through ONNX Runtime
(the fastembed library), with no PyTorch needed.

## Cosine similarity
Similarity of two vectors is the cosine of the angle between them:

```
cos(q, d) = (q · d) / (|q| · |d|)        1 = same direction, 0 = unrelated
```

**Trick:** normalise every vector to length 1 once, at indexing time. Then `|q| = |d| = 1` and cosine is
just the dot product. Searching all chunks becomes **one matrix-vector multiplication**:

```python
scores = matrix @ query_vec          # (n_chunks × 384) · (384,) → (n_chunks,)
top = argpartition(-scores, k)       # O(n) selection of the k best, then sort only those k
```
That's all of [`vector_index.py`](../../src/studyrag/vector_index.py).

## Why brute force and not FAISS/HNSW?
Approximate nearest-neighbour indexes (HNSW, IVF) trade a little accuracy for speed when you have
**millions** of vectors. A student's library has hundreds to a few thousand chunks:

| Chunks | Brute-force search time on our laptop CPU (measured, see RESULTS.md) |
|---|---|
| 1,000 | well under 1 ms |
| 100,000 | a few ms |

At this scale exact search is fast, simple, and has 100% recall. Choosing the simplest correct tool is
an engineering decision worth explaining in interviews.

## Query instruction
BGE models were trained with an instruction prefix on the **query** side for retrieval:
`"Represent this sentence for searching relevant passages: "`. We add it to queries, not to chunks
([`embeddings.py`](../../src/studyrag/embeddings.py)), because questions and passages are different kinds of text.

## Bi-encoder vs cross-encoder
The embedder is a **bi-encoder**: it encodes the query and each chunk *separately*, so chunk vectors can
be computed once and stored. That is what makes it fast, and also what limits it: the model never sees the
query and chunk together. The reranker fixes that (→ [reranking](05-reranking.md)).

## Interview questions
- *Why normalise vectors?* Cosine becomes a dot product, so the search is one matrix multiply.
- *Dimension trade-off?* More dimensions capture more nuance but cost memory and time. 384 is plenty here.
- *When would you need an ANN index?* Above roughly 1M vectors, or under strict latency budgets.
- *Weakness of embeddings?* Exact identifiers and rare terms (BM25's strength), and domain jargon the model never saw.
