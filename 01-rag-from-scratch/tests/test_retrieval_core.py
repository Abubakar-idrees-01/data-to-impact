"""Tokeniser, BM25, vector index and Reciprocal Rank Fusion."""

import math

import numpy as np

from studyrag.bm25 import BM25Index
from studyrag.hybrid import reciprocal_rank_fusion
from studyrag.text import stem, tokenize
from studyrag.vector_index import VectorIndex


def test_stemmer_conflates_word_families():
    assert stem("marketing") == stem("marketed") == stem("markets") == "market"
    assert stem("advertising") == stem("advertise") == stem("advertisement") == stem("advertised")
    assert stem("pages") == stem("page")
    assert stem("strategies") == stem("strategy")
    assert stem("business") == "business"          # protected "ss" ending
    assert stem("shopping") == "shop"


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("What is the brand's SEO strategy?") == ["brand", "seo", "strategy"]


def test_bm25_matches_hand_computed_score():
    corpus = [["seo", "rank", "seo"], ["email", "list"], ["seo", "email", "metric", "rate"]]
    idx = BM25Index(k1=1.5, b=0.75).build(corpus)
    n, avgdl = 3, (3 + 2 + 4) / 3
    df_seo = 2
    idf = math.log((n - df_seo + 0.5) / (df_seo + 0.5) + 1)
    tf, dl = 2, 3
    expected = idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert math.isclose(float(idx.scores(["seo"])[0]), expected, rel_tol=1e-5)
    assert idx.scores(["seo"])[1] == 0.0


def test_bm25_prefers_rare_terms_and_respects_mask():
    corpus = [["market", "seo"], ["market", "email"], ["market", "social"]]
    idx = BM25Index().build(corpus)
    hits = idx.search(["market", "email"], top_k=3)
    assert hits[0][0] == 1                                    # "email" is rare -> decides the ranking
    masked = idx.search(["email"], top_k=3, allowed=np.array([True, False, True]))
    assert masked == []                                       # the only match was filtered out


def test_vector_index_exact_cosine_topk():
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    idx = VectorIndex(vecs)
    hits = idx.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert [i for i, _ in hits] == [0, 1]
    assert math.isclose(hits[0][1], 1.0, rel_tol=1e-6)
    allowed = np.array([False, False, True, True])
    assert [i for i, _ in idx.search(np.array([1, 0, 0], dtype=np.float32), 5, allowed)] in ([2, 3], [3, 2])


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]], k=60)
    order = [i for i, _ in fused]
    assert order[0] == 1                       # rank 1 + rank 2 beats rank 3 + rank 1
    assert set(order) == {1, 2, 3, 4}
    assert math.isclose(dict(fused)[1], 1 / 61 + 1 / 62)
