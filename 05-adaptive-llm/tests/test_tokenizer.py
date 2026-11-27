"""The tokenizer must round-trip any text, merge frequent pairs first, and train identically to a naive version."""

from collections import Counter

import pytest

from tinygpt.tokenizer import EOT, BPETokenizer, pretokens

TEXT = ("Once upon a time there was a little cat. The cat was happy. " * 20 +
        f"{EOT}Tim saw a big dog. The dog was big and happy! {EOT}" * 10)


def naive_merges(text, n):
    """Reference implementation: recount every pair after every merge (slow but obviously right)."""
    counts = Counter()
    for part in text.split(EOT):
        counts.update(pretokens(part))
    words = {tuple(w.encode()): c for w, c in counts.items()}
    merges = []
    for k in range(n):
        pairs = Counter()
        for w, c in words.items():
            for p in zip(w, w[1:]):
                pairs[p] += c
        if not pairs:
            break
        top = max(pairs.values())
        pair = min(p for p, c in pairs.items() if c == top)     # ties: smallest pair, like the heap
        merges.append(pair)
        new = {}
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                    out.append(256 + k)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + c
        words = new
    return merges


def test_training_matches_the_naive_algorithm():
    tok = BPETokenizer.train(TEXT, vocab_size=256 + 60 + 1)
    assert tok.merges == naive_merges(TEXT, 60)


def test_round_trip_any_text():
    tok = BPETokenizer.train(TEXT, vocab_size=300)
    for s in ["Once upon a time", "naïve café — “quotes” 🌴 一了", "  spaces\n\nand\ttabs ", f"a{EOT}b", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_frequent_words_become_single_tokens_and_eot_is_special():
    tok = BPETokenizer.train(TEXT, vocab_size=400)
    assert len(tok.encode(" cat")) == 1 and len(tok.encode(" happy")) == 1
    ids = tok.encode(f"The cat.{EOT}Tim")
    assert ids.count(tok.eot) == 1 and tok.eot == tok.vocab_size - 1
    assert len(tok.encode("<|endoftext")) > 1           # only the exact marker is special


def test_save_and_load(tmp_path):
    tok = BPETokenizer.train(TEXT, vocab_size=320)
    tok.save(tmp_path / "tok.json")
    tok2 = BPETokenizer.load(tmp_path / "tok.json")
    assert tok2.encode(TEXT) == tok.encode(TEXT) and tok2.vocab_size == tok.vocab_size


def test_vocab_too_small():
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, vocab_size=200)
