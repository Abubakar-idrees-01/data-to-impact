"""Byte-level BPE (byte-pair encoding), written from scratch.

A model reads numbers, not text. BPE decides which numbers:
1. Start from the 256 possible bytes, so *any* text (emoji, Chinese, typos) can be encoded: no "unknown" token.
2. Count every pair of neighbouring tokens in a training text; merge the most frequent pair into a new token.
3. Repeat until the vocabulary has the size you want. Frequent words end up as one token ("Once", " upon"),
   rare ones as a few pieces.

Text is first split into pre-tokens (words with their leading space, numbers, punctuation runs), and merges never
cross a pre-token boundary, so " cat" and "cat." don't produce tokens like "t." that glue unrelated pieces.

Training keeps, for every pair, how often it occurs and in which words, so each merge only touches the words that
contain the merged pair (instead of rescanning the whole corpus 4,000 times).

    tok = BPETokenizer.train(text, vocab_size=4096)
    ids = tok.encode("Once upon a time")
    tok.decode(ids) == "Once upon a time"
"""

from __future__ import annotations

import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# GPT-2's split, restricted to what Python's built-in `re` supports: contractions, words, numbers and punctuation
# runs (each with at most one leading space), then whitespace.
PRETOKEN = re.compile(r"'(?:s|t|re|ve|m|ll|d)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")
EOT = "<|endoftext|>"


def pretokens(text: str) -> list[str]:
    return PRETOKEN.findall(text)


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]], specials: list[str] | None = None):
        """merges[i] = (a, b) creates token 256 + i. Special tokens come after the merges."""
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: list[bytes] = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.specials = {s: len(self.vocab) + i for i, s in enumerate(specials or [EOT])}
        self.special_re = re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")")
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.specials)

    @property
    def eot(self) -> int:
        return self.specials[EOT]

    # ------------------------------------------------------------------ training
    @classmethod
    def train(cls, text: str, vocab_size: int, specials: list[str] | None = None, verbose: bool = False) -> "BPETokenizer":
        specials = specials or [EOT]
        n_merges = vocab_size - 256 - len(specials)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(specials)}")
        # count pre-tokens once; train on unique words weighted by their frequency
        counts: Counter[str] = Counter()
        for part in re.split("|".join(re.escape(s) for s in specials), text):
            counts.update(pretokens(part))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())

        pair_count: Counter[tuple[int, int]] = Counter()
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, w in enumerate(words):
            for p in zip(w, w[1:]):
                pair_count[p] += freqs[wi]
                where[p].add(wi)
        # max-heap of (-count, pair); stale entries are skipped when popped
        heap = [(-c, p) for p, c in pair_count.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg, pair = heapq.heappop(heap)
            if pair_count.get(pair, 0) != -neg or -neg <= 0:
                continue                                   # stale: the count changed since it was pushed
            new = 256 + len(merges)
            merges.append(pair)
            changed: set[tuple[int, int]] = set()
            for wi in list(where.pop(pair, ())):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):                    # remove this word's old pairs
                    pair_count[p] -= f
                    changed.add(p)
                merged, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == pair:
                        merged.append(new)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for p in zip(merged, merged[1:]):          # add its new pairs
                    pair_count[p] += f
                    where[p].add(wi)
                    changed.add(p)
            for p in changed:
                if pair_count.get(p, 0) > 0:
                    heapq.heappush(heap, (-pair_count[p], p))
                else:
                    pair_count.pop(p, None)
            pair_count.pop(pair, None)
            if verbose and len(merges) % 500 == 0:
                print(f"  {len(merges)}/{n_merges} merges", flush=True)
        return cls(merges, specials)

    # ------------------------------------------------------------------ encode / decode
    def _bpe(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = list(word.encode("utf-8"))
        while len(ids) > 1:
            # merge the pair that was learned earliest (lowest rank), exactly as in training
            best = min(((self.ranks.get(p, 1 << 30), i) for i, p in enumerate(zip(ids, ids[1:]))))
            if best[0] == 1 << 30:
                break
            i = best[1]
            ids[i:i + 2] = [256 + best[0]]
        if len(self._cache) < 500_000:
            self._cache[word] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for part in self.special_re.split(text):
            if part in self.specials:
                out.append(self.specials[part])
            elif part:
                for w in pretokens(part):
                    out.extend(self._bpe(w))
        return out

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.specials.items()}
        parts, buf = [], b""
        for i in ids:
            if i in inv:
                parts.append(buf.decode("utf-8", errors="replace") + inv[i])
                buf = b""
            else:
                buf += self.vocab[i]
        return "".join(parts) + buf.decode("utf-8", errors="replace")

    def token_str(self, i: int) -> str:
        """A readable form of one token (for inspecting the vocabulary)."""
        inv = {v: k for k, v in self.specials.items()}
        return inv[i] if i in inv else self.vocab[i].decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ save / load
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"type": "byte-bpe", "merges": self.merges,
                                          "specials": list(self.specials)}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in d["merges"]], d["specials"])
