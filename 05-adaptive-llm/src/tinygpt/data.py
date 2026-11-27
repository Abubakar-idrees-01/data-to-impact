"""Turn TinyStories into training data: train the tokenizer, encode every story, save token ids to disk.

    python -m tinygpt.data --vocab 4096 --sample-mb 20

Writes (all gitignored):
    models/tokenizer.json        the learned merges
    data/tokens/train.bin        uint16 token ids, stories separated by <|endoftext|>
    data/tokens/valid.bin
    data/tokens/meta.json        counts, so the numbers in the docs can be checked

uint16 fits any vocabulary under 65,536 and halves the file size; training reads the file with numpy.memmap, so the
whole dataset never has to be in memory.
"""

from __future__ import annotations

import argparse
import json
import time
from array import array
from pathlib import Path

from .tokenizer import EOT, BPETokenizer

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TOKENS = ROOT / "data" / "tokens"
TOKENIZER = ROOT / "models" / "tokenizer.json"


def stories(path: Path) -> list[str]:
    """Stories in a file; the last one is dropped if the file was cut mid-story (our partial download)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = [s.strip() for s in text.split(EOT)]
    if not text.rstrip().endswith(EOT):
        parts = parts[:-1]
    return [s for s in parts if s]


def encode_file(tok: BPETokenizer, items: list[str], out: Path) -> int:
    buf = array("H")
    t0 = time.time()
    for i, s in enumerate(items, 1):
        buf.extend(tok.encode(s))
        buf.append(tok.eot)
        if i % 50_000 == 0:
            print(f"  {out.name}: {i:,}/{len(items):,} stories, {len(buf):,} tokens, {time.time() - t0:.0f}s", flush=True)
    with out.open("wb") as f:
        buf.tofile(f)
    return len(buf)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--sample-mb", type=float, default=20, help="how much training text the tokenizer learns from")
    a = ap.parse_args()
    TOKENS.mkdir(parents=True, exist_ok=True)
    TOKENIZER.parent.mkdir(parents=True, exist_ok=True)

    train, valid = stories(RAW / "train.txt"), stories(RAW / "valid.txt")
    print(f"{len(train):,} train stories, {len(valid):,} valid stories")
    sample, size = [], 0
    for s in train:                                     # the first stories, up to sample-mb of text
        sample.append(s)
        size += len(s)
        if size > a.sample_mb * 1e6:
            break
    t0 = time.time()
    tok = BPETokenizer.train(EOT.join(sample), a.vocab, verbose=True)
    train_s = time.time() - t0
    tok.save(TOKENIZER)
    print(f"tokenizer: {tok.vocab_size} tokens, trained on {len(sample):,} stories in {train_s:.0f}s")

    t0 = time.time()
    n_train = encode_file(tok, train, TOKENS / "train.bin")
    n_valid = encode_file(tok, valid, TOKENS / "valid.bin")
    chars = sum(map(len, train)) + sum(map(len, valid))
    meta = {"vocab_size": tok.vocab_size, "eot": tok.eot, "train_stories": len(train), "valid_stories": len(valid),
            "train_tokens": n_train, "valid_tokens": n_valid, "chars_per_token": round(chars / (n_train + n_valid), 3),
            "tokenizer_sample_stories": len(sample), "tokenizer_train_seconds": round(train_s, 1),
            "encode_seconds": round(time.time() - t0, 1)}
    (TOKENS / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
