"""Read the metadata of a GGUF model file (llama.cpp's format) with only the standard library.

A GGUF file starts with a header and a list of key-value pairs, then tensor descriptions, then the weights:

    magic "GGUF" | version u32 | tensor_count u64 | kv_count u64 | kv_count × (key string, value type u32, value)

Strings are a u64 length + UTF-8 bytes; arrays are an element type u32 + a u64 count + the elements. We need only
a few keys (architecture, layer count, attention heads, context length, quantization) to estimate how much memory a
model needs, so large arrays (the tokenizer's 150k-word vocabulary) are skipped without being kept.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
STRING, ARRAY = 8, 9

# llama.cpp file types (general.file_type) -> name, for the quantizations we meet
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
              12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
              32: "BF16"}


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF file")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", errors="replace")


def _value(f: BinaryIO, vtype: int, keep_arrays: bool):
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == STRING:
        return _string(f)
    if vtype == ARRAY:
        etype, n = _read(f, "<I"), _read(f, "<Q")
        if etype in _SCALARS and not keep_arrays:          # skip fixed-size elements in one seek
            f.seek(n * struct.calcsize(_SCALARS[etype]), 1)
            return f"<array of {n}>"
        items = [_value(f, etype, keep_arrays) for _ in range(n)]
        return items if keep_arrays or n <= 16 else f"<array of {n}>"
    raise ValueError(f"unknown GGUF value type {vtype}")


def read_metadata(path: str | Path, keep_arrays: bool = False, tensors: bool = False) -> dict:
    """All key-value metadata of a GGUF file (big arrays replaced by a '<array of N>' placeholder).

    With tensors=True, also reads the tensor table that follows (name, shape, type per tensor) and adds
    'gguf.n_params', the total number of weights.
    """
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")
        meta = {"gguf.version": version, "gguf.tensor_count": n_tensors}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"), keep_arrays)
        if tensors:
            total = 0
            for _ in range(n_tensors):
                _string(f)                                       # name
                shape = [_read(f, "<Q") for _ in range(_read(f, "<I"))]
                _read(f, "<I"), _read(f, "<Q")                   # type, offset
                n = 1
                for s in shape:
                    n *= s
                total += n
            meta["gguf.n_params"] = total
    return meta


def model_info(path: str | Path) -> dict:
    """The facts the planner needs, from the metadata (keys are prefixed with the architecture name)."""
    path = Path(path)
    m = read_metadata(path, tensors=True)
    arch = m.get("general.architecture", "?")
    g = lambda key, default=None: m.get(f"{arch}.{key}", default)   # noqa: E731
    n_layer, n_head, d = g("block_count"), g("attention.head_count"), g("embedding_length")
    n_head_kv = g("attention.head_count_kv", n_head)
    if isinstance(n_head_kv, list):                    # some architectures store one value per layer
        n_head_kv = max(n_head_kv)
    head_dim = g("attention.key_length") or (d // n_head if d and n_head else None)
    # hybrid models (linear-attention layers + a full-attention layer every k-th) keep a KV cache only for the latter
    interval = g("full_attention_interval")
    kv_layers = n_layer // interval if interval else n_layer
    return {"path": str(path), "file": path.name, "bytes": path.stat().st_size, "arch": arch,
            "name": m.get("general.name", path.stem), "quant": FILE_TYPES.get(m.get("general.file_type"), "?"),
            "n_layer": n_layer, "kv_layers": kv_layers, "n_head": n_head, "n_head_kv": n_head_kv,
            "head_dim": head_dim, "d": d, "ctx_train": g("context_length"),
            "params": m["gguf.n_params"], "bits_per_weight": round(path.stat().st_size * 8 / m["gguf.n_params"], 2),
            "size_label": m.get("general.size_label")}


def kv_cache_bytes(info: dict, n_ctx: int, bytes_per_value: int = 2) -> int:
    """Memory of the attention cache: keys + values, for every attention layer and every position (f16 by default)."""
    return 2 * info["kv_layers"] * info["n_head_kv"] * info["head_dim"] * n_ctx * bytes_per_value
