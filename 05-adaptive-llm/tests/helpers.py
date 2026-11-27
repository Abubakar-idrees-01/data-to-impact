"""Test helpers: a tiny GGUF file written by hand in the format the reader expects."""

import struct


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, kv: dict, tensors: list[tuple[str, list[int]]]) -> None:
    body = b""
    for k, v in kv.items():
        body += _s(k)
        if isinstance(v, str):
            body += struct.pack("<I", 8) + _s(v)
        elif isinstance(v, list) and isinstance(v[0], str):
            body += struct.pack("<IIQ", 9, 8, len(v)) + b"".join(_s(x) for x in v)
        elif isinstance(v, list):
            body += struct.pack("<IIQ", 9, 5, len(v)) + b"".join(struct.pack("<i", x) for x in v)
        else:
            body += struct.pack("<I", 4) + struct.pack("<I", v)
    for name, shape in tensors:
        body += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        body += struct.pack("<IQ", 12, 0)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)) + body + b"\0" * 1000)
