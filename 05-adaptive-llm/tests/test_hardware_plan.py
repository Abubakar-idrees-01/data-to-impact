"""B1: the profiler parses each OS's facts correctly, the GGUF reader matches what we wrote, the planner's rules hold."""

import struct

import pytest

from adaptive.gguf import kv_cache_bytes, model_info, read_metadata
from adaptive.hardware import GB, Hardware, parse_cpuinfo, parse_meminfo, slug
from adaptive.plan import gpu_layers, plan_local, quant_grid

INTEL = "\n\n".join(f"""processor\t: {i}
model name\t: Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz
physical id\t: 0
core id\t\t: {i // 2}
flags\t\t: fpu sse sse2 avx avx2 fma f16c""" for i in range(4))          # 2 cores, each with 2 hyper-threads

ARM = "\n\n".join(f"""processor\t: {i}
BogoMIPS\t: 108.00
Features\t: fp asimd evtstrm aes crc32 asimddp""" for i in range(4)) + "\n\nHardware\t: BCM2835\nModel\t: Raspberry Pi 5"


def test_cpuinfo_counts_physical_cores_not_threads():
    name, phys, logical, flags = parse_cpuinfo(INTEL)
    assert name.startswith("Intel") and (phys, logical) == (2, 4)
    assert {"avx2", "fma", "f16c"} <= flags and "avx512f" not in flags


def test_cpuinfo_on_arm_maps_asimd_to_neon():
    name, phys, logical, flags = parse_cpuinfo(ARM)
    assert (phys, logical) == (4, 4)                  # no core ids: one core per processor; board block ignored
    assert name == "Raspberry Pi 5"
    assert {"neon", "dotprod"} <= flags


def test_meminfo_uses_available_not_free():
    total, avail = parse_meminfo("MemTotal:       16000000 kB\nMemFree:          500000 kB\nMemAvailable:    8000000 kB\n")
    assert total == pytest.approx(16000000 * 1024 / GB) and avail == pytest.approx(8000000 * 1024 / GB)


def test_slug():
    assert slug("Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz") == "intel-core-i5-5200u"
    assert slug("AMD Ryzen 7 5800H with Radeon Graphics") == "amd-ryzen-7-5800h-with-radeon-graphics"


# ---------- a tiny GGUF file, written by hand in the format the reader expects
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


def test_gguf_reader_and_model_info(tmp_path):
    p = tmp_path / "tiny-Q4_K_M.gguf"
    vocab = [f"tok{i}" for i in range(50)]
    write_gguf(p, {"general.architecture": "llama", "general.name": "Tiny", "general.file_type": 15,
                   "tokenizer.ggml.tokens": vocab, "tokenizer.ggml.scores": list(range(30)),
                   "llama.block_count": 8, "llama.embedding_length": 64, "llama.attention.head_count": 4,
                   "llama.attention.head_count_kv": 2, "llama.context_length": 2048},
               [("tok_embd", [64, 50]), ("blk.0.attn_q", [64, 64])])
    m = read_metadata(p, tensors=True)
    assert m["general.name"] == "Tiny" and m["tokenizer.ggml.tokens"] == "<array of 50>"
    assert m["tokenizer.ggml.scores"] == "<array of 30>" and m["gguf.n_params"] == 64 * 50 + 64 * 64
    assert read_metadata(p, keep_arrays=True)["tokenizer.ggml.tokens"] == vocab
    info = model_info(p)
    assert (info["quant"], info["n_layer"], info["kv_layers"], info["head_dim"]) == ("Q4_K_M", 8, 8, 16)
    # keys + values × 8 layers × 2 kv heads × 16 dims × 100 positions × 2 bytes (f16)
    assert kv_cache_bytes(info, 100) == 2 * 8 * 2 * 16 * 100 * 2


def test_hybrid_models_cache_only_their_attention_layers(tmp_path):
    p = tmp_path / "hybrid.gguf"
    write_gguf(p, {"general.architecture": "qwen35", "qwen35.block_count": 24, "qwen35.embedding_length": 1024,
                   "qwen35.attention.head_count": 8, "qwen35.attention.head_count_kv": 2,
                   "qwen35.attention.key_length": 256, "qwen35.full_attention_interval": 4}, [("w", [4, 4])])
    info = model_info(p)
    assert info["kv_layers"] == 6 and info["head_dim"] == 256


def test_not_a_gguf(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"nope")
    with pytest.raises(ValueError):
        read_metadata(tmp_path / "x.gguf")


# ---------- planner
def machine(ram_avail: float, bandwidth: float = 20.0, gpus=()) -> Hardware:
    return Hardware(os="Linux", arch="x86_64", cpu="test", physical_cores=4, logical_cores=8, flags=["avx2"],
                    ram_total_gb=ram_avail * 2, ram_available_gb=ram_avail, gpus=list(gpus), bandwidth_gbs=bandwidth)


def fake_model(name: str, params_b: float, gb: float) -> dict:
    return {"file": name, "params": int(params_b * 1e9), "quant": "Q4_K_M", "bytes": int(gb * GB),
            "kv_layers": 28, "n_head_kv": 2, "head_dim": 128}


MODELS = [fake_model("small", 0.8, 0.5), fake_model("mid", 1.5, 1.0), fake_model("big", 7.0, 4.4)]


def test_plan_takes_the_largest_model_that_fits_and_is_fast_enough():
    assert plan_local(machine(16), MODELS, min_tok_s=4, eff=1.0)["choice"] == "big"  # 20 GB/s / 4.4 GB = 4.5 tok/s...
    assert plan_local(machine(16), MODELS, min_tok_s=5, eff=1.0)["choice"] == "mid"  # ...too slow at a 5 tok/s floor
    assert plan_local(machine(3), MODELS, min_tok_s=1, eff=1.0)["choice"] == "mid"   # big needs ~5 GB: doesn't fit


def test_plan_when_nothing_is_fast_or_nothing_fits():
    slow = plan_local(machine(16, bandwidth=1.0), MODELS, eff=1.0)
    assert slow["choice"] == "small" and "fastest" in slow["why"]
    none = plan_local(machine(1.2), MODELS, eff=1.0)
    assert none["choice"] is None


def test_threads_and_gpu_layers():
    assert plan_local(machine(16), MODELS, eff=1.0)["threads"] == 4                  # physical cores, not 8 threads
    assert gpu_layers(machine(16, gpus=["Apple GPU (Metal, unified memory)"])) == -1
    assert gpu_layers(machine(16)) == 0


def test_quant_grid_never_recommends_below_q4():
    for row in quant_grid(machine(6, bandwidth=20), eff=1.0):
        assert row["best"] in (None, "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
    rows = {r["params_b"]: r for r in quant_grid(machine(6, bandwidth=20), eff=1.0)}
    assert rows[0.5]["best"] == "Q8_0"                  # a small model: the highest quality fits easily
    assert rows[14]["best"] is None                     # 14B at 4 bits needs ~8.6 GB > 5 GB budget
