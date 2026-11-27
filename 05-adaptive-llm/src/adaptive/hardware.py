"""What machine are we on? Cores, CPU instruction sets, RAM, GPU, and measured memory bandwidth.

Standard library only, with one code path per OS:
- Linux: /proc/cpuinfo (model, flags, physical cores = distinct (socket, core id) pairs) and /proc/meminfo.
- macOS: `sysctl` (hw.physicalcpu, hw.memsize, CPU features) and `vm_stat` for free memory.
- Windows: kernel32 GlobalMemoryStatusEx and IsProcessorFeaturePresent through ctypes; cores from PowerShell.

Why each fact matters for running a language model on a CPU:
- **physical cores** -> threads. Hyper-threads share a core's arithmetic units; for matrix maths they add contention
  (measured in Part A: 2 threads beat 4 by 12-20% on this 2-core laptop).
- **AVX2 / AVX-512 / FMA / F16C / NEON** -> which fast kernels llama.cpp can use; without AVX2 an x86 CPU is several
  times slower.
- **available RAM** -> the largest model (+ its attention cache) that fits without swapping.
- **memory bandwidth** -> speed. Generating one token reads every weight once, so tokens/s is at most
  bandwidth / model size. We measure it with a large array copy rather than trusting a spec sheet.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

GB = 1024 ** 3
FLAGS_OF_INTEREST = ["avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni", "avx512_bf16", "amx_tile", "neon",
                     "dotprod", "i8mm", "sve"]


@dataclass
class Hardware:
    os: str
    arch: str
    cpu: str
    physical_cores: int
    logical_cores: int
    flags: list[str]
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[str] = field(default_factory=list)
    bandwidth_gbs: float | None = None          # measured; None until measure_bandwidth() runs

    def has(self, flag: str) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict:
        return asdict(self)


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ---------- Linux
def parse_cpuinfo(text: str) -> tuple[str, int, int, set[str]]:
    """(model name, physical cores, logical cores, flags) from /proc/cpuinfo. ARM lists 'Features' instead of 'flags'."""
    blocks = [b for b in text.strip().split("\n\n") if b.strip()]
    name, flags, cores, logical = "", set(), set(), 0
    for b in blocks:
        kv = {}
        for line in b.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                kv[k.strip()] = v.strip()
        name = name or kv.get("model name") or kv.get("Model") or kv.get("Hardware", "")
        if "processor" not in kv:                   # ARM ends with a board block (Hardware, Model): not a CPU
            continue
        logical += 1
        flags |= set((kv.get("flags") or kv.get("Features") or "").split())
        cores.add((kv.get("physical id", "0"), kv.get("core id", kv["processor"])))
    if "asimd" in flags:                             # Linux calls ARM's NEON "asimd"
        flags.add("neon")
    if "asimddp" in flags:
        flags.add("dotprod")
    return name, len(cores) or logical, logical, flags


def parse_meminfo(text: str) -> tuple[float, float]:
    kb = {m[1]: int(m[2]) for m in re.finditer(r"^(\w+):\s+(\d+) kB", text, re.M)}
    total = kb.get("MemTotal", 0)
    avail = kb.get("MemAvailable", kb.get("MemFree", 0) + kb.get("Cached", 0))
    return total * 1024 / GB, avail * 1024 / GB


def _linux() -> dict:
    name, phys, logical, flags = parse_cpuinfo(Path("/proc/cpuinfo").read_text())
    total, avail = parse_meminfo(Path("/proc/meminfo").read_text())
    return dict(cpu=name, physical_cores=phys, logical_cores=logical, flags=flags, ram_total_gb=total,
                ram_available_gb=avail)


# ---------- macOS
def _mac() -> dict:
    sysctl = lambda key: _run(["sysctl", "-n", key]).strip()       # noqa: E731
    flags = set((sysctl("machdep.cpu.features") + " " + sysctl("machdep.cpu.leaf7_features")).lower().split())
    if platform.machine() == "arm64":                             # Apple Silicon: NEON always, dot product on all M-series
        flags |= {"neon", "dotprod"}
        if sysctl("hw.optional.arm.FEAT_I8MM") == "1":
            flags.add("i8mm")
    page = int(sysctl("hw.pagesize") or 4096)
    pages = {m[1]: int(m[2]) for m in re.finditer(r"Pages (\w+[\w ]*):\s+(\d+)", _run(["vm_stat"]))}
    avail = (pages.get("free", 0) + pages.get("inactive", 0) + pages.get("speculative", 0)) * page
    return dict(cpu=sysctl("machdep.cpu.brand_string"), physical_cores=int(sysctl("hw.physicalcpu") or os.cpu_count()),
                logical_cores=int(sysctl("hw.logicalcpu") or os.cpu_count()), flags=flags,
                ram_total_gb=int(sysctl("hw.memsize") or 0) / GB, ram_available_gb=avail / GB)


# ---------- Windows
class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _windows() -> dict:
    k32 = ctypes.windll.kernel32                                  # type: ignore[attr-defined]
    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    k32.GlobalMemoryStatusEx(ctypes.byref(st))
    feature = {"avx": 39, "avx2": 40, "avx512f": 41, "neon": 19, "dotprod": 43}   # PF_* constants
    flags = {f for f, pf in feature.items() if k32.IsProcessorFeaturePresent(pf)}
    if "avx2" in flags:
        flags |= {"fma", "f16c"}                                  # every AVX2 CPU has them (Haswell and later)
    ps = _run(["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores | ConvertTo-Json"])
    try:
        procs = json.loads(ps)
        procs = procs if isinstance(procs, list) else [procs]
        name, phys = procs[0]["Name"].strip(), sum(p["NumberOfCores"] for p in procs)
    except (ValueError, KeyError, IndexError, TypeError):
        name, phys = platform.processor(), os.cpu_count() or 1
    return dict(cpu=name, physical_cores=phys, logical_cores=os.cpu_count() or phys, flags=flags,
                ram_total_gb=st.ullTotalPhys / GB, ram_available_gb=st.ullAvailPhys / GB)


def _gpus(system: str) -> list[str]:
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        return [f"NVIDIA {l.strip()}" for l in out.splitlines() if l.strip()]
    if system == "Darwin" and platform.machine() == "arm64":
        return ["Apple GPU (Metal, unified memory)"]
    return []


def profile(bandwidth: bool = True) -> Hardware:
    system = platform.system()
    facts = {"Linux": _linux, "Darwin": _mac, "Windows": _windows}.get(system, _linux)()
    facts["flags"] = sorted(f for f in facts["flags"] if f in FLAGS_OF_INTEREST)
    hw = Hardware(os=system, arch=platform.machine(), gpus=_gpus(system),
                  **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in facts.items()})
    if bandwidth:
        hw.bandwidth_gbs = measure_bandwidth(threads=hw.physical_cores)
    return hw


def slug(cpu: str) -> str:
    """A file name for a CPU: 'Intel(R) Core(TM) i5-5200U CPU @ 2.20GHz' -> 'intel-core-i5-5200u'."""
    cpu = re.sub(r"\(r\)|\(tm\)|\bcpu\b|@.*$", "", cpu.lower())
    return re.sub(r"[^a-z0-9]+", "-", cpu).strip("-") or "machine"


def measure_bandwidth(mb: int = 512, threads: int | None = None, repeats: int = 5) -> float:
    """Effective memory *read* bandwidth in GB/s, using `threads` threads (default: physical cores).

    Generating a token reads every weight once and writes almost nothing, so a read test matches the workload. The
    array is far bigger than any CPU cache, so this measures RAM. Each thread sums its own slice; numpy releases the
    GIL inside the sum, so the threads really run in parallel. Best of `repeats` runs (others are noise from the OS).
    A single thread cannot saturate the memory bus on most CPUs, which is why the default uses every core.
    """
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    threads = threads or _physical_cores()
    data = np.ones(mb * 1024 * 1024 // 8)
    parts = np.array_split(data, threads)
    best = float("inf")
    with ThreadPoolExecutor(threads) as pool:
        for _ in range(repeats):
            t = time.perf_counter()
            list(pool.map(np.sum, parts))
            best = min(best, time.perf_counter() - t)
    return round(data.nbytes / best / 1e9, 2)


def _physical_cores() -> int:
    try:
        return {"Linux": _linux, "Darwin": _mac, "Windows": _windows}.get(platform.system(), _linux)()["physical_cores"]
    except Exception:                                             # noqa: BLE001 - any failure: fall back to a guess
        return max(1, (os.cpu_count() or 2) // 2)
