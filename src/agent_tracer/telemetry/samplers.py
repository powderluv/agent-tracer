"""Probe functions for individual telemetry sources.

Each probe returns a list of zero-or-more records on success and an empty
list on any failure (missing binary, parse error, GPU absent). The daemon
calls them every tick; intermittent failures (e.g., a GPU getting reset)
must not crash the loop.

The probes intentionally know nothing about storage — they return plain
dataclasses. The daemon stamps a single ``ts_us`` per tick onto whatever
each probe returned, so all rows from one tick share a timestamp.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Tools that take noticeably long can starve the loop. Cap each probe so
# one slow tool can't block the next tick by more than a fraction of the
# interval. nvidia-smi is usually <50ms; rocm-smi 50-200ms; amd-smi
# similar.
_PROBE_TIMEOUT_S = 1.5


# rocm-smi / amd-smi are often installed outside $PATH:
#   /opt/rocm/bin/, /opt/rocm-*/bin/, <venv>/bin/ (TheRock wheels do this),
#   or wherever an env var pins them. Resolve in priority order and cache.
def _resolve_tool(name: str, env_var: str) -> str | None:
    """Find an executable named ``name`` across standard ROCm install paths.

    Lookup order: env var override, the active venv's bin/, /opt/rocm/bin,
    /opt/rocm-*/bin (glob), then $PATH via shutil.which.
    """
    override = os.environ.get(env_var)
    if override and os.access(override, os.X_OK):
        return override

    candidates: list[str] = []
    venv_bin = Path(sys.executable).resolve().parent
    candidates.append(str(venv_bin / name))
    candidates.append(f"/opt/rocm/bin/{name}")
    candidates.extend(sorted(glob.glob(f"/opt/rocm-*/bin/{name}"), reverse=True))

    for c in candidates:
        if os.access(c, os.X_OK):
            return c

    return shutil.which(name)


_tool_cache: dict[str, str | None] = {}


def _tool(name: str, env_var: str) -> str | None:
    if name not in _tool_cache:
        _tool_cache[name] = _resolve_tool(name, env_var)
    return _tool_cache[name]


def reset_tool_cache() -> None:
    """Force re-resolution of binary paths. For tests and rare reload cases."""
    _tool_cache.clear()


@dataclass(slots=True)
class GpuSample:
    vendor: str             # "amd" | "nvidia"
    gpu_idx: int
    gpu_name: str | None = None
    util_pct: float | None = None       # 0..100
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    power_w: float | None = None
    temp_c: float | None = None
    # vendor-specific extras get stashed here for the wide-table view in
    # Lance — null columns compress well.
    extras: dict[str, float | str] = field(default_factory=dict)


@dataclass(slots=True)
class SystemSample:
    cpu_util_pct: float | None
    mem_used_mb: float | None
    mem_total_mb: float | None
    load_1m: float | None
    load_5m: float | None
    load_15m: float | None
    host: str


# --- system (always available on Linux) -----------------------------------


def _proc_stat_cpu_jiffies() -> tuple[int, int] | None:
    """Return (busy_jiffies, total_jiffies) from /proc/stat's aggregate row."""
    try:
        with open("/proc/stat", "rb") as f:
            first = f.readline().decode("ascii", errors="replace")
    except OSError:
        return None
    if not first.startswith("cpu "):
        return None
    parts = first.split()[1:]
    try:
        vals = [int(v) for v in parts]
    except ValueError:
        return None
    if len(vals) < 4:
        return None
    idle = vals[3]
    total = sum(vals)
    return total - idle, total


class _CpuMeter:
    """Stateful delta meter for /proc/stat. Persists across ticks via instance."""

    def __init__(self) -> None:
        self._prev: tuple[int, int] | None = None

    def sample_pct(self) -> float | None:
        cur = _proc_stat_cpu_jiffies()
        if cur is None:
            return None
        prev = self._prev
        self._prev = cur
        if prev is None:
            return None
        d_busy = cur[0] - prev[0]
        d_total = cur[1] - prev[1]
        if d_total <= 0:
            return None
        return min(100.0, max(0.0, d_busy * 100.0 / d_total))


def _read_meminfo() -> dict[str, float]:
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo", "rb") as f:
            for raw in f:
                line = raw.decode("ascii", errors="replace")
                if ":" not in line:
                    continue
                k, rest = line.split(":", 1)
                parts = rest.split()
                if not parts:
                    continue
                try:
                    val = float(parts[0])
                except ValueError:
                    continue
                # /proc/meminfo is in kB; convert to MB.
                if len(parts) > 1 and parts[1].lower() == "kb":
                    val = val / 1024.0
                out[k.strip()] = val
    except OSError:
        pass
    return out


def _read_loadavg() -> tuple[float, float, float] | None:
    try:
        with open("/proc/loadavg", "rb") as f:
            line = f.readline().decode("ascii", errors="replace")
    except OSError:
        return None
    try:
        a, b, c = (float(x) for x in line.split()[:3])
    except (ValueError, IndexError):
        return None
    return a, b, c


def sample_system(cpu_meter: _CpuMeter) -> SystemSample:
    meminfo = _read_meminfo()
    load = _read_loadavg()
    total_mb = meminfo.get("MemTotal")
    avail_mb = meminfo.get("MemAvailable")
    used_mb = (total_mb - avail_mb) if (total_mb and avail_mb) else None
    return SystemSample(
        cpu_util_pct=cpu_meter.sample_pct(),
        mem_used_mb=used_mb,
        mem_total_mb=total_mb,
        load_1m=load[0] if load else None,
        load_5m=load[1] if load else None,
        load_15m=load[2] if load else None,
        host=socket.gethostname(),
    )


# --- nvidia-smi ------------------------------------------------------------


_NVIDIA_QUERY = (
    "index,name,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,temperature.gpu"
)


def sample_nvidia() -> list[GpuSample]:
    binary = _tool("nvidia-smi", "AGENT_TRACER_NVIDIA_SMI")
    if not binary:
        return []
    try:
        out = subprocess.run(
            [
                binary,
                f"--query-gpu={_NVIDIA_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0 or not out.stdout:
        return []
    samples: list[GpuSample] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        s = GpuSample(vendor="nvidia", gpu_idx=idx, gpu_name=parts[1] or None)
        s.util_pct = _safe_float(parts[2])
        s.vram_used_mb = _safe_float(parts[4])
        s.vram_total_mb = _safe_float(parts[5])
        s.power_w = _safe_float(parts[6])
        s.temp_c = _safe_float(parts[7])
        mem_util = _safe_float(parts[3])
        if mem_util is not None:
            s.extras["mem_util_pct"] = mem_util
        samples.append(s)
    return samples


# --- rocm-smi --------------------------------------------------------------


def sample_amd() -> list[GpuSample]:
    """Use rocm-smi's --json output. Returns [] if the driver isn't loaded."""
    binary = _tool("rocm-smi", "AGENT_TRACER_ROCM_SMI")
    if not binary:
        return []
    try:
        out = subprocess.run(
            [
                binary,
                "--showuse",
                "--showmemuse",
                "--showpower",
                "--showtemp",
                "--showmeminfo",
                "vram",
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            check=False,
            # Keep stderr quiet on bricked GPUs; the empty return is the signal.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0 or not out.stdout:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    samples: list[GpuSample] = []
    for key, fields in data.items():
        if not isinstance(fields, dict):
            continue
        # Keys look like "card0", "card1", ... and a meta "system" row.
        if not key.startswith("card"):
            continue
        try:
            idx = int(key[len("card"):])
        except ValueError:
            continue
        s = GpuSample(vendor="amd", gpu_idx=idx)
        s.util_pct = _scrape_num(fields, ("GPU use (%)", "GPU Activity (%)"))
        s.power_w = _scrape_num(
            fields, ("Average Graphics Package Power (W)", "Current Socket Graphics Package Power (W)")
        )
        s.temp_c = _scrape_num(
            fields, ("Temperature (Sensor edge) (C)", "Temperature (Sensor junction) (C)")
        )
        s.vram_used_mb = _scrape_bytes_to_mb(fields, ("VRAM Total Used Memory (B)",))
        s.vram_total_mb = _scrape_bytes_to_mb(fields, ("VRAM Total Memory (B)",))
        mem_util = _scrape_num(fields, ("GPU memory use (%)", "GPU Memory Activity (%)"))
        if mem_util is not None:
            s.extras["mem_util_pct"] = mem_util
        samples.append(s)
    return samples


# --- helpers ---------------------------------------------------------------


def _safe_float(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s or s in {"-", "N/A", "[N/A]", "[Not Supported]"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _scrape_num(fields: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k in fields:
            v = _safe_float(str(fields[k]))
            if v is not None:
                return v
    return None


def _scrape_bytes_to_mb(fields: dict, keys: tuple[str, ...]) -> float | None:
    v = _scrape_num(fields, keys)
    return None if v is None else v / (1024 * 1024)


__all__ = [
    "GpuSample",
    "SystemSample",
    "_CpuMeter",
    "reset_tool_cache",
    "sample_amd",
    "sample_nvidia",
    "sample_system",
]
