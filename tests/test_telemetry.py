"""Telemetry: probe parsers, tool resolution, store, daemon one-tick."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from agent_tracer.telemetry import samplers
from agent_tracer.telemetry.daemon import run
from agent_tracer.telemetry.samplers import (
    GpuSample,
    SystemSample,
    _CpuMeter,
    _resolve_tool,
    reset_tool_cache,
    sample_amd,
    sample_nvidia,
    sample_system,
)
from agent_tracer.telemetry.store import TelemetryWriter

pyarrow = pytest.importorskip("pyarrow")
lancedb = pytest.importorskip("lancedb")


# --- tool resolver ----------------------------------------------------------


def test_resolve_tool_env_override_wins(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "rocm-smi"
    fake.write_text("#!/bin/sh\necho fake\n")
    fake.chmod(0o755)
    monkeypatch.setenv("AGENT_TRACER_ROCM_SMI", str(fake))
    assert _resolve_tool("rocm-smi", "AGENT_TRACER_ROCM_SMI") == str(fake)


def test_resolve_tool_checks_venv_bin(tmp_path: Path, monkeypatch) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "rocm-smi"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    monkeypatch.delenv("AGENT_TRACER_ROCM_SMI", raising=False)
    monkeypatch.setattr(samplers.os.environ, "get", lambda k, *a: None)
    monkeypatch.setattr(samplers.shutil, "which", lambda _: None)
    monkeypatch.setattr(samplers.glob, "glob", lambda _: [])
    # Block the /opt/rocm fallback path.
    with mock.patch("os.access", side_effect=lambda p, _: p == str(fake)):
        assert _resolve_tool("rocm-smi", "AGENT_TRACER_ROCM_SMI") == str(fake)


def test_resolve_tool_returns_none_when_nothing_found(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_TRACER_ROCM_SMI", raising=False)
    with mock.patch("os.access", return_value=False), mock.patch(
        "shutil.which", return_value=None
    ), mock.patch("glob.glob", return_value=[]):
        assert _resolve_tool("rocm-smi", "AGENT_TRACER_ROCM_SMI") is None


# --- nvidia-smi parser ------------------------------------------------------


def test_sample_nvidia_parses_csv(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "_tool", lambda name, env: f"/usr/bin/{name}")

    class FakeCompleted:
        returncode = 0
        stdout = (
            "0, NVIDIA RTX PRO 6000 Blackwell, 23, 4, 1234, 97887, 145.5, 56\n"
            "1, NVIDIA RTX A6000, 0, 0, 100, 49152, 30.1, 32\n"
        )

    monkeypatch.setattr(
        samplers.subprocess, "run", lambda *a, **kw: FakeCompleted()
    )
    out = sample_nvidia()
    assert len(out) == 2
    g0 = out[0]
    assert g0.vendor == "nvidia"
    assert g0.gpu_idx == 0
    assert g0.gpu_name.startswith("NVIDIA RTX PRO 6000")
    assert g0.util_pct == 23
    assert g0.vram_used_mb == 1234
    assert g0.vram_total_mb == 97887
    assert g0.power_w == 145.5
    assert g0.temp_c == 56
    assert g0.extras["mem_util_pct"] == 4


def test_sample_nvidia_silent_when_tool_missing(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "_tool", lambda *a, **kw: None)
    assert sample_nvidia() == []


def test_sample_nvidia_handles_subprocess_failure(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "_tool", lambda *a, **kw: "/usr/bin/nvidia-smi")

    class FakeCompleted:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(samplers.subprocess, "run", lambda *a, **kw: FakeCompleted())
    assert sample_nvidia() == []


# --- rocm-smi parser --------------------------------------------------------


def test_sample_amd_parses_json(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "_tool", lambda *a, **kw: "/usr/bin/rocm-smi")

    class FakeCompleted:
        returncode = 0
        stdout = json.dumps(
            {
                "card0": {
                    "GPU use (%)": "7",
                    "GPU memory use (%)": "3",
                    "Average Graphics Package Power (W)": "150.2",
                    "Temperature (Sensor edge) (C)": "55",
                    "VRAM Total Used Memory (B)": str(2 * 1024 * 1024 * 1024),
                    "VRAM Total Memory (B)": str(24 * 1024 * 1024 * 1024),
                },
                "system": {"meta": "ignored"},
            }
        )

    monkeypatch.setattr(samplers.subprocess, "run", lambda *a, **kw: FakeCompleted())
    out = sample_amd()
    assert len(out) == 1
    g = out[0]
    assert g.vendor == "amd"
    assert g.gpu_idx == 0
    assert g.util_pct == 7
    assert g.power_w == 150.2
    assert g.temp_c == 55
    assert g.vram_used_mb == 2 * 1024
    assert g.vram_total_mb == 24 * 1024


def test_sample_amd_returns_empty_on_driver_not_initialized(monkeypatch) -> None:
    monkeypatch.setattr(samplers, "_tool", lambda *a, **kw: "/usr/bin/rocm-smi")

    class FakeCompleted:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(samplers.subprocess, "run", lambda *a, **kw: FakeCompleted())
    assert sample_amd() == []


# --- system probe -----------------------------------------------------------


def test_sample_system_returns_finite_load_and_mem() -> None:
    meter = _CpuMeter()
    meter.sample_pct()  # prime
    s = sample_system(meter)
    assert s.host
    if s.mem_total_mb is not None and s.mem_used_mb is not None:
        assert 0 <= s.mem_used_mb <= s.mem_total_mb
    if s.load_1m is not None:
        assert s.load_1m >= 0


def test_cpu_meter_returns_none_on_first_call() -> None:
    m = _CpuMeter()
    assert m.sample_pct() is None
    # Some time later, after a second call, it can produce a value.
    v = m.sample_pct()
    assert v is None or (0.0 <= v <= 100.0)


# --- LanceDB writer ---------------------------------------------------------


def test_writer_round_trip(tmp_path: Path) -> None:
    w = TelemetryWriter(path=tmp_path / "t.lance", flush_every=10, flush_seconds=1000)
    w.add_gpu(
        1_000_000,
        "host1",
        GpuSample(vendor="nvidia", gpu_idx=0, gpu_name="X", util_pct=12.5, vram_used_mb=100),
    )
    w.add_system(
        1_000_000,
        SystemSample(
            cpu_util_pct=4.5,
            mem_used_mb=8000,
            mem_total_mb=128000,
            load_1m=0.5,
            load_5m=0.4,
            load_15m=0.3,
            host="host1",
        ),
    )
    gpu_n, sys_n = w.flush()
    assert gpu_n == 1 and sys_n == 1

    import lancedb as _lancedb

    db = _lancedb.connect(tmp_path / "t.lance")
    g = db.open_table("gpu_telemetry").to_arrow()
    s = db.open_table("system_telemetry").to_arrow()
    assert g.num_rows == 1 and g.column("util_pct").to_pylist() == [12.5]
    assert s.num_rows == 1 and s.column("cpu_util_pct").to_pylist() == [4.5]


def test_writer_appends_across_flushes(tmp_path: Path) -> None:
    p = tmp_path / "t.lance"
    for batch in (10.0, 20.0):
        w = TelemetryWriter(path=p, flush_every=1)
        w.add_system(
            1_000_000,
            SystemSample(
                cpu_util_pct=batch,
                mem_used_mb=None,
                mem_total_mb=None,
                load_1m=None,
                load_5m=None,
                load_15m=None,
                host="h",
            ),
        )
        w.close()
    import lancedb as _lancedb

    db = _lancedb.connect(p)
    s = db.open_table("system_telemetry").to_arrow()
    assert sorted(s.column("cpu_util_pct").to_pylist()) == [10.0, 20.0]


# --- daemon one-tick --------------------------------------------------------


def test_run_once_writes_one_system_row_with_real_proc(tmp_path: Path, monkeypatch) -> None:
    # Force probes to return no GPUs so the daemon only writes a system row.
    monkeypatch.setattr(samplers, "sample_amd", lambda: [])
    monkeypatch.setattr(samplers, "sample_nvidia", lambda: [])
    from agent_tracer.telemetry import daemon as daemon_mod

    monkeypatch.setattr(daemon_mod, "sample_amd", lambda: [])
    monkeypatch.setattr(daemon_mod, "sample_nvidia", lambda: [])

    out = tmp_path / "telemetry.lance"
    stats = run(interval_s=0.01, dataset=out, once=True, quiet=True)
    assert stats.ticks == 1
    assert stats.system_rows == 1
    # The Lance dataset should have a system table now; GPU table should not
    # exist since we stubbed both GPU probes.
    import lancedb as _lancedb

    db = _lancedb.connect(out)
    tables = getattr(db.list_tables(), "tables", None) or list(db.list_tables())
    assert "system_telemetry" in tables
    assert "gpu_telemetry" not in tables


# Keep the tool cache fresh for tests that monkeypatch _tool resolution.
@pytest.fixture(autouse=True)
def _reset_caches():
    reset_tool_cache()
    yield
    reset_tool_cache()


# Touch os module so the lint doesn't complain about unused import.
_ = os
