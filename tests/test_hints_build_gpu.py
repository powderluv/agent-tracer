"""Build-pattern and telemetry-correlated detectors."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints.build import (
    EXPUNGE_CHAIN_MIN,
    REBUILD_MIN_PER_DAY,
    SSH_OVERHEAD_MIN_S,
    detect_expunge_chain,
    detect_repeated_rebuilds,
    detect_ssh_overhead,
)
from agent_tracer.hints.gpu import (
    GPU_IDLE_BUILD_MIN_SPAN_S,
    detect_gpu_idle_build,
    detect_host_bound_gpu,
    detect_vram_pressure,
)
from agent_tracer.telemetry.reader import TelemetryReader

pyarrow = pytest.importorskip("pyarrow")
lancedb = pytest.importorskip("lancedb")

# 2026-05-12 00:00:00Z in epoch microseconds.
DAY_BASE_US = 1778803200_000_000


def _bash_build(target: str, ts_us: int, dur_us: int, source="claude", session_id="s1"):
    return AgentEvent(
        source=source,  # type: ignore[arg-type]
        session_id=session_id,
        kind=EventKind.TOOL_CALL,
        name="Bash",
        ts_start_us=ts_us,
        ts_end_us=ts_us + dur_us,
        category="build",
        payload={"input": {"command": f"ninja -j 32 {target}"}},
    )


def _ssh(cmd: str, ts_us: int, dur_us: int, session_id="s1"):
    return AgentEvent(
        source="codex",
        session_id=session_id,
        kind=EventKind.TOOL_CALL,
        name="exec_command",
        ts_start_us=ts_us,
        ts_end_us=ts_us + dur_us,
        category="network",
        payload={"input": {"cmd": cmd}},
    )


# --- repeated rebuilds ----------------------------------------------------


def test_repeated_rebuilds_fires_at_threshold() -> None:
    target = "clr+dist"
    events = [
        _bash_build(target, DAY_BASE_US + i * 3_600_000_000, 60_000_000)
        for i in range(REBUILD_MIN_PER_DAY)
    ]
    hints = list(detect_repeated_rebuilds(events))
    assert len(hints) == 1
    assert "clr+dist" in hints[0].title
    assert hints[0].occurrences == REBUILD_MIN_PER_DAY


def test_repeated_rebuilds_groups_by_day_and_target() -> None:
    events = []
    # Day 1: 4 of target A
    for i in range(4):
        events.append(_bash_build("A", DAY_BASE_US + i * 60_000_000, 1))
    # Day 2: 1 of target A → below threshold
    for i in range(1):
        events.append(_bash_build("A", DAY_BASE_US + 86_400_000_000 + i, 1))
    hints = list(detect_repeated_rebuilds(events))
    assert len(hints) == 1
    assert hints[0].evidence["target"] == "A"


# --- expunge chain --------------------------------------------------------


def test_expunge_chain_pairs_adjacent_expunge_then_dist() -> None:
    events = []
    for i in range(EXPUNGE_CHAIN_MIN):
        events.append(_bash_build("X+expunge", DAY_BASE_US + i * 200_000_000, 1))
        events.append(_bash_build("X+dist", DAY_BASE_US + i * 200_000_000 + 1, 1))
    hints = list(detect_expunge_chain(events))
    assert len(hints) == 1 and hints[0].occurrences == EXPUNGE_CHAIN_MIN


def test_expunge_chain_ignores_dist_without_preceding_expunge() -> None:
    events = [_bash_build("X+dist", DAY_BASE_US + i, 1) for i in range(5)]
    assert list(detect_expunge_chain(events)) == []


# --- ssh overhead ---------------------------------------------------------


def test_ssh_overhead_fires_over_threshold() -> None:
    # 4 × 10s ssh = 40s, above the 30s threshold.
    events = [
        _ssh("ssh nod@host pwd", i * 10_000_000, 10_000_000)
        for i in range(4)
    ]
    hints = list(detect_ssh_overhead(events))
    assert len(hints) == 1
    assert hints[0].est_wall_saved_s is not None
    assert hints[0].est_wall_saved_s > 0


def test_ssh_overhead_filters_non_ssh_network_calls() -> None:
    events = [
        _ssh(f"curl https://example.com/{i}", i * 10_000_000, 10_000_000)
        for i in range(int(SSH_OVERHEAD_MIN_S) + 1)
    ]
    assert list(detect_ssh_overhead(events)) == []


# --- telemetry reader -----------------------------------------------------


@pytest.fixture()
def telemetry_dataset(tmp_path: Path) -> Path:
    """Build a tiny populated Lance dataset for the GPU detectors."""
    from agent_tracer.telemetry.samplers import GpuSample, SystemSample
    from agent_tracer.telemetry.store import TelemetryWriter

    path = tmp_path / "telemetry.lance"
    w = TelemetryWriter(path=path, flush_every=1)
    # 60 samples at 1Hz starting at DAY_BASE_US.
    for i in range(60):
        ts = DAY_BASE_US + i * 1_000_000
        w.add_gpu(
            ts,
            "h",
            GpuSample(
                vendor="amd",
                gpu_idx=0,
                gpu_name="rx9070xt",
                util_pct=1.0 if i < 30 else 85.0,
                vram_used_mb=2000 if i < 30 else 23000,
                vram_total_mb=24000,
                power_w=50.0,
                temp_c=40.0,
            ),
        )
        w.add_system(
            ts,
            SystemSample(
                cpu_util_pct=95.0 if i < 30 else 95.0,
                mem_used_mb=8000,
                mem_total_mb=128000,
                load_1m=8.0,
                load_5m=8.0,
                load_15m=8.0,
                host="h",
            ),
        )
    w.close()
    return path


def test_telemetry_reader_window_aggregates(telemetry_dataset: Path) -> None:
    r = TelemetryReader(dataset=telemetry_dataset)
    assert r.load()
    # Window over the first 30 samples — GPU idle, CPU high.
    ws = r.window_stats(DAY_BASE_US, DAY_BASE_US + 30 * 1_000_000)
    assert ws.n_gpu_samples == 30
    assert ws.n_system_samples == 30
    assert ws.gpu_util_mean is not None and ws.gpu_util_mean < 2
    assert ws.cpu_util_mean is not None and ws.cpu_util_mean > 90


def test_telemetry_reader_returns_empty_stats_when_dataset_absent(tmp_path: Path) -> None:
    r = TelemetryReader(dataset=tmp_path / "does-not-exist.lance")
    assert not r.load()
    ws = r.window_stats(0, 1_000_000)
    assert ws.n_gpu_samples == 0 and ws.n_system_samples == 0


# --- gpu detectors --------------------------------------------------------


def test_gpu_idle_build_fires_on_long_idle_span(telemetry_dataset: Path) -> None:
    # Build span = first 30s, during which GPU util mean is ~1%.
    span_us = int(GPU_IDLE_BUILD_MIN_SPAN_S * 1_000_000)
    events = [_bash_build("clr+dist", DAY_BASE_US, span_us)]
    r = TelemetryReader(dataset=telemetry_dataset)
    hints = list(detect_gpu_idle_build(events, r))
    assert len(hints) == 1
    assert hints[0].occurrences == 1


def test_host_bound_gpu_fires_when_both_high(telemetry_dataset: Path) -> None:
    # Window over samples 30..60: GPU 85%, CPU 95%.
    ts_start = DAY_BASE_US + 30 * 1_000_000
    ev = AgentEvent(
        source="claude",
        session_id="s1",
        kind=EventKind.TOOL_CALL,
        name="Bash",
        ts_start_us=ts_start,
        ts_end_us=ts_start + 30_000_000,
        category="test",
        payload={"input": {"command": "pytest tests/"}},
    )
    r = TelemetryReader(dataset=telemetry_dataset)
    hints = list(detect_host_bound_gpu([ev], r))
    assert len(hints) == 1


def test_vram_pressure_fires_when_used_over_90pct(telemetry_dataset: Path) -> None:
    # Window over samples 30..60: VRAM 23000/24000 ≈ 95.8%.
    ts_start = DAY_BASE_US + 30 * 1_000_000
    ev = AgentEvent(
        source="claude",
        session_id="s1",
        kind=EventKind.TOOL_CALL,
        name="Bash",
        ts_start_us=ts_start,
        ts_end_us=ts_start + 30_000_000,
        category="gpu",
        payload={"input": {"command": "rocprof ./bench"}},
    )
    r = TelemetryReader(dataset=telemetry_dataset)
    hints = list(detect_vram_pressure([ev], r))
    assert len(hints) == 1


def test_gpu_detectors_silent_without_telemetry(tmp_path: Path) -> None:
    events = [_bash_build("X", DAY_BASE_US, 60_000_000)]
    r = TelemetryReader(dataset=tmp_path / "nope.lance")
    # load() returns False, run_all_gpu returns [].
    from agent_tracer.hints.gpu import run_all_gpu

    assert run_all_gpu(events, r) == []
