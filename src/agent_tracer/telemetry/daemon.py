"""Telemetry sampler main loop. Run via ``agent-tracer sample``.

Steady-rate sampler: every ``interval_s`` it probes (AMD, NVIDIA, system)
and persists the results to a LanceDB dataset. SIGINT/SIGTERM trigger a
clean flush. ``--once`` runs a single tick and exits (handy for ad-hoc
checks and for testing).
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from agent_tracer.telemetry.samplers import (
    _CpuMeter,
    sample_amd,
    sample_nvidia,
    sample_system,
)
from agent_tracer.telemetry.store import TelemetryWriter

DEFAULT_DATASET = Path.home() / ".cache" / "agent-tracer" / "telemetry.lance"


@dataclass(slots=True)
class TickStats:
    ticks: int = 0
    gpu_rows: int = 0
    system_rows: int = 0
    last_amd: int = 0
    last_nvidia: int = 0


def run(
    *,
    interval_s: float = 1.0,
    dataset: Path = DEFAULT_DATASET,
    once: bool = False,
    quiet: bool = False,
    status_every: float = 30.0,
    stop_event: threading.Event | None = None,
) -> TickStats:
    """Sample-and-write loop. Returns when ``once`` or ``stop_event`` fires.

    The function returns a :class:`TickStats` so callers (and tests) can
    assert what happened without parsing stderr.
    """
    if interval_s <= 0:
        raise ValueError("interval_s must be > 0")
    writer = TelemetryWriter(path=Path(dataset))
    cpu_meter = _CpuMeter()
    # Take a throwaway CPU sample so the next one has a delta.
    cpu_meter.sample_pct()

    stop = stop_event or threading.Event()
    if not once:
        _install_signal_handlers(stop)

    stats = TickStats()
    last_status = time.monotonic()
    next_tick = time.monotonic()

    try:
        while not stop.is_set():
            ts_us = int(time.time() * 1_000_000)
            amd = sample_amd()
            nv = sample_nvidia()
            sys_sample = sample_system(cpu_meter)
            stats.last_amd = len(amd)
            stats.last_nvidia = len(nv)
            for s in amd:
                writer.add_gpu(ts_us, sys_sample.host, s)
                stats.gpu_rows += 1
            for s in nv:
                writer.add_gpu(ts_us, sys_sample.host, s)
                stats.gpu_rows += 1
            writer.add_system(ts_us, sys_sample)
            stats.system_rows += 1
            stats.ticks += 1

            if once:
                break

            if not quiet and (time.monotonic() - last_status) >= status_every:
                _emit_status(stats)
                last_status = time.monotonic()

            # Drift-compensating sleep — accumulated lateness self-corrects.
            next_tick += interval_s
            wait = next_tick - time.monotonic()
            if wait > 0:
                stop.wait(wait)
            else:
                # We're behind by more than one interval — skip the catch-up
                # to avoid a tight loop, and resync.
                next_tick = time.monotonic()
    finally:
        writer.close()

    if not quiet:
        _emit_status(stats, final=True)
    return stats


def _install_signal_handlers(stop: threading.Event) -> None:
    def _handler(signum: int, _frame) -> None:  # noqa: ANN001
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _emit_status(s: TickStats, *, final: bool = False) -> None:
    tag = "final" if final else "status"
    print(
        f"[telemetry {tag}] ticks={s.ticks} gpu_rows={s.gpu_rows} "
        f"sys_rows={s.system_rows} amd_last={s.last_amd} nv_last={s.last_nvidia}",
        file=sys.stderr,
    )
