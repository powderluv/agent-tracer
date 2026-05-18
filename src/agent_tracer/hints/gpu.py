"""Telemetry-correlated detectors.

These need a populated ``telemetry.lance`` dataset to fire. When the
dataset is missing or doesn't cover the event window, the detectors
silently return nothing (rather than failing loudly), so they can be
folded into ``run_all`` unconditionally.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints.types import Anchor, Hint, Severity
from agent_tracer.telemetry.reader import TelemetryReader, WindowStats

# Threshold knobs — explicit so they're easy to tune in one place.
GPU_IDLE_BUILD_MIN_SPAN_S = 30.0
GPU_IDLE_UTIL_MEAN_PCT = 5.0
HOST_BOUND_GPU_UTIL_PCT = 70.0
HOST_BOUND_CPU_UTIL_PCT = 90.0
VRAM_PRESSURE_PCT = 90.0
MIN_SAMPLES_FOR_VERDICT = 5


def run_all_gpu(events: Iterable[AgentEvent], reader: TelemetryReader) -> list[Hint]:
    """Run all telemetry-correlated detectors. No-op if telemetry is absent."""
    if not reader.load():
        return []
    materialized = list(events)
    hints: list[Hint] = []
    hints.extend(detect_gpu_idle_build(materialized, reader))
    hints.extend(detect_host_bound_gpu(materialized, reader))
    hints.extend(detect_vram_pressure(materialized, reader))
    return hints


# ---------------------------------------------------------------------------


def detect_gpu_idle_build(
    events: Iterable[AgentEvent],
    reader: TelemetryReader,
) -> Iterator[Hint]:
    """`cat:build` span > GPU_IDLE_BUILD_MIN_SPAN_S with mean GPU util < threshold.

    Flags spans where you wait minutes on a build with the GPU sitting at
    near-idle — i.e., the build is CPU-bound and could likely benefit from
    higher parallelism or ccache.
    """
    offenders: list[tuple[AgentEvent, WindowStats]] = []
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.category != "build":
            continue
        if ev.ts_end_us is None or ev.duration_us is None:
            continue
        if ev.duration_us < GPU_IDLE_BUILD_MIN_SPAN_S * 1_000_000:
            continue
        ws = reader.window_stats(ev.ts_start_us, ev.ts_end_us)
        if ws.n_gpu_samples < MIN_SAMPLES_FOR_VERDICT or ws.gpu_util_mean is None:
            continue
        if ws.gpu_util_mean < GPU_IDLE_UTIL_MEAN_PCT:
            offenders.append((ev, ws))

    if not offenders:
        return
    total_wall_us = sum((e.duration_us or 0) for e, _ in offenders)
    anchors = [
        Anchor(
            source=e.source,
            session_id=e.session_id,
            ts_us=e.ts_start_us,
            detail=(
                f"{(e.duration_us or 0) / 1_000_000:.0f}s build, "
                f"GPU util mean {ws.gpu_util_mean:.1f}% — {_short_cmd(e)}"
            ),
        )
        for e, ws in sorted(offenders, key=lambda x: -(x[0].duration_us or 0))[:8]
    ]
    yield Hint(
        detector="gpu_idle_build",
        category="gpu",
        title=(
            f"{len(offenders)} build span(s) ran with mean GPU util "
            f"<{GPU_IDLE_UTIL_MEAN_PCT:.0f}% ({total_wall_us / 1_000_000:.0f}s total)"
        ),
        severity=Severity.MEDIUM if total_wall_us > 300_000_000 else Severity.LOW,
        occurrences=len(offenders),
        anchors=anchors,
        remediation=(
            "These builds are CPU-bound — the GPU isn't being used. "
            "Increase `-j`, enable ccache/sccache, or use distcc. If the build "
            "is supposed to exercise the GPU (e.g., a kernel test), the GPU "
            "may not actually be active during the run."
        ),
    )


# ---------------------------------------------------------------------------


def detect_host_bound_gpu(
    events: Iterable[AgentEvent],
    reader: TelemetryReader,
) -> Iterator[Hint]:
    """High GPU util and high host CPU util at the same time.

    Suggests host-side kernel-launch / data-loader overhead is the
    bottleneck; the GPU is being kept busy but the host can't feed it any
    faster. Most useful for ``cat:test`` / ``cat:gpu`` spans.
    """
    offenders: list[tuple[AgentEvent, WindowStats]] = []
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL:
            continue
        if ev.category not in {"test", "gpu", "gpu_query"}:
            continue
        if ev.ts_end_us is None or ev.duration_us is None:
            continue
        if ev.duration_us < 5 * 1_000_000:
            continue
        ws = reader.window_stats(ev.ts_start_us, ev.ts_end_us)
        if ws.n_gpu_samples < MIN_SAMPLES_FOR_VERDICT or ws.n_system_samples < MIN_SAMPLES_FOR_VERDICT:
            continue
        if ws.gpu_util_mean is None or ws.cpu_util_mean is None:
            continue
        if (
            ws.gpu_util_mean >= HOST_BOUND_GPU_UTIL_PCT
            and ws.cpu_util_mean >= HOST_BOUND_CPU_UTIL_PCT
        ):
            offenders.append((ev, ws))

    if not offenders:
        return
    anchors = [
        Anchor(
            source=e.source,
            session_id=e.session_id,
            ts_us=e.ts_start_us,
            detail=(
                f"GPU {ws.gpu_util_mean:.0f}% / CPU {ws.cpu_util_mean:.0f}% — "
                f"{_short_cmd(e)}"
            ),
        )
        for e, ws in offenders[:8]
    ]
    yield Hint(
        detector="host_bound_gpu",
        category="gpu",
        title=f"{len(offenders)} GPU run(s) saturated GPU *and* host CPU",
        severity=Severity.MEDIUM,
        occurrences=len(offenders),
        anchors=anchors,
        remediation=(
            "GPU and CPU are both near 100% — likely host-side overhead "
            "(data loader, kernel launch latency, single-threaded pre/post). "
            "Look for opportunities to launch fewer/larger kernels, move work "
            "off the critical thread, or pipeline host/device work."
        ),
    )


# ---------------------------------------------------------------------------


def detect_vram_pressure(
    events: Iterable[AgentEvent],
    reader: TelemetryReader,
) -> Iterator[Hint]:
    """VRAM used > VRAM_PRESSURE_PCT of total during a span."""
    offenders: list[tuple[AgentEvent, WindowStats, float]] = []
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.ts_end_us is None:
            continue
        ws = reader.window_stats(ev.ts_start_us, ev.ts_end_us)
        if not ws.vram_used_max_mb or not ws.vram_total_mb:
            continue
        pct = ws.vram_used_max_mb / ws.vram_total_mb * 100
        if pct >= VRAM_PRESSURE_PCT:
            offenders.append((ev, ws, pct))
    if not offenders:
        return
    # Group by session for cleaner output.
    by_session: dict[tuple[str, str], list[tuple[AgentEvent, WindowStats, float]]] = (
        collections.defaultdict(list)
    )
    for ev, ws, pct in offenders:
        by_session[(ev.source, ev.session_id)].append((ev, ws, pct))
    for (source, sid), evs in by_session.items():
        worst = max(evs, key=lambda x: x[2])
        anchors = [
            Anchor(
                source=source,
                session_id=sid,
                ts_us=e.ts_start_us,
                detail=(
                    f"VRAM {ws.vram_used_max_mb:.0f}/{ws.vram_total_mb:.0f}MB "
                    f"({pct:.0f}%) — {_short_cmd(e)}"
                ),
            )
            for e, ws, pct in sorted(evs, key=lambda x: -x[2])[:5]
        ]
        yield Hint(
            detector="vram_pressure",
            category="gpu",
            title=(
                f"{len(evs)} span(s) hit ≥{VRAM_PRESSURE_PCT:.0f}% VRAM "
                f"(peak {worst[2]:.0f}%)"
            ),
            severity=Severity.HIGH if worst[2] >= 98 else Severity.MEDIUM,
            occurrences=len(evs),
            anchors=anchors,
            remediation=(
                "Near-OOM means the next allocation can fail or trigger paging. "
                "Reduce batch size / activation footprint, free intermediates "
                "before allocating the next stage, or switch to a model that "
                "fits with headroom."
            ),
        )


# --- helpers ---------------------------------------------------------------


def _short_cmd(ev: AgentEvent) -> str:
    if not isinstance(ev.payload, dict):
        return ev.name
    inp = ev.payload.get("input")
    if isinstance(inp, dict):
        for k in ("command", "cmd"):
            v = inp.get(k)
            if isinstance(v, str):
                return v if len(v) <= 80 else v[:80] + "…"
    return ev.name
