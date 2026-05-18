"""Optimization-hint detectors.

Three groups:

* ``hints.agent``  — pure event analysis (redundant Reads, repeated Bash,
                     compaction frequency, hot tool time).
* ``hints.build``  — event-only build patterns (repeated rebuilds, expunge
                     chains, ssh/scp overhead).
* ``hints.gpu``    — telemetry-correlated (GPU-idle build, host-bound GPU,
                     VRAM pressure). Silently no-ops without telemetry.

``run_all`` runs every detector group with the events it has. Pass a
``TelemetryReader`` to enable the gpu group; omit for event-only mode.
"""

from __future__ import annotations

from collections.abc import Iterable

from agent_tracer.events import AgentEvent
from agent_tracer.hints.agent import (
    detect_compaction_frequency,
    detect_hot_tool_time,
    detect_redundant_reads,
    detect_repeated_bash,
)
from agent_tracer.hints.agent import (
    run_all as run_all_agent,
)
from agent_tracer.hints.build import (
    detect_expunge_chain,
    detect_repeated_rebuilds,
    detect_ssh_overhead,
    run_all_build,
)
from agent_tracer.hints.gpu import (
    detect_gpu_idle_build,
    detect_host_bound_gpu,
    detect_vram_pressure,
    run_all_gpu,
)
from agent_tracer.hints.types import Anchor, Hint, Severity
from agent_tracer.telemetry.reader import TelemetryReader


def run_all(
    events: Iterable[AgentEvent],
    *,
    telemetry: TelemetryReader | None = None,
) -> list[Hint]:
    """Run every detector group. Materializes the event iterator once."""
    materialized = list(events)
    hints: list[Hint] = []
    hints.extend(run_all_agent(materialized))
    hints.extend(run_all_build(materialized))
    if telemetry is not None:
        hints.extend(run_all_gpu(materialized, telemetry))
    # Stable severity sort: HIGH first, then MEDIUM, then LOW.
    sev_rank = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}
    hints.sort(key=lambda h: (sev_rank[h.severity], -h.occurrences))
    return hints


__all__ = [
    "Anchor",
    "Hint",
    "Severity",
    "TelemetryReader",
    "detect_compaction_frequency",
    "detect_expunge_chain",
    "detect_gpu_idle_build",
    "detect_host_bound_gpu",
    "detect_hot_tool_time",
    "detect_redundant_reads",
    "detect_repeated_bash",
    "detect_repeated_rebuilds",
    "detect_ssh_overhead",
    "detect_vram_pressure",
    "run_all",
    "run_all_agent",
    "run_all_build",
    "run_all_gpu",
]
