"""Optimization-hint detectors.

Each detector is a pure function over an iterable of ``AgentEvent`` that
returns ``Iterable[Hint]``. Hints carry concrete anchors (session id,
timestamp, command snippet) so the user can verify in the trace.

P5 ships agent-side detectors only (no telemetry needed). P6 adds
telemetry-driven detectors for GPU/CPU idle, VRAM pressure, etc.
"""

from __future__ import annotations

from agent_tracer.hints.agent import (
    detect_compaction_frequency,
    detect_hot_tool_time,
    detect_redundant_reads,
    detect_repeated_bash,
    run_all,
)
from agent_tracer.hints.types import Anchor, Hint, Severity

__all__ = [
    "Anchor",
    "Hint",
    "Severity",
    "detect_compaction_frequency",
    "detect_hot_tool_time",
    "detect_redundant_reads",
    "detect_repeated_bash",
    "run_all",
]
