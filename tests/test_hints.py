"""Agent-side hint detectors."""

from __future__ import annotations

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints.agent import (
    COMPACTION_MIN,
    REDUNDANT_READ_MIN,
    REPEATED_BASH_MIN,
    detect_compaction_frequency,
    detect_hot_tool_time,
    detect_redundant_reads,
    detect_repeated_bash,
    run_all,
)
from agent_tracer.hints.types import Severity


def _tool(name, ts_start, ts_end=None, source="claude", session_id="s1", **kw):
    return AgentEvent(
        source=source,
        session_id=session_id,
        kind=EventKind.TOOL_CALL,
        name=name,
        ts_start_us=ts_start,
        ts_end_us=ts_end if ts_end is not None else ts_start + 1,
        **kw,
    )


def _compaction(ts, source="claude", session_id="s1"):
    return AgentEvent(
        source=source,
        session_id=session_id,
        kind=EventKind.COMPACTION,
        name="compacted",
        ts_start_us=ts,
    )


def test_redundant_reads_fires_at_threshold() -> None:
    events = [
        _tool("Read", i * 1_000_000, payload={"input": {"file_path": "/x/foo.py"}})
        for i in range(REDUNDANT_READ_MIN)
    ]
    hints = list(detect_redundant_reads(events))
    assert len(hints) == 1
    h = hints[0]
    assert h.detector == "redundant_reads"
    assert h.occurrences == REDUNDANT_READ_MIN - 1
    assert "/x/foo.py" in h.anchors[0].detail


def test_redundant_reads_does_not_fire_below_threshold() -> None:
    events = [
        _tool("Read", i * 1_000_000, payload={"input": {"file_path": "/x/y"}})
        for i in range(REDUNDANT_READ_MIN - 1)
    ]
    assert list(detect_redundant_reads(events)) == []


def test_redundant_reads_skips_failed_reads() -> None:
    events = [
        _tool("Read", i, payload={"input": {"file_path": "/x"}}, is_error=True)
        for i in range(REDUNDANT_READ_MIN + 2)
    ]
    assert list(detect_redundant_reads(events)) == []


def test_repeated_bash_collapses_identical_commands() -> None:
    events = [
        _tool(
            "Bash",
            i,
            payload={"input": {"command": "ninja clr+dist"}},
        )
        for i in range(REPEATED_BASH_MIN)
    ]
    hints = list(detect_repeated_bash(events))
    assert len(hints) == 1
    h = hints[0]
    assert "ninja clr+dist" in h.anchors[0].detail


def test_repeated_bash_filters_trivial_commands() -> None:
    events = [
        _tool("Bash", i, payload={"input": {"command": "ls"}})
        for i in range(REPEATED_BASH_MIN + 2)
    ]
    assert list(detect_repeated_bash(events)) == []


def test_repeated_bash_severity_scales_with_wall_time() -> None:
    # 4 runs × 30s each = 90s "extra" after the first → HIGH.
    events = [
        _tool(
            "Bash",
            i * 100_000_000,
            (i * 100_000_000) + 30_000_000,
            payload={"input": {"command": "ninja clr+dist"}},
        )
        for i in range(4)
    ]
    h = next(iter(detect_repeated_bash(events)))
    assert h.severity == Severity.HIGH
    assert h.est_wall_saved_s is not None and h.est_wall_saved_s > 60


def test_compaction_frequency_fires_at_threshold() -> None:
    events = [_compaction(i * 1_000_000) for i in range(COMPACTION_MIN)]
    hints = list(detect_compaction_frequency(events))
    assert len(hints) == 1 and hints[0].occurrences == COMPACTION_MIN


def test_hot_tool_time_only_fires_when_concentrated() -> None:
    # Single tool dominating ≥50% wall-clock over the threshold.
    events = [
        _tool("Bash", 0, 40_000_000),       # 40s
        _tool("Read", 100_000_000, 100_100_000),  # 0.1s
    ]
    hints = list(detect_hot_tool_time(events))
    assert hints and hints[0].detector == "hot_tool_time"
    # Diversely-distributed work below the share thresholds should not fire.
    diverse = [
        _tool("Bash", 0, 12_000_000),
        _tool("Read", 13_000_000, 25_000_000),
        _tool("Edit", 26_000_000, 38_000_000),
        _tool("Grep", 39_000_000, 60_000_000),
    ]
    assert list(detect_hot_tool_time(diverse)) == []


def test_run_all_returns_sorted_hints() -> None:
    events = [
        _tool("Read", i, payload={"input": {"file_path": "/a"}})
        for i in range(REDUNDANT_READ_MIN)
    ]
    events += [_compaction(i * 1_000_000 + 50_000_000) for i in range(COMPACTION_MIN)]
    hints = run_all(events)
    severities = [h.severity for h in hints]
    # HIGH < MEDIUM < LOW in display order.
    assert severities == sorted(severities, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s.value])
