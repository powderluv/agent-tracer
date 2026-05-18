"""Detectors that need only the event stream (no telemetry).

Each detector takes ``Iterable[AgentEvent]`` and yields ``Hint`` instances.
``run_all`` returns them concatenated. Detectors are independent — they
can be called individually for testing or composed via ``run_all``.

The min-evidence thresholds are tuned to suppress noise; tweak them in
one place if needed.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable, Iterator
from typing import Any

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints.types import Anchor, Hint, Severity

# Thresholds — keep them explicit so callers can read them.
REDUNDANT_READ_MIN = 3
REPEATED_BASH_MIN = 3
COMPACTION_MIN = 3
HOT_TOOL_MIN_WALL_S = 30.0


# ---------------------------------------------------------------------------


def run_all(events: Iterable[AgentEvent]) -> list[Hint]:
    """Materialize the event stream once and run every detector against it.

    Some detectors need a second pass; this helper consumes the iterator
    into a list so callers don't have to.
    """
    materialized = list(events)
    hints: list[Hint] = []
    hints.extend(detect_redundant_reads(materialized))
    hints.extend(detect_repeated_bash(materialized))
    hints.extend(detect_compaction_frequency(materialized))
    hints.extend(detect_hot_tool_time(materialized))
    # Severity-then-occurrences sort for stable display.
    hints.sort(key=lambda h: (_severity_rank(h.severity), -h.occurrences))
    return hints


def _severity_rank(s: Severity) -> int:
    return {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}[s]


# ---------------------------------------------------------------------------


def detect_redundant_reads(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """Same file Read ≥REDUNDANT_READ_MIN times in one session.

    Evidence: identical (session, file_path) seen N+ times. We only count
    successful Reads (is_error is not True); failed Reads typically retry
    legitimately and aren't useful to flag.
    """
    by_session: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    first_seen: dict[tuple[str, str, str], int] = {}

    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.name != "Read" or ev.is_error:
            continue
        path = _extract_path(ev)
        if not path:
            continue
        key = (ev.source, ev.session_id)
        by_session[key][path] += 1
        first_seen.setdefault((ev.source, ev.session_id, path), ev.ts_start_us)

    for (source, sid), counter in by_session.items():
        offenders = [(path, n) for path, n in counter.items() if n >= REDUNDANT_READ_MIN]
        if not offenders:
            continue
        offenders.sort(key=lambda x: -x[1])
        total_extra = sum(n - 1 for _, n in offenders)
        anchors = [
            Anchor(
                source=source,
                session_id=sid,
                ts_us=first_seen[(source, sid, path)],
                detail=f"{n}× Read {path}",
            )
            for path, n in offenders[:8]
        ]
        yield Hint(
            detector="redundant_reads",
            category="agent",
            title=f"{len(offenders)} file(s) Read {REDUNDANT_READ_MIN}+ times in one session",
            severity=Severity.MEDIUM if total_extra >= 10 else Severity.LOW,
            occurrences=total_extra,
            anchors=anchors,
            remediation=(
                "Hold the file's contents in context after the first Read; "
                "use Grep for known substrings instead of re-Reading whole files."
            ),
            evidence={"files": dict(offenders[:20])},
        )


# ---------------------------------------------------------------------------


def detect_repeated_bash(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """Identical Bash/exec_command run ≥REPEATED_BASH_MIN times in one session.

    Normalizes very-common ``cd``/``pwd``/``ls`` invocations out of the
    signature so we don't flag innocuous navigation.
    """
    by_session: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    first_seen: dict[tuple[str, str, str], int] = {}
    wall_us: dict[tuple[str, str, str], int] = collections.defaultdict(int)

    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.name not in {"Bash", "exec_command"}:
            continue
        cmd = _extract_command(ev).strip()
        if not cmd or _is_trivial(cmd):
            continue
        key = (ev.source, ev.session_id)
        by_session[key][cmd] += 1
        first_seen.setdefault((ev.source, ev.session_id, cmd), ev.ts_start_us)
        wall_us[(ev.source, ev.session_id, cmd)] += ev.duration_us or 0

    for (source, sid), counter in by_session.items():
        offenders = [(cmd, n) for cmd, n in counter.items() if n >= REPEATED_BASH_MIN]
        if not offenders:
            continue
        offenders.sort(key=lambda x: -x[1])
        anchors = []
        total_extra_wall_us = 0
        total_extra = 0
        for cmd, n in offenders[:8]:
            extra = n - 1
            total_extra += extra
            total_extra_wall_us += int((wall_us[(source, sid, cmd)] / n) * extra)
            anchors.append(
                Anchor(
                    source=source,
                    session_id=sid,
                    ts_us=first_seen[(source, sid, cmd)],
                    detail=f"{n}× {cmd[:120]}",
                )
            )
        severity = (
            Severity.HIGH
            if total_extra_wall_us > 60_000_000
            else (Severity.MEDIUM if total_extra >= 6 else Severity.LOW)
        )
        yield Hint(
            detector="repeated_bash",
            category="agent",
            title=f"{len(offenders)} shell command(s) re-run ≥{REPEATED_BASH_MIN}× in one session",
            severity=severity,
            occurrences=total_extra,
            est_wall_saved_s=total_extra_wall_us / 1_000_000,
            anchors=anchors,
            remediation=(
                "Cache the output (write to a tmpfile and Read it back); "
                "or move the command into a helper script the agent invokes once."
            ),
            evidence={"commands": dict(offenders[:20])},
        )


# ---------------------------------------------------------------------------


def detect_compaction_frequency(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """A session firing context-compaction >COMPACTION_MIN times is too long.

    Compaction fires when context fills up; it costs tokens and loses fidelity.
    We surface the count and suggest /clear or splitting into multiple sessions.
    """
    by_session: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for ev in events:
        if ev.kind == EventKind.COMPACTION:
            by_session[(ev.source, ev.session_id)].append(ev.ts_start_us)

    for (source, sid), tss in by_session.items():
        if len(tss) < COMPACTION_MIN:
            continue
        anchors = [
            Anchor(source=source, session_id=sid, ts_us=ts, detail=f"compaction #{i + 1}")
            for i, ts in enumerate(tss[:8])
        ]
        yield Hint(
            detector="compaction_frequency",
            category="agent",
            title=f"Session compacted {len(tss)} times",
            severity=Severity.HIGH if len(tss) >= 6 else Severity.MEDIUM,
            occurrences=len(tss),
            anchors=anchors,
            remediation=(
                "Run /clear earlier or split the work into separate sessions. "
                "Each compaction rewrites the context window — repeated firings mean "
                "the session is doing work the model can't keep in working memory."
            ),
        )


# ---------------------------------------------------------------------------


def detect_hot_tool_time(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """Surface tool kinds dominating wall-clock per session.

    Not a "bad" pattern on its own — but a useful pointer: if 60% of a
    session's tool time is ``Bash``, the optimization conversation starts
    with shell commands; if it's ``Read``, with reading less.
    """
    by_session: dict[tuple[str, str], collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    first_seen: dict[tuple[str, str], int] = {}
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL:
            continue
        key = (ev.source, ev.session_id)
        by_session[key][ev.name] += ev.duration_us or 0
        first_seen.setdefault(key, ev.ts_start_us)

    for (source, sid), counter in by_session.items():
        total_us = sum(counter.values())
        if total_us < HOT_TOOL_MIN_WALL_S * 1_000_000:
            continue
        top = counter.most_common(3)
        # Only flag when a single tool kind dominates (>50%) or top-3 share >80%.
        top_share = top[0][1] / total_us
        top3_share = sum(n for _, n in top) / total_us
        if top_share < 0.5 and top3_share < 0.8:
            continue
        detail = ", ".join(f"{name} {n/1_000_000:.1f}s" for name, n in top)
        yield Hint(
            detector="hot_tool_time",
            category="agent",
            title=f"Tool time concentrated: {top[0][0]} = {top_share:.0%} of {total_us/1_000_000:.0f}s",
            severity=Severity.LOW,
            occurrences=top[0][1] // 1_000_000,
            anchors=[
                Anchor(source=source, session_id=sid, ts_us=first_seen[(source, sid)], detail=detail)
            ],
            remediation=(
                f"{top[0][0]} dominates this session's tool wall-clock — "
                "look there first for batching opportunities or slow command patterns."
            ),
            evidence={"by_tool_seconds": {name: n / 1_000_000 for name, n in counter.most_common(10)}},
        )


# --- helpers ---------------------------------------------------------------


_TRIVIAL_CMDS = {"pwd", "ls", "cd", "echo", "true", "false", "clear", "date"}


def _is_trivial(cmd: str) -> bool:
    head = cmd.strip().split()
    return bool(head) and head[0] in _TRIVIAL_CMDS


def _extract_path(ev: AgentEvent) -> str:
    if not isinstance(ev.payload, dict):
        return ""
    inp = ev.payload.get("input")
    if isinstance(inp, dict):
        for k in ("file_path", "path", "file"):
            v = inp.get(k)
            if isinstance(v, str):
                return v
    return ""


def _extract_command(ev: AgentEvent) -> str:
    if not isinstance(ev.payload, dict):
        return ""
    inp = ev.payload.get("input")
    if isinstance(inp, str):
        return inp
    if isinstance(inp, dict):
        for k in ("command", "cmd", "script", "shell_command"):
            v = inp.get(k)
            if isinstance(v, str) and v:
                return v
    parsed = ev.payload.get("parsed_cmd") if isinstance(ev.payload, dict) else None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        v: Any = parsed[0].get("cmd")
        if isinstance(v, str):
            return v
    return ""
