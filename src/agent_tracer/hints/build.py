"""Build-pattern detectors that need only the event stream.

These complement the agent-side detectors in ``hints.agent`` by looking at
*what* the agent runs rather than how often any one thing repeats. They
surface common waste in the actual CPU/GPU work driven by the agent:
repeated cold rebuilds, expunge chains, and SSH overhead.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints.types import Anchor, Hint, Severity

REBUILD_MIN_PER_DAY = 4
EXPUNGE_CHAIN_MIN = 2
SSH_OVERHEAD_MIN_S = 30.0

_NINJA_TARGET_RE = re.compile(r"\bninja(?:\s+-[A-Za-z0-9]+(?:\s+\S+)?)*\s+(\S+)")
_EXPUNGE_RE = re.compile(r"\b([\w.+-]+)\+expunge\b")
_DIST_RE = re.compile(r"\b([\w.+-]+)\+dist\b")


# ---------------------------------------------------------------------------


def run_all_build(events: Iterable[AgentEvent]) -> list[Hint]:
    materialized = list(events)
    hints: list[Hint] = []
    hints.extend(detect_repeated_rebuilds(materialized))
    hints.extend(detect_expunge_chain(materialized))
    hints.extend(detect_ssh_overhead(materialized))
    return hints


# ---------------------------------------------------------------------------


def detect_repeated_rebuilds(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """Same ``ninja <target>`` run ≥REBUILD_MIN_PER_DAY times in one calendar day.

    Aggregates across sessions — if you ran the same target many times in a day
    across multiple Claude/Codex sessions, that's still a ccache opportunity.
    """
    by_day_target: dict[tuple[str, str], list[AgentEvent]] = collections.defaultdict(list)
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.category != "build":
            continue
        cmd = _extract_command(ev)
        target = _ninja_target(cmd)
        if not target:
            continue
        day = _utc_day(ev.ts_start_us)
        by_day_target[(day, target)].append(ev)

    for (day, target), evs in by_day_target.items():
        if len(evs) < REBUILD_MIN_PER_DAY:
            continue
        total_wall_us = sum(e.duration_us or 0 for e in evs)
        anchors = [
            Anchor(
                source=e.source,
                session_id=e.session_id,
                ts_us=e.ts_start_us,
                detail=f"ninja {target}  ({(e.duration_us or 0) / 1_000_000:.1f}s)",
            )
            for e in evs[:8]
        ]
        sev = (
            Severity.HIGH
            if total_wall_us > 600_000_000
            else (Severity.MEDIUM if len(evs) >= 8 else Severity.LOW)
        )
        yield Hint(
            detector="repeated_rebuilds",
            category="build",
            title=(
                f"`ninja {target}` ran {len(evs)}× on {day} "
                f"({total_wall_us / 1_000_000:.0f}s total wall)"
            ),
            severity=sev,
            occurrences=len(evs),
            est_wall_saved_s=total_wall_us / 1_000_000 * 0.5,  # ccache often saves ~50%
            anchors=anchors,
            remediation=(
                "Enable ccache/sccache for the build, or stop running expunge "
                "between iterations. Each full rebuild here costs "
                f"{total_wall_us / len(evs) / 1_000_000:.0f}s on average."
            ),
            evidence={"day": day, "target": target, "wall_s": total_wall_us / 1_000_000},
        )


# ---------------------------------------------------------------------------


def detect_expunge_chain(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """`<target>+expunge` immediately followed by `<target>+dist` ≥N times.

    TheRock's standard "rebuild from clean" pattern. Worth flagging because
    expunge often isn't actually necessary between iterations — a no-op
    rebuild post-expunge can take many minutes.
    """
    by_session: dict[tuple[str, str], list[tuple[str, AgentEvent]]] = collections.defaultdict(list)
    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.category != "build":
            continue
        cmd = _extract_command(ev)
        m_e = _EXPUNGE_RE.search(cmd)
        m_d = _DIST_RE.search(cmd)
        if m_e:
            by_session[(ev.source, ev.session_id)].append(("expunge", ev))
        elif m_d:
            by_session[(ev.source, ev.session_id)].append(("dist", ev))

    for (source, sid), seq in by_session.items():
        chains: list[tuple[AgentEvent, AgentEvent]] = []
        for i in range(len(seq) - 1):
            if seq[i][0] == "expunge" and seq[i + 1][0] == "dist":
                chains.append((seq[i][1], seq[i + 1][1]))
        if len(chains) < EXPUNGE_CHAIN_MIN:
            continue
        total_us = sum(
            ((e_evt.duration_us or 0) + (d_evt.duration_us or 0))
            for e_evt, d_evt in chains
        )
        anchors = [
            Anchor(
                source=source,
                session_id=sid,
                ts_us=e.ts_start_us,
                detail=(
                    f"expunge {(e.duration_us or 0) / 1_000_000:.0f}s "
                    f"→ dist {(d.duration_us or 0) / 1_000_000:.0f}s"
                ),
            )
            for e, d in chains[:6]
        ]
        yield Hint(
            detector="expunge_chain",
            category="build",
            title=(
                f"{len(chains)} `<target>+expunge && <target>+dist` chain(s) "
                f"({total_us / 1_000_000:.0f}s total)"
            ),
            severity=Severity.MEDIUM if total_us > 60_000_000 else Severity.LOW,
            occurrences=len(chains),
            est_wall_saved_s=total_us / 1_000_000 * 0.7,  # most expunges are unnecessary
            anchors=anchors,
            remediation=(
                "Only expunge when you've actually changed the build configuration. "
                "For source-only changes, `ninja <target>+dist` alone is enough."
            ),
        )


# ---------------------------------------------------------------------------


def detect_ssh_overhead(events: Iterable[AgentEvent]) -> Iterator[Hint]:
    """Sum of ssh/scp/sshpass/rsync wall-time per session over a threshold."""
    by_session: dict[tuple[str, str], dict[str, int]] = collections.defaultdict(
        lambda: {"wall_us": 0, "count": 0}
    )
    samples: dict[tuple[str, str], list[AgentEvent]] = collections.defaultdict(list)

    for ev in events:
        if ev.kind != EventKind.TOOL_CALL or ev.category != "network":
            continue
        cmd = _extract_command(ev)
        if not any(cmd.lstrip().startswith(x) for x in ("ssh ", "sshpass ", "scp ", "rsync ")):
            continue
        key = (ev.source, ev.session_id)
        d = ev.duration_us or 0
        by_session[key]["wall_us"] += d
        by_session[key]["count"] += 1
        samples[key].append(ev)

    for key, agg in by_session.items():
        wall_s = agg["wall_us"] / 1_000_000
        if wall_s < SSH_OVERHEAD_MIN_S:
            continue
        source, sid = key
        # Anchors: the slowest individual ssh invocations.
        worst = sorted(samples[key], key=lambda e: -(e.duration_us or 0))[:6]
        anchors = [
            Anchor(
                source=source,
                session_id=sid,
                ts_us=e.ts_start_us,
                detail=(
                    f"{(e.duration_us or 0) / 1_000_000:.1f}s — "
                    f"{_truncate(_extract_command(e), 100)}"
                ),
            )
            for e in worst
        ]
        sev = (
            Severity.HIGH
            if wall_s > 300
            else (Severity.MEDIUM if wall_s > 120 else Severity.LOW)
        )
        yield Hint(
            detector="ssh_overhead",
            category="build",
            title=(
                f"{agg['count']} ssh/scp invocation(s) totaling {wall_s:.0f}s of wall-clock"
            ),
            severity=sev,
            occurrences=agg["count"],
            est_wall_saved_s=wall_s * 0.5,
            anchors=anchors,
            remediation=(
                "Use SSH ControlMaster to multiplex connections "
                "(~/.ssh/config: ControlPath/ControlPersist), or hold a tmux "
                "session open on the remote host instead of reconnecting per "
                "command. For file copies, batch with rsync rather than scp loops."
            ),
        )


# --- helpers ---------------------------------------------------------------


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
        v = parsed[0].get("cmd")
        if isinstance(v, str):
            return v
    return ""


def _ninja_target(cmd: str) -> str | None:
    m = _NINJA_TARGET_RE.search(cmd)
    return m.group(1) if m else None


def _utc_day(ts_us: int) -> str:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC).date().isoformat()


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"
