"""Aggregate stats over an ``AgentEvent`` stream.

All functions take an iterable and stream once, holding O(unique-keys) state.
The CLI prints tables; these functions return plain dataclasses so they're
trivially testable.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from dataclasses import dataclass, field

from agent_tracer.events import AgentEvent, EventKind


@dataclass(slots=True)
class SessionSummary:
    source: str
    session_id: str
    ts_start_us: int
    ts_end_us: int
    tool_calls: int = 0
    user_turns: int = 0
    assistant_msgs: int = 0
    thinking_blocks: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    cache_read: int = 0
    cache_create: int = 0
    tool_wall_us: int = 0
    by_category: collections.Counter = field(default_factory=collections.Counter)
    by_tool: collections.Counter = field(default_factory=collections.Counter)

    @property
    def wallclock_s(self) -> float:
        return max(0, self.ts_end_us - self.ts_start_us) / 1_000_000

    @property
    def cache_hit_rate(self) -> float | None:
        denom = self.cache_read + self.cache_create
        return None if denom == 0 else self.cache_read / denom


@dataclass(slots=True)
class StatsReport:
    sessions: dict[tuple[str, str], SessionSummary] = field(default_factory=dict)
    overall_by_category: collections.Counter = field(default_factory=collections.Counter)
    overall_by_tool: collections.Counter = field(default_factory=collections.Counter)
    overall_tool_wall_us: dict[str, int] = field(default_factory=dict)
    top_commands: collections.Counter = field(default_factory=collections.Counter)

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_input + s.tokens_output for s in self.sessions.values())


def compute_stats(events: Iterable[AgentEvent]) -> StatsReport:
    report = StatsReport()
    overall_tool_wall: dict[str, int] = collections.defaultdict(int)

    for ev in events:
        key = (ev.source, ev.session_id)
        summary = report.sessions.get(key)
        if summary is None:
            summary = SessionSummary(
                source=ev.source,
                session_id=ev.session_id,
                ts_start_us=ev.ts_start_us,
                ts_end_us=ev.ts_end_us or ev.ts_start_us,
            )
            report.sessions[key] = summary
        summary.ts_start_us = min(summary.ts_start_us, ev.ts_start_us)
        summary.ts_end_us = max(summary.ts_end_us, ev.ts_end_us or ev.ts_start_us)

        if ev.category:
            summary.by_category[ev.category] += 1
            report.overall_by_category[ev.category] += 1

        if ev.kind == EventKind.TOOL_CALL:
            summary.tool_calls += 1
            summary.by_tool[ev.name] += 1
            report.overall_by_tool[ev.name] += 1
            dur = ev.duration_us or 0
            summary.tool_wall_us += dur
            overall_tool_wall[ev.name] += dur
            if ev.name in {"Bash", "exec_command"}:
                cmd = _extract_command(ev)
                if cmd:
                    report.top_commands[_command_signature(cmd)] += 1
        elif ev.kind == EventKind.USER_TURN:
            summary.user_turns += 1
        elif ev.kind == EventKind.ASSISTANT_MSG:
            summary.assistant_msgs += 1
            summary.tokens_input += ev.tokens_input or 0
            summary.tokens_output += ev.tokens_output or 0
            summary.cache_read += ev.cache_read or 0
            summary.cache_create += ev.cache_create or 0
        elif ev.kind == EventKind.THINKING:
            summary.thinking_blocks += 1

    report.overall_tool_wall_us = dict(overall_tool_wall)
    return report


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


def _command_signature(cmd: str) -> str:
    """First two whitespace tokens of the command, lowercased.

    Good enough for grouping: ``ninja clr+dist`` collapses with ``ninja
    clr+expunge``? No — ``clr+dist`` != ``clr+expunge``. That's intentional:
    we want to distinguish those two patterns since the build cost differs.
    """
    tokens = cmd.strip().split()
    if not tokens:
        return ""
    head = tokens[0]
    # Strip a sudo / sh -c / bash -lc wrapper so the verb is the real verb.
    if head in {"sudo", "time", "nice", "stdbuf"} and len(tokens) > 1:
        return _command_signature(" ".join(tokens[1:]))
    if head in {"bash", "sh", "zsh"} and len(tokens) > 2 and tokens[1] in {"-c", "-lc"}:
        return _command_signature(" ".join(tokens[2:]))
    if len(tokens) == 1:
        return head
    return f"{head} {tokens[1]}"
