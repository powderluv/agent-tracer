"""Emit Chrome Trace Event JSON (loadable in ``ui.perfetto.dev``).

Format reference: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU

Mapping rules (from PLAN.md):

* ``pid`` per session: ``<source>:<session_id_short>``.
* ``tid`` per lane within a session: main = 0, subagents = 1+ in
  discovery order. The lane name is taken from the subagent's ``slug``
  if we have one, else the ``agent_id``.
* Tool calls → ``ph:'X'`` (complete) with ``dur``. Even zero-duration
  spans get ``dur=1`` so they remain selectable.
* Token-bearing assistant messages → ``ph:'C'`` (counter) on the
  session's main tid.
* Everything else (user turns, thinking, errors, progress) → ``ph:'i'``
  instants scoped to the thread.
* ``cat`` carries the content-miner tag (filled in P3) or the event
  kind as a fallback so Perfetto's category filter is useful from day one.
* ``args`` carries inputs/results/payloads (already truncated by the
  normalizer).

This module produces a Python dict; callers serialize with ``json.dump``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from agent_tracer.events import AgentEvent, EventKind


@dataclass(slots=True)
class _LaneKey:
    pid: int
    tid: int


class TraceBuilder:
    """Accumulator that assigns pids/tids deterministically and emits events."""

    def __init__(self) -> None:
        self._processes: dict[tuple[str, str], int] = {}
        self._next_pid = 1
        # (pid, agent_id_or_None) -> tid; main lane is always tid 0.
        self._lanes: dict[tuple[int, str | None], int] = {}
        # Friendly lane names captured from the first event we see on a lane.
        self._lane_names: dict[tuple[int, int], str] = {}
        self._events: list[dict[str, Any]] = []
        # We must emit metadata events *before* data events for some viewers;
        # we'll add them at finalize() time using the dicts above.

    # ----- public ---------------------------------------------------------

    def add(self, ev: AgentEvent) -> None:
        pid = self._pid_for(ev)
        tid = self._tid_for(pid, ev)
        if ev.kind == EventKind.TOOL_CALL and ev.ts_end_us is not None:
            self._add_complete(ev, pid, tid)
        elif ev.kind == EventKind.ASSISTANT_MSG:
            self._add_token_counters(ev, pid, tid)
        else:
            self._add_instant(ev, pid, tid)

    def finalize(self) -> dict[str, Any]:
        meta: list[dict[str, Any]] = []
        for (source, session_id), pid in self._processes.items():
            meta.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"name": f"{source}:{session_id[:8]}"},
                }
            )
            meta.append(
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": pid},
                }
            )
        for (pid, _agent_id), tid in self._lanes.items():
            name = self._lane_names.get((pid, tid), "main" if tid == 0 else f"tid-{tid}")
            meta.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
            meta.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": tid},
                }
            )
        return {
            "displayTimeUnit": "ms",
            "traceEvents": meta + self._events,
        }

    # ----- internals ------------------------------------------------------

    def _pid_for(self, ev: AgentEvent) -> int:
        key = (ev.source, ev.session_id)
        if key not in self._processes:
            self._processes[key] = self._next_pid
            self._next_pid += 1
        return self._processes[key]

    def _tid_for(self, pid: int, ev: AgentEvent) -> int:
        # Main lane uses agent_id = None → tid 0; each distinct subagent gets
        # an incrementally-allocated tid within the session.
        is_main = ev.agent_id is None or ev.source != "claude"
        lane_key = (pid, None if is_main else ev.agent_id)
        if lane_key not in self._lanes:
            # Count existing tids for this pid to allocate sequentially.
            existing = [tid for (p, _a), tid in self._lanes.items() if p == pid]
            if is_main:
                tid = 0
            else:
                tid = (max(existing) + 1) if existing else 1
                if tid == 0:
                    tid = 1
            self._lanes[lane_key] = tid
        tid = self._lanes[lane_key]
        # Capture a friendly name if we can derive one from this event.
        if (pid, tid) not in self._lane_names:
            self._lane_names[(pid, tid)] = self._lane_name(ev, tid)
        return tid

    @staticmethod
    def _lane_name(ev: AgentEvent, tid: int) -> str:
        if tid == 0 or ev.agent_id is None:
            return "main"
        # ``payload`` might carry slug under "slug" if the parser ever stores
        # it; for now the agent_id short prefix is the fallback.
        slug = ev.payload.get("slug") if isinstance(ev.payload, dict) else None
        if isinstance(slug, str) and slug:
            return slug
        return ev.agent_id[:12]

    # ----- event encoders -------------------------------------------------

    def _add_complete(self, ev: AgentEvent, pid: int, tid: int) -> None:
        dur = max(1, (ev.ts_end_us or ev.ts_start_us) - ev.ts_start_us)
        name = ev.name
        if ev.subagent_type:
            name = f"{name}:{ev.subagent_type}"
        self._events.append(
            {
                "name": name,
                "ph": "X",
                "pid": pid,
                "tid": tid,
                "ts": ev.ts_start_us,
                "dur": dur,
                "cat": ev.category or "tool",
                "args": self._args(ev),
            }
        )
        if ev.is_error:
            self._events.append(
                {
                    "name": f"!{name}",
                    "ph": "i",
                    "pid": pid,
                    "tid": tid,
                    "ts": ev.ts_start_us,
                    "s": "t",
                    "cat": "error",
                    "args": {"tool_use_id": ev.tool_use_id},
                }
            )

    def _add_token_counters(self, ev: AgentEvent, pid: int, tid: int) -> None:
        # Counters live on the main lane regardless of which subagent
        # produced them so token timelines per session are easy to read.
        main_tid = 0
        counters = {
            "in": ev.tokens_input or 0,
            "out": ev.tokens_output or 0,
            "cache_r": ev.cache_read or 0,
            "cache_w": ev.cache_create or 0,
        }
        if any(counters.values()):
            self._events.append(
                {
                    "name": "tokens",
                    "ph": "C",
                    "pid": pid,
                    "tid": main_tid,
                    "ts": ev.ts_start_us,
                    "cat": "tokens",
                    "args": counters,
                }
            )

    def _add_instant(self, ev: AgentEvent, pid: int, tid: int) -> None:
        self._events.append(
            {
                "name": ev.name,
                "ph": "i",
                "pid": pid,
                "tid": tid,
                "ts": ev.ts_start_us,
                "s": "t",  # thread-scoped
                "cat": ev.category or str(ev.kind.value),
                "args": self._args(ev),
            }
        )

    @staticmethod
    def _args(ev: AgentEvent) -> dict[str, Any]:
        args: dict[str, Any] = {}
        if ev.model:
            args["model"] = ev.model
        if ev.cwd:
            args["cwd"] = ev.cwd
        if ev.tool_use_id:
            args["tool_use_id"] = ev.tool_use_id
        if ev.subagent_type:
            args["subagent_type"] = ev.subagent_type
        if ev.payload:
            args.update(ev.payload)
        return args


def build_trace(events: Iterable[AgentEvent]) -> dict[str, Any]:
    """Convenience wrapper: feed events through a fresh :class:`TraceBuilder`."""
    b = TraceBuilder()
    for ev in events:
        b.add(ev)
    return b.finalize()
