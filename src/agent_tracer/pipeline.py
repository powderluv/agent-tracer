"""Shared event-streaming pipeline used by ``build``, ``stats``, ``hints``.

Reads JSONLs, normalizes per source, applies category tagging, filters by
time window / session / project slug, and yields the resulting ``AgentEvent``
stream. P5 keeps the pipeline stateless — when LanceDB lands the pipeline
will read from the materialized store instead, but the iteration interface
stays the same.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from agent_tracer.categorize import categorize_in_place
from agent_tracer.events import AgentEvent
from agent_tracer.normalize import normalize_claude_session, normalize_codex_session
from agent_tracer.parsers import claude, codex
from agent_tracer.timeutil import iso_to_us


@dataclass(slots=True)
class Filters:
    since_us: int | None = None
    until_us: int | None = None
    sources: frozenset[str] = frozenset({"claude", "codex"})
    sessions: frozenset[str] | None = None
    project_slug: str | None = None

    @classmethod
    def parse(
        cls,
        *,
        since: str | None = None,
        until: str | None = None,
        sources: Iterable[str] | None = None,
        sessions: Iterable[str] | None = None,
        project_slug: str | None = None,
    ) -> Filters:
        return cls(
            since_us=_parse_when(since),
            until_us=_parse_when(until),
            sources=frozenset(sources) if sources else frozenset({"claude", "codex"}),
            sessions=frozenset(sessions) if sessions else None,
            project_slug=project_slug,
        )

    def in_window(self, ev_ts: int) -> bool:
        if self.since_us is not None and ev_ts < self.since_us:
            return False
        return not (self.until_us is not None and ev_ts >= self.until_us)


def _parse_when(s: str | None) -> int | None:
    if not s:
        return None
    if len(s) == 10:
        s = s + "T00:00:00Z"
    return iso_to_us(s)


def iter_events(filters: Filters, *, categorize: bool = True) -> Iterator[AgentEvent]:
    """Yield filtered, normalized, optionally-categorized events from disk.

    Order is per-file (file emission order is session-id sorted within a
    project, then by start-time across sources only after de-interleaving —
    if you need strict global time order, sort downstream).
    """
    if "claude" in filters.sources:
        yield from _iter_claude(filters, categorize)
    if "codex" in filters.sources:
        yield from _iter_codex(filters, categorize)


def _iter_claude(filters: Filters, categorize_flag: bool) -> Iterator[AgentEvent]:
    for sf in claude.iter_session_files(project_slug=filters.project_slug):
        if filters.sessions is not None and sf.session_id not in filters.sessions:
            continue
        for ev in normalize_claude_session(
            (rec for _, rec in claude.iter_raw_records(sf.path)),
            source_session_id=sf.session_id,
            source_agent_id=sf.agent_id,
        ):
            if not filters.in_window(ev.ts_start_us):
                continue
            if categorize_flag:
                categorize_in_place(ev)
            yield ev


def _iter_codex(filters: Filters, categorize_flag: bool) -> Iterator[AgentEvent]:
    for sf in codex.iter_session_files():
        if filters.sessions is not None and sf.session_id not in filters.sessions:
            continue
        for ev in normalize_codex_session(
            (rec for _, rec in codex.iter_raw_records(sf.path)),
            source_session_id=sf.session_id,
        ):
            if not filters.in_window(ev.ts_start_us):
                continue
            if (
                filters.project_slug
                and ev.cwd
                and claude.project_slug_for_cwd(ev.cwd) != filters.project_slug
            ):
                continue
            if categorize_flag:
                categorize_in_place(ev)
            yield ev
