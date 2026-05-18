"""One-shot markdown report combining stats + hints + extras.

Designed for the "show me everything you can tell from my data" use case:
no telemetry needed, runs over every session on this machine by default.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable
from datetime import UTC, datetime
from io import StringIO

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.hints import Hint, run_all
from agent_tracer.stats import StatsReport, compute_stats


def _utc_day(ts_us: int) -> str:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC).date().isoformat()


def _iso(ts_us: int) -> str:
    return datetime.fromtimestamp(ts_us / 1_000_000, tz=UTC).isoformat(timespec="seconds")


def _humanize_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"


def _humanize_int(n: int) -> str:
    return f"{n:,}"


def generate_report(events: Iterable[AgentEvent], *, title: str = "agent-tracer report") -> str:
    """Materialize events once, compute everything, format markdown."""
    materialized = list(events)
    stats = compute_stats(materialized)
    hints = run_all(materialized)  # no telemetry — gpu detectors silently skipped
    extras = _compute_extras(materialized)
    return _format(stats, hints, extras, title=title)


# --- extras (things stats.py doesn't already compute) ---------------------


def _compute_extras(events: list[AgentEvent]) -> dict[str, object]:
    file_reads: collections.Counter = collections.Counter()
    file_edits: collections.Counter = collections.Counter()
    cwd_dist: collections.Counter = collections.Counter()
    subagent_types: collections.Counter = collections.Counter()
    activity_by_day: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"events": 0, "tool_calls": 0, "sessions": set()}  # type: ignore[dict-item]
    )
    user_prompts = 0
    user_rejections = 0
    thinking_blocks = 0
    thinking_text_len_total = 0
    ts_min = None
    ts_max = None
    error_events = 0
    sessions_by_source: collections.Counter = collections.Counter()

    for ev in events:
        ts_min = ev.ts_start_us if ts_min is None else min(ts_min, ev.ts_start_us)
        ts_max = ev.ts_start_us if ts_max is None else max(ts_max, ev.ts_start_us)
        day = _utc_day(ev.ts_start_us)
        d = activity_by_day[day]
        d["events"] += 1
        d["sessions"].add(f"{ev.source}:{ev.session_id}")  # type: ignore[union-attr]
        if ev.kind == EventKind.TOOL_CALL:
            d["tool_calls"] += 1
            if ev.cwd:
                cwd_dist[ev.cwd] += 1
            if ev.subagent_type:
                subagent_types[ev.subagent_type] += 1
            if ev.name == "Read":
                p = _file_path(ev)
                if p:
                    file_reads[p] += 1
            elif ev.name in ("Edit", "Write", "MultiEdit"):
                p = _file_path(ev)
                if p:
                    file_edits[p] += 1
            elif ev.name == "apply_patch":
                # Codex apply_patch lists changed files in payload['changes'].
                changes = ev.payload.get("changes") if isinstance(ev.payload, dict) else None
                if isinstance(changes, list):
                    for fp in changes:
                        if isinstance(fp, str):
                            file_edits[fp] += 1
        elif ev.kind == EventKind.USER_TURN:
            user_prompts += 1
        elif ev.kind == EventKind.ERROR:
            error_events += 1
            if ev.name == "user_rejected":
                user_rejections += 1
        elif ev.kind == EventKind.THINKING:
            thinking_blocks += 1
            if isinstance(ev.payload, dict):
                tlen = ev.payload.get("text_len")
                if isinstance(tlen, int):
                    thinking_text_len_total += tlen

    for ev in events:
        sessions_by_source[ev.source] += 0  # initialize keys
    # Rebuild sessions_by_source from stats so we don't double-count.
    return {
        "ts_min_us": ts_min,
        "ts_max_us": ts_max,
        "activity_by_day": {
            day: {
                "events": d["events"],
                "tool_calls": d["tool_calls"],
                "sessions": len(d["sessions"]),  # type: ignore[arg-type]
            }
            for day, d in sorted(activity_by_day.items())
        },
        "file_reads": file_reads,
        "file_edits": file_edits,
        "cwd_dist": cwd_dist,
        "subagent_types": subagent_types,
        "user_prompts": user_prompts,
        "user_rejections": user_rejections,
        "thinking_blocks": thinking_blocks,
        "thinking_text_len_total": thinking_text_len_total,
        "error_events": error_events,
    }


def _file_path(ev: AgentEvent) -> str | None:
    if not isinstance(ev.payload, dict):
        return None
    inp = ev.payload.get("input")
    if isinstance(inp, dict):
        for k in ("file_path", "path", "file"):
            v = inp.get(k)
            if isinstance(v, str):
                return v
    return None


# --- formatting -----------------------------------------------------------


def _format(
    stats: StatsReport,
    hints: list[Hint],
    extras: dict[str, object],
    *,
    title: str,
) -> str:
    out = StringIO()
    p = out.write

    p(f"# {title}\n\n")
    p(f"_Generated_: {_iso(int(datetime.now(tz=UTC).timestamp() * 1_000_000))}\n\n")

    # --- Overview ---
    ts_min = extras["ts_min_us"]
    ts_max = extras["ts_max_us"]
    span_s = ((ts_max - ts_min) / 1_000_000) if (ts_min and ts_max) else 0  # type: ignore[operator]
    src_counts = collections.Counter(src for (src, _) in stats.sessions)
    p("## Overview\n\n")
    p(f"- **Time span**: {_iso(ts_min)} → {_iso(ts_max)}  ({_humanize_seconds(span_s)})\n")  # type: ignore[arg-type]
    p(
        f"- **Sessions**: {stats.total_sessions} "
        f"({src_counts['claude']} Claude, {src_counts['codex']} Codex)\n"
    )
    total_tool_calls = sum(s.tool_calls for s in stats.sessions.values())
    total_tool_wall_us = sum(s.tool_wall_us for s in stats.sessions.values())
    p(
        f"- **Tool calls**: {_humanize_int(total_tool_calls)} "
        f"({_humanize_seconds(total_tool_wall_us / 1_000_000)} total wall)\n"
    )
    p(f"- **User prompts**: {_humanize_int(extras['user_prompts'])}\n")  # type: ignore[arg-type]
    p(
        f"- **Tokens (in/out)**: "
        f"{_humanize_int(sum(s.tokens_input for s in stats.sessions.values()))} in / "
        f"{_humanize_int(sum(s.tokens_output for s in stats.sessions.values()))} out\n"
    )
    total_cache_r = sum(s.cache_read for s in stats.sessions.values())
    total_cache_w = sum(s.cache_create for s in stats.sessions.values())
    if total_cache_r + total_cache_w:
        rate = total_cache_r / (total_cache_r + total_cache_w) * 100
        p(
            f"- **Prompt cache**: {_humanize_int(total_cache_r)} read / "
            f"{_humanize_int(total_cache_w)} created  ({rate:.1f}% hit rate)\n"
        )
    p(f"- **Thinking blocks**: {_humanize_int(extras['thinking_blocks'])}\n")  # type: ignore[arg-type]
    p(f"- **Error events**: {_humanize_int(extras['error_events'])} "  # type: ignore[arg-type]
      f"(of which {extras['user_rejections']} were user-denied tool calls)\n")
    p("\n")

    # --- Activity by day ---
    p("## Activity by day\n\n")
    p("| Day | Sessions | Events | Tool calls |\n")
    p("|---|---:|---:|---:|\n")
    activity = extras["activity_by_day"]  # type: ignore[assignment]
    for day, d in activity.items():  # type: ignore[attr-defined]
        p(f"| {day} | {d['sessions']} | {_humanize_int(d['events'])} | "
          f"{_humanize_int(d['tool_calls'])} |\n")
    p("\n")

    # --- Per-session ---
    p("## Per-session\n\n")
    p("| Session | Source | Span | Wall | Tools | Tok in | Tok out | Cache hit |\n")
    p("|---|---|---|---:|---:|---:|---:|---:|\n")
    for s in sorted(stats.sessions.values(), key=lambda s: s.ts_start_us):
        hit = "—" if s.cache_hit_rate is None else f"{s.cache_hit_rate * 100:.0f}%"
        p(
            f"| `{s.session_id[:8]}` | {s.source} | "
            f"{_iso(s.ts_start_us)[:10]} | "
            f"{_humanize_seconds(s.wallclock_s)} | "
            f"{s.tool_calls} | {_humanize_int(s.tokens_input)} | "
            f"{_humanize_int(s.tokens_output)} | {hit} |\n"
        )
    p("\n")

    # --- Categories ---
    p("## Event categories\n\n")
    total_cat = sum(stats.overall_by_category.values()) or 1
    p("| Category | Count | Share |\n|---|---:|---:|\n")
    for cat, n in stats.overall_by_category.most_common():
        p(f"| {cat} | {_humanize_int(n)} | {n / total_cat * 100:.1f}% |\n")
    p("\n")

    # --- Top tools ---
    p("## Top tools — by count\n\n")
    p("| Tool | Count | Wall |\n|---|---:|---:|\n")
    for name, n in stats.overall_by_tool.most_common(20):
        wall = stats.overall_tool_wall_us.get(name, 0) / 1_000_000
        p(f"| {name} | {_humanize_int(n)} | {_humanize_seconds(wall)} |\n")
    p("\n")

    p("## Top tools — by wall-clock\n\n")
    p("| Tool | Wall | Count |\n|---|---:|---:|\n")
    by_wall = sorted(stats.overall_tool_wall_us.items(), key=lambda kv: -kv[1])
    for name, us in by_wall[:20]:
        count = stats.overall_by_tool[name]
        p(f"| {name} | {_humanize_seconds(us / 1_000_000)} | {_humanize_int(count)} |\n")
    p("\n")

    # --- Top commands ---
    p("## Top shell commands\n\n")
    p("| Command (signature) | Count |\n|---|---:|\n")
    for sig, n in stats.top_commands.most_common(25):
        p(f"| `{sig}` | {_humanize_int(n)} |\n")
    p("\n")

    # --- Top files ---
    file_reads = extras["file_reads"]  # type: ignore[assignment]
    file_edits = extras["file_edits"]  # type: ignore[assignment]
    if file_reads or file_edits:
        p("## Top files\n\n")
        if file_reads:
            p("### Most-Read files\n\n")
            p("| File | Reads |\n|---|---:|\n")
            for fp, n in file_reads.most_common(15):  # type: ignore[attr-defined]
                p(f"| `{_short(fp)}` | {_humanize_int(n)} |\n")
            p("\n")
        if file_edits:
            p("### Most-edited files (Edit/Write/MultiEdit/apply_patch)\n\n")
            p("| File | Edits |\n|---|---:|\n")
            for fp, n in file_edits.most_common(15):  # type: ignore[attr-defined]
                p(f"| `{_short(fp)}` | {_humanize_int(n)} |\n")
            p("\n")

    # --- Project distribution ---
    cwd_dist = extras["cwd_dist"]  # type: ignore[assignment]
    if cwd_dist:
        p("## Tool calls by cwd\n\n")
        p("| Cwd | Tool calls |\n|---|---:|\n")
        for cwd, n in cwd_dist.most_common(15):  # type: ignore[attr-defined]
            p(f"| `{cwd}` | {_humanize_int(n)} |\n")
        p("\n")

    # --- Subagent types ---
    sub_types = extras["subagent_types"]  # type: ignore[assignment]
    if sub_types:
        p("## Claude subagent dispatches\n\n")
        p("| subagent_type | Count |\n|---|---:|\n")
        for st, n in sub_types.most_common():  # type: ignore[attr-defined]
            p(f"| {st} | {_humanize_int(n)} |\n")
        p("\n")

    # --- Hints ---
    p("## Optimization hints\n\n")
    if not hints:
        p("_No hints fired._\n\n")
    else:
        for h in hints:
            tag = f"**[{h.severity.value.upper()}]**"
            savings = ""
            if h.est_wall_saved_s and h.est_wall_saved_s >= 1:
                savings = f"  _est. {_humanize_seconds(h.est_wall_saved_s)} saveable_"
            elif h.est_tokens_saved:
                savings = f"  _est. {_humanize_int(h.est_tokens_saved)} tokens saveable_"
            p(f"### {tag} {h.title}{savings}\n\n")
            p(
                f"*detector*: `{h.detector}` · *category*: `{h.category}` · "
                f"*occurrences*: {h.occurrences}\n\n"
            )
            if h.remediation:
                p(h.remediation + "\n\n")
            for a in h.anchors[:8]:
                p(f"- `{a.source}:{a.session_id[:8]}` @ {_iso(a.ts_us)} — {a.detail}\n")
            if len(h.anchors) > 8:
                p(f"- … +{len(h.anchors) - 8} more\n")
            p("\n")

    return out.getvalue()


def _short(p: str, n: int = 80) -> str:
    return p if len(p) <= n else "…" + p[-(n - 1):]
