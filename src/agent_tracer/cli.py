"""Argparse-based entry point.

Commands:
* ``discover``  — schema sanity report
* ``list``      — raw JSONL records (debug)
* ``build``     — Perfetto trace JSON
* ``stats``     — tables (per-session summary, tool histogram, top commands)
* ``hints``     — ranked optimization hints (markdown)
* ``sample``    — telemetry sampler daemon (rocm-smi/nvidia-smi/proc → LanceDB)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC
from pathlib import Path

from agent_tracer import __version__
from agent_tracer.events import EventKind
from agent_tracer.hints import Hint, run_all
from agent_tracer.parsers import claude, codex, discover
from agent_tracer.perfetto import TraceBuilder
from agent_tracer.pipeline import Filters, iter_events
from agent_tracer.stats import StatsReport, compute_stats

# --- subcommand: discover ---------------------------------------------------


def _cmd_discover(args: argparse.Namespace) -> int:
    claude_r = discover.scan_claude(file_limit=args.file_limit)
    codex_r = discover.scan_codex(file_limit=args.file_limit)
    print(discover.format_report(claude_r, codex_r))
    return 0


# --- subcommand: list -------------------------------------------------------


def _claude_records(limit: int | None) -> Iterable[dict]:
    n = 0
    for sf in claude.iter_session_files():
        for _, rec in claude.iter_raw_records(sf.path):
            rec.setdefault("_source", "claude")
            rec.setdefault("_file", str(sf.path))
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


def _codex_records(limit: int | None) -> Iterable[dict]:
    n = 0
    for sf in codex.iter_session_files():
        for _, rec in codex.iter_raw_records(sf.path):
            rec.setdefault("_source", "codex")
            rec.setdefault("_file", str(sf.path))
            yield rec
            n += 1
            if limit is not None and n >= limit:
                return


def _cmd_list(args: argparse.Namespace) -> int:
    records = (
        _claude_records(args.limit) if args.source == "claude" else _codex_records(args.limit)
    )
    for rec in records:
        json.dump(rec, sys.stdout, default=str)
        sys.stdout.write("\n")
    return 0


# --- subcommand: build ------------------------------------------------------


def _filters_from_args(args: argparse.Namespace) -> Filters:
    return Filters.parse(
        since=getattr(args, "since", None),
        until=getattr(args, "until", None),
        sources=getattr(args, "source", None),
        sessions=getattr(args, "session", None),
        project_slug=getattr(args, "project_slug", None),
    )


def _cmd_build(args: argparse.Namespace) -> int:
    filters = _filters_from_args(args)
    builder = TraceBuilder()
    sessions_used: set[tuple[str, str]] = set()
    total_events = 0

    for ev in iter_events(filters):
        builder.add(ev)
        sessions_used.add((ev.source, ev.session_id))
        total_events += 1

    trace = builder.finalize()
    out_path = Path(args.out)
    out_path.write_text(json.dumps(trace, separators=(",", ":")))
    size = out_path.stat().st_size
    print(
        f"wrote {out_path}  "
        f"events={total_events}  sessions={len(sessions_used)}  "
        f"size={size / (1 << 20):.2f}MiB",
        file=sys.stderr,
    )
    return 0


# --- subcommand: stats ------------------------------------------------------


def _cmd_stats(args: argparse.Namespace) -> int:
    filters = _filters_from_args(args)
    report = compute_stats(iter_events(filters))
    _print_stats(report, top_n=args.top)
    return 0


def _print_stats(report: StatsReport, *, top_n: int) -> None:
    print(f"== sessions ({report.total_sessions}) ==")
    rows = sorted(report.sessions.values(), key=lambda s: s.ts_start_us)
    print(
        f"{'source':6s} {'session':10s} {'wall':>8s} "
        f"{'tools':>6s} {'tok_in':>9s} {'tok_out':>8s} {'cache_hit':>10s}"
    )
    for s in rows:
        cache_pct = "—"
        if s.cache_hit_rate is not None:
            cache_pct = f"{s.cache_hit_rate * 100:.0f}%"
        print(
            f"{s.source:6s} {s.session_id[:8]:10s} "
            f"{s.wallclock_s:>7.0f}s {s.tool_calls:>6d} "
            f"{s.tokens_input:>9d} {s.tokens_output:>8d} {cache_pct:>10s}"
        )

    print()
    print(f"== categories ({sum(report.overall_by_category.values())} events) ==")
    for cat, n in report.overall_by_category.most_common(top_n):
        print(f"  {cat:12s} {n:>7d}")

    print()
    print(f"== top tools by count (top {top_n}) ==")
    for name, n in report.overall_by_tool.most_common(top_n):
        wall = report.overall_tool_wall_us.get(name, 0) / 1_000_000
        print(f"  {name:24s} {n:>7d}   {wall:>8.1f}s wall")

    print()
    print(f"== top tools by wall-clock (top {top_n}) ==")
    by_wall = sorted(report.overall_tool_wall_us.items(), key=lambda kv: -kv[1])[:top_n]
    for name, us in by_wall:
        print(f"  {name:24s} {us / 1_000_000:>8.1f}s")

    print()
    print(f"== top shell commands (top {top_n}) ==")
    for sig, n in report.top_commands.most_common(top_n):
        print(f"  {sig:30s} {n:>5d}")


# --- subcommand: hints ------------------------------------------------------


def _cmd_hints(args: argparse.Namespace) -> int:
    filters = _filters_from_args(args)
    hints = run_all(iter_events(filters))
    if args.category:
        hints = [h for h in hints if h.category in args.category]
    if args.json:
        json.dump([_hint_to_json(h) for h in hints], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        _print_hints_markdown(hints)
    return 0


def _hint_to_json(h: Hint) -> dict:
    return {
        "detector": h.detector,
        "category": h.category,
        "title": h.title,
        "severity": h.severity.value,
        "occurrences": h.occurrences,
        "est_wall_saved_s": h.est_wall_saved_s,
        "est_tokens_saved": h.est_tokens_saved,
        "remediation": h.remediation,
        "anchors": [
            {"source": a.source, "session_id": a.session_id, "ts_us": a.ts_us, "detail": a.detail}
            for a in h.anchors
        ],
        "evidence": h.evidence,
    }


def _print_hints_markdown(hints: list[Hint]) -> None:
    if not hints:
        print("# Hints\n\nNo hints fired in the selected window.")
        return
    print(f"# Hints ({len(hints)} found)\n")
    for h in hints:
        tag = f"[{h.severity.value.upper()}]"
        savings = ""
        if h.est_wall_saved_s and h.est_wall_saved_s >= 1:
            savings = f" — est. {h.est_wall_saved_s:.0f}s saveable"
        elif h.est_tokens_saved:
            savings = f" — est. {h.est_tokens_saved:,} tokens saveable"
        print(f"## {tag} {h.title}{savings}")
        print(f"_detector_: `{h.detector}`  _category_: `{h.category}`  _occurrences_: {h.occurrences}\n")
        if h.remediation:
            print(h.remediation + "\n")
        for a in h.anchors[:8]:
            ts_iso = _us_to_iso(a.ts_us)
            print(f"- `{a.source}:{a.session_id[:8]}` @ {ts_iso} — {a.detail}")
        if len(h.anchors) > 8:
            print(f"- … +{len(h.anchors) - 8} more")
        print()


def _us_to_iso(us: int) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(us / 1_000_000, tz=UTC).isoformat(timespec="seconds")


# --- subcommand: sample -----------------------------------------------------


def _cmd_sample(args: argparse.Namespace) -> int:
    try:
        from agent_tracer.telemetry.daemon import DEFAULT_DATASET, run
    except ImportError as e:
        if "lancedb" in str(e) or "pyarrow" in str(e):
            return _err(
                "sample requires the [store] extras. "
                "Install with: pip install -e '.[store]'"
            )
        raise
    dataset = Path(args.dataset) if args.dataset else DEFAULT_DATASET
    run(
        interval_s=args.interval,
        dataset=dataset,
        once=args.once,
        quiet=args.quiet,
    )
    return 0


# --- parser ----------------------------------------------------------------


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--since", default=None, help="Drop events before this UTC time (YYYY-MM-DD or ISO).")
    p.add_argument("--until", default=None, help="Drop events at or after this UTC time.")
    p.add_argument(
        "--session",
        action="append",
        default=None,
        help="Restrict to these session ids (repeatable).",
    )
    p.add_argument(
        "--project-slug",
        default=None,
        help="Restrict to one project slug (e.g., '-home-nod-github-...').",
    )
    p.add_argument(
        "--source",
        action="append",
        choices=["claude", "codex"],
        default=None,
        help="Restrict to one or more sources (default: both).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-tracer")
    p.add_argument("--version", action="version", version=f"agent-tracer {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Schema-discovery sanity report")
    d.add_argument("--file-limit", type=int, default=None,
                   help="Limit files scanned per source (faster sanity check).")
    d.set_defaults(func=_cmd_discover)

    li = sub.add_parser("list", help="Stream raw records from a source (debug)")
    li.add_argument("--source", required=True, choices=["claude", "codex"])
    li.add_argument("--limit", type=int, default=20)
    li.set_defaults(func=_cmd_list)

    b = sub.add_parser("build", help="Build a Perfetto trace JSON from sessions")
    _add_filter_args(b)
    b.add_argument("-o", "--out", default="trace.json", help="Output path.")
    b.set_defaults(func=_cmd_build)

    s = sub.add_parser("stats", help="Print per-session and aggregate statistics")
    _add_filter_args(s)
    s.add_argument("--top", type=int, default=15, help="Top-N rows per table.")
    s.set_defaults(func=_cmd_stats)

    h = sub.add_parser("hints", help="Print ranked optimization hints (markdown)")
    _add_filter_args(h)
    h.add_argument(
        "--category",
        action="append",
        default=None,
        help="Restrict to hint categories (e.g. agent, gpu, build). Repeatable.",
    )
    h.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    h.set_defaults(func=_cmd_hints)

    sa = sub.add_parser(
        "sample",
        help="Run the telemetry sampler daemon (rocm-smi/nvidia-smi/proc → LanceDB)",
    )
    sa.add_argument("--interval", type=float, default=1.0, help="Seconds between ticks.")
    sa.add_argument(
        "--dataset",
        default=None,
        help="LanceDB dataset path (default: ~/.cache/agent-tracer/telemetry.lance).",
    )
    sa.add_argument("--once", action="store_true", help="Run a single tick and exit.")
    sa.add_argument("--quiet", action="store_true", help="Suppress periodic status lines.")
    sa.set_defaults(func=_cmd_sample)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())


# Re-export so existing imports (`from agent_tracer.cli import _err`) still work
# in any test code that may have grabbed it. Not used internally.
def _err(msg: str) -> int:
    print(f"agent-tracer: error: {msg}", file=sys.stderr)
    return 2


__all__ = ["build_parser", "main", "EventKind"]
