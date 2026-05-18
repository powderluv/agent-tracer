"""Argparse-based entry point.

P0 shipped ``discover`` and ``list``. P1 adds ``build`` (Claude → Perfetto
trace JSON). Codex is folded into ``build`` in P2.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from agent_tracer import __version__
from agent_tracer.normalize import normalize_claude_session
from agent_tracer.parsers import claude, codex, discover
from agent_tracer.perfetto import TraceBuilder
from agent_tracer.timeutil import iso_to_us


def _cmd_discover(args: argparse.Namespace) -> int:
    claude_r = discover.scan_claude(file_limit=args.file_limit)
    codex_r = discover.scan_codex(file_limit=args.file_limit)
    print(discover.format_report(claude_r, codex_r))
    return 0


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
    if args.source == "claude":
        records = _claude_records(args.limit)
    elif args.source == "codex":
        records = _codex_records(args.limit)
    else:
        return _err("--source must be 'claude' or 'codex'")
    for rec in records:
        json.dump(rec, sys.stdout, default=str)
        sys.stdout.write("\n")
    return 0


def _parse_since(s: str | None) -> int | None:
    if not s:
        return None
    # Accept ``YYYY-MM-DD`` or full ISO; treat date-only as UTC midnight.
    if len(s) == 10:
        s = s + "T00:00:00Z"
    return iso_to_us(s)


def _cmd_build(args: argparse.Namespace) -> int:
    since_us = _parse_since(args.since)
    until_us = _parse_since(args.until)
    builder = TraceBuilder()

    sessions_filter = set(args.session) if args.session else None
    total_events = 0
    files_used = 0
    sessions_used: set[str] = set()

    for sf in claude.iter_session_files(project_slug=args.project_slug):
        if sessions_filter is not None and sf.session_id not in sessions_filter:
            continue
        had_event = False
        for ev in normalize_claude_session(
            (rec for _, rec in claude.iter_raw_records(sf.path)),
            source_session_id=sf.session_id,
            source_agent_id=sf.agent_id,
        ):
            if since_us is not None and ev.ts_start_us < since_us:
                continue
            if until_us is not None and ev.ts_start_us >= until_us:
                continue
            builder.add(ev)
            total_events += 1
            had_event = True
        if had_event:
            files_used += 1
            sessions_used.add(sf.session_id)

    trace = builder.finalize()
    out_path = Path(args.out)
    out_path.write_text(json.dumps(trace, separators=(",", ":")))
    size = out_path.stat().st_size
    print(
        f"wrote {out_path}  "
        f"events={total_events}  files={files_used}  sessions={len(sessions_used)}  "
        f"size={size / (1 << 20):.2f}MiB",
        file=sys.stderr,
    )
    return 0


def _err(msg: str) -> int:
    print(f"agent-tracer: error: {msg}", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent-tracer")
    p.add_argument("--version", action="version", version=f"agent-tracer {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Schema-discovery sanity report")
    d.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit files scanned per source (faster sanity check).",
    )
    d.set_defaults(func=_cmd_discover)

    li = sub.add_parser("list", help="Stream raw records from a source (debug)")
    li.add_argument("--source", required=True, choices=["claude", "codex"])
    li.add_argument("--limit", type=int, default=20)
    li.set_defaults(func=_cmd_list)

    b = sub.add_parser("build", help="Build a Perfetto trace JSON from sessions")
    b.add_argument(
        "--since",
        default=None,
        help="Drop events before this UTC time (YYYY-MM-DD or ISO).",
    )
    b.add_argument(
        "--until",
        default=None,
        help="Drop events at or after this UTC time (YYYY-MM-DD or ISO).",
    )
    b.add_argument(
        "--session",
        action="append",
        default=None,
        help="Restrict to these session ids (repeatable).",
    )
    b.add_argument(
        "--project-slug",
        default=None,
        help="Restrict to one project slug (e.g., '-home-nod-github-...').",
    )
    b.add_argument("-o", "--out", default="trace.json", help="Output path.")
    b.set_defaults(func=_cmd_build)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
