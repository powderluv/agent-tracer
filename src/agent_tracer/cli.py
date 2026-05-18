"""Argparse-based entry point. P0 ships ``discover`` and ``list``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from agent_tracer import __version__
from agent_tracer.parsers import claude, codex, discover


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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
