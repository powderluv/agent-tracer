"""Schema-discovery sanity report.

Run as::

    agent-tracer discover

Scans local Claude and Codex logs, prints record counts, top-level types,
nested payload-type distributions, and file counts. Used in P0 to confirm
the parser assumptions match what's actually on disk; useful afterwards as
a regression check when the upstream tools roll a new event shape.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field

from agent_tracer.parsers import claude, codex


@dataclass(slots=True)
class ClaudeReport:
    files: int = 0
    records: int = 0
    main_files: int = 0
    subagent_files: int = 0
    compaction_files: int = 0
    types: collections.Counter = field(default_factory=collections.Counter)
    block_types: collections.Counter = field(default_factory=collections.Counter)


@dataclass(slots=True)
class CodexReport:
    files: int = 0
    records: int = 0
    types: collections.Counter = field(default_factory=collections.Counter)
    payload_types: collections.Counter = field(default_factory=collections.Counter)


def scan_claude(file_limit: int | None = None) -> ClaudeReport:
    report = ClaudeReport()
    for i, sf in enumerate(claude.iter_session_files()):
        if file_limit is not None and i >= file_limit:
            break
        report.files += 1
        if sf.is_compaction:
            report.compaction_files += 1
        elif sf.is_subagent:
            report.subagent_files += 1
        else:
            report.main_files += 1
        for _, rec in claude.iter_raw_records(sf.path):
            report.records += 1
            report.types[rec.get("type", "?")] += 1
            msg = rec.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                role = msg.get("role", "?")
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict):
                            report.block_types[(role, blk.get("type", "?"))] += 1
                elif isinstance(content, str):
                    report.block_types[(role, "str")] += 1
    return report


def scan_codex(file_limit: int | None = None) -> CodexReport:
    report = CodexReport()
    for i, sf in enumerate(codex.iter_session_files()):
        if file_limit is not None and i >= file_limit:
            break
        report.files += 1
        for _, rec in codex.iter_raw_records(sf.path):
            report.records += 1
            t = rec.get("type", "?")
            report.types[t] += 1
            p = rec.get("payload")
            if isinstance(p, dict):
                report.payload_types[(t, p.get("type", "?"))] += 1
    return report


def format_report(claude_r: ClaudeReport, codex_r: CodexReport) -> str:
    lines: list[str] = []
    lines.append("== Claude ==")
    lines.append(
        f"files: {claude_r.files} "
        f"(main {claude_r.main_files}, subagent {claude_r.subagent_files}, "
        f"compaction {claude_r.compaction_files})"
    )
    lines.append(f"records: {claude_r.records}")
    lines.append(f"top-level types: {dict(claude_r.types.most_common())}")
    lines.append("message.role × content-block type:")
    for k, v in claude_r.block_types.most_common():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("== Codex ==")
    lines.append(f"files: {codex_r.files}")
    lines.append(f"records: {codex_r.records}")
    lines.append(f"top-level types: {dict(codex_r.types.most_common())}")
    lines.append("(record.type, payload.type) pairs:")
    for k, v in codex_r.payload_types.most_common(25):
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
