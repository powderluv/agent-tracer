"""Raw record iteration for Claude Code session logs (READ-ONLY).

Layout discovered on this machine::

    ~/.claude/projects/<cwd-slug>/<sessionId>.jsonl                 # main session
    ~/.claude/projects/<cwd-slug>/<sessionId>/subagents/agent-<agentId>.jsonl
    ~/.claude/projects/<cwd-slug>/<sessionId>/subagents/agent-<agentId>.meta.json
    ~/.claude/projects/<cwd-slug>/memory/...                        # ignored

The main session JSONL is a *sibling* to the session directory, not inside it.
Some sessions have only a main file, some only subagents, some both.

Each JSONL line is one record. Top-level ``type`` is one of
``assistant`` | ``user`` | ``progress`` | ``system``. Every record carries
``parentUuid, isSidechain, userType, cwd, sessionId, version, gitBranch,
agentId, slug, type, uuid, timestamp``.

Tool spans are reconstructed by pairing assistant-message ``tool_use`` content
blocks with the matching user-message ``tool_result`` block, linked by
``tool_use.id == tool_result.tool_use_id`` (and confirmed by
``tool_result.parentUuid == tool_use.uuid``).

**Read-only contract**: every file is opened with ``"rb"`` mode. This module
must never write, truncate, rename, or chmod anything under ``CLAUDE_PROJECTS_DIR``.
A regression test (``tests/test_read_only.py``) enforces it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass(slots=True)
class ClaudeSessionFile:
    """Locator for one Claude JSONL file (main or subagent)."""

    path: Path
    project_slug: str   # the cwd-slug directory name
    session_id: str
    agent_id: str | None  # None for main session file
    is_subagent: bool
    is_compaction: bool  # subagents named agent-acompact-*

    @property
    def kind(self) -> str:
        if self.is_compaction:
            return "subagent_compaction"
        if self.is_subagent:
            return "subagent"
        return "main"


def iter_session_files(
    root: Path | None = None,
    project_slug: str | None = None,
) -> Iterator[ClaudeSessionFile]:
    """Yield all Claude session files under ``root``.

    Defaults to :data:`CLAUDE_PROJECTS_DIR` resolved at call time (so the
    module-level constant can be monkeypatched in tests).

    Filters to ``project_slug`` (the cwd-slug directory name) if given.

    The yield order pairs each main session file with its subagents:
    main first (if it exists), then subagents alphabetically.
    """
    if root is None:
        root = CLAUDE_PROJECTS_DIR
    if not root.exists():
        return
    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_slug is not None and project_dir.name != project_slug:
            continue

        # 1) Main session files are siblings: <project>/<sessionId>.jsonl.
        # 2) Subagent files live under <project>/<sessionId>/subagents/.
        # Build the union of session_ids from both sources, then yield in
        # session-id order so a main and its subagents appear together.
        session_ids: set[str] = set()
        for entry in project_dir.iterdir():
            if entry.is_file() and entry.suffix == ".jsonl":
                session_ids.add(entry.stem)
            elif entry.is_dir() and entry.name != "memory":
                session_ids.add(entry.name)

        for session_id in sorted(session_ids):
            main = project_dir / f"{session_id}.jsonl"
            if main.is_file():
                yield ClaudeSessionFile(
                    path=main,
                    project_slug=project_dir.name,
                    session_id=session_id,
                    agent_id=None,
                    is_subagent=False,
                    is_compaction=False,
                )
            sub_dir = project_dir / session_id / "subagents"
            if sub_dir.is_dir():
                for sub in sorted(sub_dir.glob("agent-*.jsonl")):
                    name = sub.stem  # agent-<id> or agent-acompact-<id>
                    is_compaction = name.startswith("agent-acompact-")
                    agent_id = (
                        name.split("-", 2)[-1] if is_compaction else name[len("agent-"):]
                    )
                    yield ClaudeSessionFile(
                        path=sub,
                        project_slug=project_dir.name,
                        session_id=session_id,
                        agent_id=agent_id,
                        is_subagent=True,
                        is_compaction=is_compaction,
                    )


def iter_raw_records(
    path: Path,
    start_offset: int = 0,
) -> Iterator[tuple[int, dict]]:
    """Yield ``(byte_offset, record)`` pairs from a Claude JSONL file.

    Read-only: opens the file with ``"rb"``. Streams from ``start_offset``;
    the offset returned for each record is the position *after* that record
    so the caller can resume from there next time. Skips malformed lines
    silently (the live TUI writes incrementally).
    """
    with path.open("rb") as f:
        if start_offset:
            f.seek(start_offset)
        for line in f:
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            yield f.tell(), rec


def project_slug_for_cwd(cwd: str | os.PathLike) -> str:
    """Mirror Claude's cwd → slug rule: leading '/' stripped, '/' → '-'."""
    s = str(cwd)
    if s.startswith("/"):
        s = s[1:]
    return "-" + s.replace("/", "-")
