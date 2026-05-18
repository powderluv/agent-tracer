"""Raw record iteration for Claude Code session logs.

Layout::

    ~/.claude/projects/<cwd-slug>/<sessionId>/<sessionId>.jsonl   (main thread)
    ~/.claude/projects/<cwd-slug>/<sessionId>/subagents/agent-<agentId>.jsonl
    ~/.claude/projects/<cwd-slug>/<sessionId>/subagents/agent-<agentId>.meta.json

Each JSONL line is one record. Top-level ``type`` is one of
``assistant`` | ``user`` | ``progress`` | ``system``. Every record carries
``parentUuid, isSidechain, userType, cwd, sessionId, version, gitBranch,
agentId, slug, type, uuid, timestamp``.

Tool spans are reconstructed by pairing assistant-message ``tool_use`` content
blocks with the matching user-message ``tool_result`` block, linked by
``tool_use.id == tool_result.tool_use_id`` (and confirmed by
``tool_result.parentUuid == tool_use.uuid``).

This module is intentionally just the *raw iteration* layer — normalization
into ``AgentEvent`` lives in a separate normalizer module added in P1.
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
        for session_dir in sorted(project_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            main = session_dir / f"{session_id}.jsonl"
            if main.exists():
                yield ClaudeSessionFile(
                    path=main,
                    project_slug=project_dir.name,
                    session_id=session_id,
                    agent_id=None,
                    is_subagent=False,
                    is_compaction=False,
                )
            sub_dir = session_dir / "subagents"
            if sub_dir.is_dir():
                for sub in sorted(sub_dir.glob("agent-*.jsonl")):
                    name = sub.stem  # agent-<id> or agent-acompact-<id>
                    is_compaction = name.startswith("agent-acompact-")
                    # agent id is everything after the first/second hyphen
                    agent_id = name.split("-", 2)[-1] if is_compaction else name[len("agent-"):]
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

    Streams from ``start_offset``; the offset returned for each record is the
    position *after* that record so the caller can resume from there next time.
    Skips malformed lines silently (defensive — the live TUI writes incrementally).
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
