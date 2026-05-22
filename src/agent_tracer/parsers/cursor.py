"""Raw record iteration for Cursor AI editor session logs (READ-ONLY).

Layout discovered on this machine::

    ~/.cursor/projects/<project-slug>/agent-transcripts/<tid>/<tid>.jsonl
    ~/.cursor/projects/<project-slug>/terminals/<id>.txt

Project slugs are either cwd-based (``c-develop-agent-skills``) or numeric
IDs (``1777990447990``). Numeric-only directories may not have an
``agent-transcripts/`` sub-directory.

Each transcript JSONL line is ``{role, message}`` — no timestamps, no tool
result records, no tool-use IDs. Real timestamps, token counts, and thinking
durations are recovered from the global ``state.vscdb`` SQLite database (see
:func:`load_bubble_metadata`). Terminal command durations come from the YAML
``terminals/`` files (see :func:`load_terminal_logs`).

**Read-only contract**: every file is opened with ``"rb"`` mode; SQLite
connections use ``?mode=ro`` URI.  This module must never write, truncate,
rename, or chmod anything under ``CURSOR_PROJECTS_DIR`` or the global
storage path.  A regression test (``tests/test_read_only.py``) enforces it.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from agent_tracer.timeutil import iso_to_us

CURSOR_PROJECTS_DIR = Path.home() / ".cursor" / "projects"

# Platform-aware path to the global state database.
if sys.platform == "win32":
    _APPDATA = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    CURSOR_GLOBAL_STATE_DB = _APPDATA / "Cursor" / "User" / "globalStorage" / "state.vscdb"
elif sys.platform == "darwin":
    CURSOR_GLOBAL_STATE_DB = (
        Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    )
else:
    CURSOR_GLOBAL_STATE_DB = (
        Path.home() / ".config" / "Cursor" / "User" / "globalStorage" / "state.vscdb"
    )


# ---------------------------------------------------------------------------
# Session-file discovery
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CursorSessionFile:
    """Locator for one Cursor agent transcript JSONL file."""

    path: Path
    project_slug: str   # the project directory name
    transcript_id: str  # same as composer_id in state.vscdb


def iter_session_files(
    root: Path | None = None,
    project_slug: str | None = None,
) -> Iterator[CursorSessionFile]:
    """Yield all Cursor agent-transcript JSONL files under ``root``.

    Defaults to :data:`CURSOR_PROJECTS_DIR` resolved at call time so the
    module-level constant can be monkeypatched in tests.

    Filters to ``project_slug`` if given. Skips project directories that
    have no ``agent-transcripts/`` sub-directory.
    """
    if root is None:
        root = CURSOR_PROJECTS_DIR
    if not root.exists():
        return
    for project_dir in sorted(root.iterdir()):
        if not project_dir.is_dir():
            continue
        if project_slug is not None and project_dir.name != project_slug:
            continue
        transcripts_dir = project_dir / "agent-transcripts"
        if not transcripts_dir.is_dir():
            continue
        for tid_dir in sorted(transcripts_dir.iterdir()):
            if not tid_dir.is_dir():
                continue
            jsonl = tid_dir / f"{tid_dir.name}.jsonl"
            if jsonl.is_file():
                yield CursorSessionFile(
                    path=jsonl,
                    project_slug=project_dir.name,
                    transcript_id=tid_dir.name,
                )


def iter_raw_records(
    path: Path,
    start_offset: int = 0,
) -> Iterator[tuple[int, dict]]:
    """Yield ``(byte_offset, record)`` pairs from a Cursor transcript JSONL.

    Read-only: opens with ``"rb"``.  Skips malformed lines silently.
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


# ---------------------------------------------------------------------------
# Bubble metadata from global state DB
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BubbleInfo:
    """Metadata for one conversation bubble extracted from ``state.vscdb``."""

    bubble_id: str
    created_at_us: int   # epoch microseconds
    bubble_type: int     # 1 = user, 2 = assistant
    thinking_duration_ms: int | None
    tool_call_id: str | None


@dataclass(slots=True)
class ConversationBlob:
    """One API message from the blob store (the actual data sent to/from the model).

    Only the fields needed for token estimation are kept; the full message
    content is not retained in memory.
    """

    role: str  # system, user, assistant, tool
    content_chars: int  # character count of content (for token estimation)
    is_summary: bool  # True if this is a "[Previous conversation summary]" blob


@dataclass(slots=True)
class SessionMetadata:
    """Conversation-level metadata extracted from ``composerData``."""

    bubbles: list[BubbleInfo]
    model_name: str | None
    blobs: list[ConversationBlob] | None  # API messages from conversationState


def load_bubble_metadata(
    transcript_id: str,
    db_path: Path | None = None,
) -> SessionMetadata | None:
    """Load per-bubble timestamps and metadata from ``state.vscdb``.

    Returns a :class:`SessionMetadata` with the ordered bubble list and
    the context token limit from ``composerData``, or ``None`` if the
    database is unavailable / doesn't contain this transcript.

    The database is opened read-only (``?mode=ro``).
    """
    if db_path is None:
        db_path = CURSOR_GLOBAL_STATE_DB
    if not db_path.exists():
        return None

    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.OperationalError:
        return None

    try:
        cur = conn.cursor()

        # 1) Get the bubble ordering from composerData.
        cur.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"composerData:{transcript_id}",),
        )
        row = cur.fetchone()
        if row is None:
            return None
        composer_data = json.loads(row[0])
        headers = composer_data.get("fullConversationHeadersOnly") or []
        if not headers:
            return None

        # 2) Batch-read bubbleId entries.
        bubbles: list[BubbleInfo] = []
        for hdr in headers:
            bid = hdr.get("bubbleId")
            if not bid:
                continue
            cur.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"bubbleId:{transcript_id}:{bid}",),
            )
            brow = cur.fetchone()
            grouping = hdr.get("grouping") or {}
            if brow is None:
                # Bubble data may have been evicted; use header info.
                bubbles.append(BubbleInfo(
                    bubble_id=bid,
                    created_at_us=0,
                    bubble_type=hdr.get("type", 0),
                    thinking_duration_ms=grouping.get("thinkingDurationMs"),
                    tool_call_id=grouping.get("toolCallId"),
                ))
                continue

            bdata = json.loads(brow[0])
            created_at_raw = bdata.get("createdAt", "")
            try:
                created_at_us = iso_to_us(created_at_raw) if created_at_raw else 0
            except (ValueError, TypeError):
                created_at_us = 0

            tfd = bdata.get("toolFormerData") or {}
            bubbles.append(BubbleInfo(
                bubble_id=bid,
                created_at_us=created_at_us,
                bubble_type=bdata.get("type", hdr.get("type", 0)),
                thinking_duration_ms=bdata.get("thinkingDurationMs") or grouping.get("thinkingDurationMs"),
                tool_call_id=tfd.get("toolCallId") or grouping.get("toolCallId"),
            ))

        if not bubbles:
            return None

        # 3) Load API-message blobs from conversationState.
        blobs = _load_conversation_blobs(cur, composer_data)

        return SessionMetadata(
            bubbles=bubbles,
            model_name=(composer_data.get("modelConfig") or {}).get("modelName"),
            blobs=blobs,
        )
    except (sqlite3.OperationalError, json.JSONDecodeError, KeyError):
        return None
    finally:
        conn.close()


def _load_conversation_blobs(
    cur: sqlite3.Cursor,
    composer_data: dict,
) -> list[ConversationBlob] | None:
    """Extract API-message blobs from ``conversationState``.

    The ``conversationState`` field is a ``~``-prefixed base64 blob containing
    a protobuf-like sequence of 32-byte SHA-256 hashes.  Each hash references
    an ``agentKv:blob:{hash}`` entry in ``cursorDiskKV`` containing a JSON
    API message (``{role, content, …}``).

    Only ``role``, ``content`` length, and summary detection are extracted;
    the full message bodies are not kept in memory.
    """
    cs = composer_data.get("conversationState")
    if not cs or not isinstance(cs, str) or not cs.startswith("~"):
        return None

    try:
        b64 = cs[1:]
        b64 += "=" * (4 - len(b64) % 4) if len(b64) % 4 else ""
        raw = base64.b64decode(b64)
    except Exception:
        return None

    # Parse repeated (field=1, wire=2, len=32) entries = SHA-256 hashes.
    hashes: list[str] = []
    i = 0
    while i + 33 < len(raw):
        if raw[i] == 0x0A and raw[i + 1] == 0x20:
            hashes.append(raw[i + 2 : i + 34].hex())
            i += 34
        else:
            break
    if not hashes:
        return None

    blobs: list[ConversationBlob] = []
    for h in hashes:
        cur.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?",
            (f"agentKv:blob:{h}",),
        )
        row = cur.fetchone()
        if row is None:
            continue
        try:
            msg = json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            continue

        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            content_chars = len(content)
        elif isinstance(content, list):
            content_chars = sum(len(json.dumps(c)) for c in content)
        else:
            content_chars = len(json.dumps(content))

        is_summary = (
            role == "user"
            and isinstance(content, str)
            and content.startswith("[Previous conversation summary]")
        )

        blobs.append(ConversationBlob(
            role=role,
            content_chars=content_chars,
            is_summary=is_summary,
        ))

    return blobs if blobs else None


# ---------------------------------------------------------------------------
# Terminal-log parsing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TerminalEntry:
    """One terminal command execution from a ``terminals/*.txt`` file."""

    command: str
    started_at_us: int
    ended_at_us: int | None
    elapsed_ms: int | None
    exit_code: int | None
    cwd: str | None


_YAML_FIELD = re.compile(r"^(\w[\w_]*):\s*(.*)$")


def _parse_ts(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return iso_to_us(raw)
    except (ValueError, TypeError):
        return None


def _parse_int(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_terminal_file(path: Path) -> TerminalEntry | None:
    """Parse a single Cursor terminal log file.

    Format is YAML-ish frontmatter between ``---`` delimiters, followed by
    command output, followed by another ``---`` delimited block with
    ``exit_code``, ``elapsed_ms``, and ``ended_at``.
    """
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError:
        return None

    sections = text.split("---")
    # Expect at least 3 sections: before first ---, header, content, footer
    if len(sections) < 3:
        return None

    # Parse header (second element after split — first is empty or whitespace).
    header: dict[str, str] = {}
    for line in sections[1].strip().splitlines():
        m = _YAML_FIELD.match(line.strip())
        if m:
            header[m.group(1)] = m.group(2).strip().strip('"').strip("'")

    # Parse footer (last non-empty section).
    footer: dict[str, str] = {}
    for section in reversed(sections[2:]):
        stripped = section.strip()
        if not stripped:
            continue
        for line in stripped.splitlines():
            m = _YAML_FIELD.match(line.strip())
            if m:
                footer[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        if footer:
            break

    cmd = header.get("command")
    started_raw = header.get("started_at")
    if not cmd or not started_raw:
        return None

    try:
        started_us = iso_to_us(started_raw)
    except (ValueError, TypeError):
        return None

    ended_us = _parse_ts(footer.get("ended_at"))
    elapsed = _parse_int(footer.get("elapsed_ms"))
    exit_code = _parse_int(footer.get("exit_code"))

    return TerminalEntry(
        command=cmd,
        started_at_us=started_us,
        ended_at_us=ended_us,
        elapsed_ms=elapsed,
        exit_code=exit_code,
        cwd=header.get("cwd"),
    )


def load_terminal_logs(project_dir: Path) -> list[TerminalEntry]:
    """Load all terminal log files from ``project_dir/terminals/``."""
    terminals_dir = project_dir / "terminals"
    if not terminals_dir.is_dir():
        return []
    entries: list[TerminalEntry] = []
    for p in sorted(terminals_dir.iterdir()):
        if p.suffix == ".txt" and p.is_file():
            entry = _parse_terminal_file(p)
            if entry is not None:
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def project_slug_for_cwd(cwd: str | os.PathLike) -> str:
    """Mirror Cursor's cwd → slug rule.

    Observed pattern: ``C:\\develop\\foo`` → ``c-develop-foo``.
    Drive letter is lowercased, colon stripped, separators become ``-``.
    """
    s = str(cwd)
    # Normalise Windows path separators.
    s = s.replace("\\", "/")
    # Strip drive colon: "C:" → "C"
    if len(s) >= 2 and s[1] == ":":
        s = s[0] + s[2:]
    # Leading / stripped if present.
    if s.startswith("/"):
        s = s[1:]
    return s.lower().replace("/", "-")
