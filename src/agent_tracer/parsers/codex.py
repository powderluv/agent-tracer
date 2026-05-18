"""Raw record iteration for Codex session logs.

Layout::

    ~/.codex/sessions/YYYY/MM/DD/rollout-<isoTs>-<sessionId>.jsonl

Each line is ``{timestamp, type, payload}``. Top-level ``type`` is one of
``response_item`` | ``event_msg`` | ``turn_context`` | ``compacted`` |
``session_meta``. The interesting nested types live under ``payload.type``:

* ``response_item.function_call`` ↔ ``event_msg.exec_command_end`` via ``call_id``
  (also ``response_item.function_call_output`` for non-exec tools)
* ``response_item.custom_tool_call`` ↔ ``response_item.custom_tool_call_output``
* ``response_item.reasoning`` — model reasoning blocks
* ``response_item.message`` — assistant/user messages
* ``event_msg.token_count`` — rate-limit + cumulative token info
* ``event_msg.agent_message`` / ``user_message`` / ``task_started`` / ``task_complete``
* ``event_msg.patch_apply_end`` — apply_patch tool completion
* ``compacted`` — context-compaction summary record
* ``session_meta`` — one per session header

The live Codex TUI also maintains ``~/.codex/logs_2.sqlite`` (WAL); we do not
touch it. The rollout JSONLs are append-only and safe to stream.

Like the Claude parser, this module is the raw iteration layer; normalization
into ``AgentEvent`` happens in a downstream module.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"

_ROLLOUT_RE = re.compile(
    r"^rollout-(?P<iso>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})-(?P<sid>[0-9a-f-]+)\.jsonl$"
)


@dataclass(slots=True)
class CodexSessionFile:
    path: Path
    session_id: str
    started_at: datetime  # parsed from filename, UTC


def iter_session_files(root: Path | None = None) -> Iterator[CodexSessionFile]:
    """Yield all Codex rollout files under ``root``, sorted by start time.

    Defaults to :data:`CODEX_SESSIONS_DIR` resolved at call time so the
    module-level constant can be monkeypatched in tests.
    """
    if root is None:
        root = CODEX_SESSIONS_DIR
    if not root.exists():
        return
    files: list[CodexSessionFile] = []
    for p in root.rglob("rollout-*.jsonl"):
        m = _ROLLOUT_RE.match(p.name)
        if not m:
            continue
        ts = datetime.strptime(m["iso"], "%Y-%m-%dT%H-%M-%S")
        files.append(CodexSessionFile(path=p, session_id=m["sid"], started_at=ts))
    files.sort(key=lambda f: f.started_at)
    yield from files


def iter_raw_records(
    path: Path,
    start_offset: int = 0,
) -> Iterator[tuple[int, dict]]:
    """Yield ``(byte_offset, record)`` pairs from a Codex rollout JSONL."""
    with path.open("rb") as f:
        if start_offset:
            f.seek(start_offset)
        for line in f:
            try:
                rec = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            yield f.tell(), rec
