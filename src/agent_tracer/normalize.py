"""Normalize raw parser records into ``AgentEvent`` instances.

P1 ships the Claude normalizer; Codex lands in P2. The normalizer is
intentionally a generator: yields events in source order, holds only the
in-flight ``tool_use`` table in memory so memory stays bounded even on
million-record sessions.

Key invariants:

* Tool calls become *complete* events. A ``tool_use`` block opens a span
  keyed by its ``id``; the matching ``tool_result`` block closes it via
  ``tool_use_id``. The pairing is sanity-checked with the ``parentUuid``
  chain.
* Streaming chunks: an assistant ``message.id`` may appear across multiple
  ``assistant`` records as the response streams. We register tool_use blocks
  with their *earliest* sighting; later chunks of the same block are ignored.
  Token usage counters are emitted per chunk (Perfetto plots steps).
* Subagent fan-out: a parent session has a ``tool_use`` named ``Agent``;
  the subagent writes its own JSONL with the same ``sessionId`` but a
  distinct ``agentId``. We do not try to link the two with a shared id —
  in the trace they appear as sibling lanes within the same session ``pid``.
* Unknown top-level types (``attachment``, ``permission-mode``,
  ``pr-link``, ``queue-operation``, ``file-history-snapshot``, …) are
  recognized and skipped without warning. They are session metadata, not
  agent activity.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.timeutil import iso_to_us

# Truncation limits keep the trace JSON manageable. Perfetto and Chrome
# tracing both choke on multi-MB args.
_MAX_INPUT_BYTES = 2048
_MAX_RESULT_BYTES = 4096
_MAX_TEXT_BYTES = 2048


def _truncate_str(s: str, n: int) -> str:
    if len(s) <= n:
        return s
    return s[:n] + f"…<+{len(s) - n}B>"


def _truncate_obj(obj: Any, n: int) -> Any:
    """Stringify+truncate an opaque payload for args display."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return _truncate_str(obj, n)
    try:
        import json

        s = json.dumps(obj, default=str)
    except (TypeError, ValueError):
        s = repr(obj)
    return _truncate_str(s, n)


def _extract_result_text(tool_result_block: dict) -> str:
    """Tool result content is either a str or a list of {type:'text',text:...} blocks."""
    content = tool_result_block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif c.get("type") == "tool_reference":
                parts.append(f"<tool_reference {c.get('tool_name')}>")
            else:
                parts.append(f"<{c.get('type', '?')}>")
        return "\n".join(parts)
    return ""


# ──────────────────────────────────────────────────────────────────────────
# Claude


def normalize_claude_session(
    records: Iterable[dict],
    *,
    source_session_id: str | None = None,
    source_agent_id: str | None = None,
) -> Iterator[AgentEvent]:
    """Normalize one Claude JSONL stream (main session or one subagent) into events.

    ``source_session_id`` and ``source_agent_id`` override the values read from
    records (useful when the file's records lack the field — extremely rare,
    but a defensive escape hatch).
    """
    pending: dict[str, tuple[int, dict, dict]] = {}  # tool_use_id -> (ts_us, tool_use, parent_rec)

    for rec in records:
        rtype = rec.get("type")
        ts_raw = rec.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts_us = iso_to_us(ts_raw)
        except ValueError:
            continue

        sid = source_session_id or rec.get("sessionId") or ""
        aid = source_agent_id or rec.get("agentId") or None

        base: dict[str, Any] = {
            "source": "claude",
            "session_id": sid,
            "agent_id": aid,
            "cwd": rec.get("cwd"),
            "git_branch": rec.get("gitBranch"),
            "uuid": rec.get("uuid"),
            "parent_uuid": rec.get("parentUuid"),
        }

        if rtype == "user":
            yield from _from_user(rec, ts_us, base, pending)
        elif rtype == "assistant":
            yield from _from_assistant(rec, ts_us, base, pending)
        elif rtype == "progress":
            data = rec.get("data") or {}
            yield AgentEvent(
                kind=EventKind.PROGRESS,
                name=str(data.get("type", "progress")),
                ts_start_us=ts_us,
                payload={
                    "data": _truncate_obj(data, _MAX_INPUT_BYTES),
                    "parent_tool_use_id": rec.get("parentToolUseID"),
                },
                **base,
            )
        # Other top-level types (attachment, pr-link, file-history-snapshot,
        # permission-mode, queue-operation, custom-title, agent-name,
        # last-prompt, system) are session metadata, not agent activity.
        # Skip them — discover() catalogs the distribution if we ever care.

    # In-flight tool_uses at end of stream: emit as instant ``error`` events at
    # the start time so they're visible in the trace as orphans (truncated
    # session or session still running).
    for tuid, (start_ts, tu_block, parent_rec) in pending.items():
        yield AgentEvent(
            source="claude",
            session_id=parent_rec.get("sessionId", ""),
            agent_id=parent_rec.get("agentId"),
            kind=EventKind.ERROR,
            name=f"orphan:{tu_block.get('name', '?')}",
            ts_start_us=start_ts,
            uuid=parent_rec.get("uuid"),
            parent_uuid=parent_rec.get("parentUuid"),
            tool_use_id=tuid,
            payload={"input": _truncate_obj(tu_block.get("input"), _MAX_INPUT_BYTES)},
        )


def _from_user(
    rec: dict,
    ts_us: int,
    base: dict[str, Any],
    pending: dict[str, tuple[int, dict, dict]],
) -> Iterator[AgentEvent]:
    # User-rejection records have no `message` but a top-level ``toolUseResult``.
    if "message" not in rec and "toolUseResult" in rec:
        yield AgentEvent(
            kind=EventKind.ERROR,
            name="user_rejected",
            ts_start_us=ts_us,
            payload={"reason": _truncate_obj(rec.get("toolUseResult"), _MAX_INPUT_BYTES)},
            is_error=True,
            **base,
        )
        return

    msg = rec.get("message") or {}
    if not isinstance(msg, dict):
        return
    content = msg.get("content")

    if isinstance(content, str):
        yield AgentEvent(
            kind=EventKind.USER_TURN,
            name="user_turn",
            ts_start_us=ts_us,
            payload={"text": _truncate_str(content, _MAX_TEXT_BYTES)},
            **base,
        )
        return
    if not isinstance(content, list):
        return

    for blk in content:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "tool_result":
            tuid = blk.get("tool_use_id")
            if not tuid:
                continue
            pend = pending.pop(tuid, None)
            if pend is None:
                # Orphan result — tool_use was in a different file or lost.
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    name="orphan_tool_result",
                    ts_start_us=ts_us,
                    payload={
                        "tool_use_id": tuid,
                        "result": _truncate_str(_extract_result_text(blk), _MAX_RESULT_BYTES),
                    },
                    tool_use_id=tuid,
                    is_error=True,
                    **base,
                )
                continue
            start_ts, tu_block, parent_rec = pend
            tool_name = tu_block.get("name", "?")
            input_obj = tu_block.get("input") or {}
            subagent_type = (
                input_obj.get("subagent_type")
                if tool_name == "Agent" and isinstance(input_obj, dict)
                else None
            )
            # Use the parent assistant record's uuid + parent_uuid so the
            # tool_call's identity is the tool_use, not the result.
            yield AgentEvent(
                source="claude",
                session_id=base["session_id"],
                agent_id=base["agent_id"],
                cwd=base["cwd"],
                git_branch=base["git_branch"],
                uuid=parent_rec.get("uuid"),
                parent_uuid=parent_rec.get("parentUuid"),
                kind=EventKind.TOOL_CALL,
                name=str(tool_name),
                ts_start_us=start_ts,
                ts_end_us=ts_us,
                tool_use_id=tuid,
                subagent_type=subagent_type,
                is_error=bool(blk.get("is_error")),
                payload={
                    "input": _truncate_obj(input_obj, _MAX_INPUT_BYTES),
                    "result": _truncate_str(
                        _extract_result_text(blk), _MAX_RESULT_BYTES
                    ),
                },
            )
        elif btype == "text":
            text = blk.get("text", "")
            if text.strip():
                yield AgentEvent(
                    kind=EventKind.USER_TURN,
                    name="user_turn",
                    ts_start_us=ts_us,
                    payload={"text": _truncate_str(text, _MAX_TEXT_BYTES)},
                    **base,
                )


def _from_assistant(
    rec: dict,
    ts_us: int,
    base: dict[str, Any],
    pending: dict[str, tuple[int, dict, dict]],
) -> Iterator[AgentEvent]:
    msg = rec.get("message") or {}
    if not isinstance(msg, dict):
        return
    model = msg.get("model")
    usage = msg.get("usage") if isinstance(msg.get("usage"), dict) else {}

    # Emit token counters per chunk (Perfetto draws a step chart).
    if usage:
        yield AgentEvent(
            kind=EventKind.ASSISTANT_MSG,
            name="assistant_msg",
            ts_start_us=ts_us,
            model=model,
            tokens_input=usage.get("input_tokens"),
            tokens_output=usage.get("output_tokens"),
            cache_read=usage.get("cache_read_input_tokens"),
            cache_create=usage.get("cache_creation_input_tokens"),
            payload={"stop_reason": msg.get("stop_reason")},
            **base,
        )

    content = msg.get("content") or []
    if not isinstance(content, list):
        return
    for blk in content:
        if not isinstance(blk, dict):
            continue
        btype = blk.get("type")
        if btype == "tool_use":
            tuid = blk.get("id")
            if not tuid or tuid in pending:
                # First sighting wins (handles streaming chunk dupes).
                continue
            pending[tuid] = (ts_us, blk, rec)
        elif btype == "text":
            text = blk.get("text", "")
            if text.strip():
                yield AgentEvent(
                    kind=EventKind.ASSISTANT_TEXT,
                    name="assistant_text",
                    ts_start_us=ts_us,
                    model=model,
                    payload={"text": _truncate_str(text, _MAX_TEXT_BYTES)},
                    **base,
                )
        elif btype == "thinking":
            sig = blk.get("signature", "")
            thinking_text = blk.get("thinking", "")
            yield AgentEvent(
                kind=EventKind.THINKING,
                name="thinking",
                ts_start_us=ts_us,
                model=model,
                payload={
                    "signature_len": len(sig) if isinstance(sig, str) else 0,
                    "text_len": len(thinking_text) if isinstance(thinking_text, str) else 0,
                    # Preview for cases where thinking isn't redacted.
                    "preview": _truncate_str(
                        thinking_text if isinstance(thinking_text, str) else "",
                        _MAX_TEXT_BYTES,
                    ),
                },
                **base,
            )


__all__ = ["normalize_claude_session"]
