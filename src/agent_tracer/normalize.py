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
from agent_tracer.redact import redact_secrets
from agent_tracer.timeutil import iso_to_us

# Truncation limits keep the trace JSON manageable. Perfetto and Chrome
# tracing both choke on multi-MB args.
_MAX_INPUT_BYTES = 2048
_MAX_RESULT_BYTES = 4096
_MAX_TEXT_BYTES = 2048


def _truncate_str(s: str, n: int) -> str:
    """Strip credentials, then truncate to ``n`` bytes with a tail marker.

    Redaction happens *before* truncation so a secret that would otherwise
    sit just past the truncation boundary still gets caught.
    """
    s = redact_secrets(s)
    if len(s) <= n:
        return s
    return s[:n] + f"…<+{len(s) - n}B>"


def _parse_json_loosely(s: str) -> Any:
    """Parse a JSON string; on failure return the original string."""
    import json as _json

    try:
        return _json.loads(s)
    except (ValueError, TypeError):
        return s


def _truncate_payload(obj: Any, n: int) -> Any:
    """Keep structured shape; truncate string values only.

    Strings longer than ``n`` get a tail marker. Lists/dicts are recursed.
    Other primitives pass through. The categorizer needs the structured
    shape; the Perfetto emitter does the final size cap when it serializes.
    """
    if isinstance(obj, str):
        return _truncate_str(obj, n)
    if isinstance(obj, dict):
        return {k: _truncate_payload(v, n) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_truncate_payload(v, n) for v in obj]
    return obj


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
                    "data": _truncate_payload(data, _MAX_INPUT_BYTES),
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
            payload={"input": _truncate_payload(tu_block.get("input"), _MAX_INPUT_BYTES)},
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
            payload={"reason": _truncate_payload(rec.get("toolUseResult"), _MAX_INPUT_BYTES)},
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
                    "input": _truncate_payload(input_obj, _MAX_INPUT_BYTES),
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


# ──────────────────────────────────────────────────────────────────────────
# Codex


def normalize_codex_session(
    records: Iterable[dict],
    *,
    source_session_id: str | None = None,
) -> Iterator[AgentEvent]:
    """Normalize one Codex rollout JSONL into events.

    Codex is single-agent (no subagents). Tool span pairing:

    * ``response_item.function_call`` ↔ ``response_item.function_call_output``
      via ``call_id``. For shell calls, ``event_msg.exec_command_end`` arrives
      between them with ``process_id``, ``parsed_cmd``, and aggregated stdout/
      stderr — we stash that as side data on the span end.
    * ``response_item.custom_tool_call`` ↔ ``response_item.custom_tool_call_output``
      via ``call_id``. ``event_msg.patch_apply_end`` is the runtime-side
      receipt for ``apply_patch`` and carries per-file diffs.

    The session id passed in (or derived from the rollout filename) flows
    onto every event because individual Codex records don't repeat it.
    """
    pending: dict[str, tuple[int, dict]] = {}  # call_id -> (start_ts_us, function_call_payload)
    runtime_meta: dict[str, dict] = {}         # call_id -> exec_command_end / patch_apply_end payload
    session_id = source_session_id or ""
    cwd: str | None = None
    model: str | None = None

    for rec in records:
        ts_raw = rec.get("timestamp")
        if not isinstance(ts_raw, str):
            continue
        try:
            ts_us = iso_to_us(ts_raw)
        except ValueError:
            continue
        rtype = rec.get("type")
        payload = rec.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")

        base = {
            "source": "codex",
            "session_id": session_id,
            "agent_id": None,
            "cwd": cwd,
            "model": model,
        }

        if rtype == "session_meta":
            session_id = session_id or payload.get("id") or ""
            cwd = payload.get("cwd") or cwd
            yield AgentEvent(
                source="codex",
                session_id=session_id,
                kind=EventKind.SESSION_META,
                name="session_meta",
                ts_start_us=ts_us,
                cwd=cwd,
                payload={
                    "originator": payload.get("originator"),
                    "cli_version": payload.get("cli_version"),
                    "model_provider": payload.get("model_provider"),
                },
            )
            continue

        if rtype == "response_item":
            if ptype == "function_call":
                cid = payload.get("call_id")
                if cid:
                    pending.setdefault(cid, (ts_us, payload))
            elif ptype == "function_call_output":
                yield from _emit_codex_tool_call(
                    ptype, payload, ts_us, pending, runtime_meta, base
                )
            elif ptype == "custom_tool_call":
                cid = payload.get("call_id")
                if cid:
                    pending.setdefault(cid, (ts_us, payload))
            elif ptype == "custom_tool_call_output":
                yield from _emit_codex_tool_call(
                    ptype, payload, ts_us, pending, runtime_meta, base
                )
            elif ptype == "reasoning":
                # Codex reasoning is usually encrypted; surface a length proxy.
                content = payload.get("content")
                summary = payload.get("summary") or []
                enc = payload.get("encrypted_content") or ""
                text_len = sum(
                    len(c.get("text", "")) for c in (content or []) if isinstance(c, dict)
                ) if isinstance(content, list) else 0
                yield AgentEvent(
                    kind=EventKind.THINKING,
                    name="reasoning",
                    ts_start_us=ts_us,
                    payload={
                        "text_len": text_len,
                        "summary_len": len(summary),
                        "encrypted_len": len(enc) if isinstance(enc, str) else 0,
                    },
                    **base,
                )
            elif ptype == "message":
                # Assistant or user message in the model-facing log.
                role = payload.get("role")
                if role in ("assistant", "user", "developer"):
                    parts = payload.get("content") or []
                    text = "\n".join(
                        c.get("text", "")
                        for c in parts
                        if isinstance(c, dict) and c.get("type") in ("input_text", "output_text", "text")
                    )
                    if text.strip():
                        kind = EventKind.USER_TURN if role == "user" else EventKind.ASSISTANT_TEXT
                        yield AgentEvent(
                            kind=kind,
                            name=f"{role}_message",
                            ts_start_us=ts_us,
                            payload={"text": _truncate_str(text, _MAX_TEXT_BYTES)},
                            **base,
                        )

        elif rtype == "event_msg":
            if ptype == "user_message":
                msg = payload.get("message", "")
                if msg:
                    yield AgentEvent(
                        kind=EventKind.USER_TURN,
                        name="user_turn",
                        ts_start_us=ts_us,
                        payload={"text": _truncate_str(msg, _MAX_TEXT_BYTES)},
                        **base,
                    )
            elif ptype == "agent_message":
                msg = payload.get("message", "")
                if msg:
                    yield AgentEvent(
                        kind=EventKind.ASSISTANT_TEXT,
                        name="assistant_text",
                        ts_start_us=ts_us,
                        payload={
                            "text": _truncate_str(msg, _MAX_TEXT_BYTES),
                            "phase": payload.get("phase"),
                        },
                        **base,
                    )
            elif ptype == "token_count":
                info = payload.get("info") or {}
                last = info.get("last_token_usage") or {}
                total = info.get("total_token_usage") or {}
                if last:
                    yield AgentEvent(
                        kind=EventKind.ASSISTANT_MSG,
                        name="token_count",
                        ts_start_us=ts_us,
                        tokens_input=last.get("input_tokens"),
                        tokens_output=last.get("output_tokens"),
                        cache_read=last.get("cached_input_tokens"),
                        payload={
                            "total_in": total.get("input_tokens"),
                            "total_out": total.get("output_tokens"),
                            "model_ctx": info.get("model_context_window"),
                            "reasoning_out": last.get("reasoning_output_tokens"),
                        },
                        **base,
                    )
            elif ptype in ("exec_command_end", "patch_apply_end"):
                cid = payload.get("call_id")
                if cid:
                    runtime_meta[cid] = payload
            elif ptype == "task_started":
                yield AgentEvent(
                    kind=EventKind.PROGRESS,
                    name="task_started",
                    ts_start_us=ts_us,
                    payload={"task_id": payload.get("task_id")},
                    **base,
                )
            elif ptype == "task_complete":
                yield AgentEvent(
                    kind=EventKind.PROGRESS,
                    name="task_complete",
                    ts_start_us=ts_us,
                    payload={"task_id": payload.get("task_id")},
                    **base,
                )
            elif ptype == "context_compacted":
                yield AgentEvent(
                    kind=EventKind.COMPACTION,
                    name="context_compacted",
                    ts_start_us=ts_us,
                    payload={},
                    **base,
                )
            elif ptype == "turn_aborted":
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    name="turn_aborted",
                    ts_start_us=ts_us,
                    is_error=True,
                    payload={"reason": _truncate_payload(payload.get("reason"), _MAX_INPUT_BYTES)},
                    **base,
                )
            elif ptype == "error":
                yield AgentEvent(
                    kind=EventKind.ERROR,
                    name="error",
                    ts_start_us=ts_us,
                    is_error=True,
                    payload={"message": _truncate_payload(payload.get("message"), _MAX_INPUT_BYTES)},
                    **base,
                )
        elif rtype == "compacted":
            yield AgentEvent(
                kind=EventKind.COMPACTION,
                name="compacted",
                ts_start_us=ts_us,
                payload={},
                **base,
            )
        # turn_context: cwd/model updates per turn — capture cwd and model for
        # subsequent events.
        if rtype == "turn_context":
            new_cwd = payload.get("cwd")
            new_model = payload.get("model")
            if isinstance(new_cwd, str):
                cwd = new_cwd
            if isinstance(new_model, str):
                model = new_model

    # In-flight at end of stream → orphan.
    for cid, (start_ts, call_payload) in pending.items():
        yield AgentEvent(
            source="codex",
            session_id=session_id,
            kind=EventKind.ERROR,
            name=f"orphan:{call_payload.get('name', '?')}",
            ts_start_us=start_ts,
            tool_use_id=cid,
            payload={"arguments": _truncate_payload(call_payload.get("arguments"), _MAX_INPUT_BYTES)},
        )


def _emit_codex_tool_call(
    output_ptype: str,
    output_payload: dict,
    ts_us: int,
    pending: dict[str, tuple[int, dict]],
    runtime_meta: dict[str, dict],
    base: dict[str, Any],
) -> Iterator[AgentEvent]:
    cid = output_payload.get("call_id")
    if not cid:
        return
    pend = pending.pop(cid, None)
    if pend is None:
        yield AgentEvent(
            kind=EventKind.ERROR,
            name="orphan_tool_output",
            ts_start_us=ts_us,
            tool_use_id=cid,
            is_error=True,
            payload={
                "output_kind": output_ptype,
                "output": _truncate_payload(output_payload.get("output"), _MAX_RESULT_BYTES),
            },
            **base,
        )
        return
    start_ts, call_payload = pend
    tool_name = call_payload.get("name") or "?"
    meta = runtime_meta.pop(cid, {}) or {}
    is_error = False
    exit_code: int | None = None
    extras: dict[str, Any] = {}
    if meta:
        exit_code = meta.get("exit_code") if "exit_code" in meta else None
        extras["process_id"] = meta.get("process_id")
        extras["parsed_cmd"] = meta.get("parsed_cmd")
        if meta.get("type") == "exec_command_end":
            extras["aggregated_output"] = _truncate_payload(
                meta.get("aggregated_output"), _MAX_RESULT_BYTES
            )
        elif meta.get("type") == "patch_apply_end":
            extras["success"] = meta.get("success")
            is_error = meta.get("success") is False
            extras["changes"] = list((meta.get("changes") or {}).keys())
    # Codex serializes ``arguments`` as a JSON-encoded string; parse so the
    # categorizer (and any downstream code) sees structured fields.
    raw_input = call_payload.get("arguments") or call_payload.get("input")
    if isinstance(raw_input, str):
        raw_input = _parse_json_loosely(raw_input)
    yield AgentEvent(
        kind=EventKind.TOOL_CALL,
        name=str(tool_name),
        ts_start_us=start_ts,
        ts_end_us=ts_us,
        tool_use_id=cid,
        is_error=is_error,
        exit_code=exit_code,
        payload={
            "input": _truncate_payload(raw_input, _MAX_INPUT_BYTES),
            "output": _truncate_payload(output_payload.get("output"), _MAX_RESULT_BYTES),
            **{k: v for k, v in extras.items() if v is not None},
        },
        **base,
    )


# ──────────────────────────────────────────────────────────────────────────
# Cursor

# Token estimation: ~4 characters per token is the standard heuristic across
# Claude/GPT tokenizers.  Used by the blob-based estimator below.
_CHARS_PER_TOKEN = 4


def _emit_blob_token_events(
    blobs: list,
    session_id: str,
    session_start_us: int,
    bubbles: list | None,
) -> Iterator[AgentEvent]:
    """Emit ``ASSISTANT_MSG`` events with token estimates from the blob store.

    The blob store (``agentKv:blob:`` in ``state.vscdb``) contains the actual
    API messages including tool results, system prompts, and condensation
    summaries.  Token counts are estimated as ``chars / 4``.

    ``tokens_input`` is the *incremental* new input since the last assistant
    response (matching Claude/Codex semantics).  ``tokens_output`` is what
    the model generated in this response.

    A ``[Previous conversation summary]`` blob resets the input accumulator
    (condensation replaces all prior context with a summary).
    """
    new_input_since_last = 0
    blob_idx = 0
    bubble_iter = iter(bubbles) if bubbles else iter([])

    def _next_assistant_ts() -> int:
        for b in bubble_iter:
            if b.bubble_type == 2 and b.created_at_us > 0:
                return b.created_at_us
        return session_start_us + blob_idx * 1_000_000

    for blob in blobs:
        blob_tokens = max(1, blob.content_chars // _CHARS_PER_TOKEN)

        if blob.is_summary:
            new_input_since_last = blob_tokens
            blob_idx += 1
            continue

        if blob.role in ("system", "user", "tool"):
            new_input_since_last += blob_tokens
        elif blob.role == "assistant":
            yield AgentEvent(
                source="cursor",
                session_id=session_id,
                kind=EventKind.ASSISTANT_MSG,
                name="assistant_msg",
                ts_start_us=_next_assistant_ts(),
                tokens_input=new_input_since_last,
                tokens_output=blob_tokens,
                payload={"estimated": True},
            )
            new_input_since_last = 0

        blob_idx += 1


def normalize_cursor_session(
    records: Iterable[dict],
    *,
    source_session_id: str | None = None,
    bubbles: list | None = None,
    terminal_logs: list | None = None,
    session_start_us: int = 0,
    blobs: list | None = None,
) -> Iterator[AgentEvent]:
    """Normalize one Cursor transcript JSONL into events.

    Produces ``USER_TURN``, ``ASSISTANT_TEXT``, ``TOOL_CALL``, ``THINKING``,
    and (when blobs are available) ``ASSISTANT_MSG`` events.

    Timestamps come from bubble metadata (``state.vscdb``) when available,
    falling back to synthetic 1-second-spaced timestamps from
    ``session_start_us``.

    Shell/Bash tool calls are matched to terminal logs for real durations.
    """
    session_id = source_session_id or ""

    # Emit blob-based token events first (before JSONL-derived events).
    if blobs:
        yield from _emit_blob_token_events(blobs, session_id, session_start_us, bubbles)

    # Terminal-log lookup by command string for Bash/Shell duration matching.
    terminal_by_cmd: dict[str, list] = {}
    for te in terminal_logs or []:
        terminal_by_cmd.setdefault(te.command, []).append(te)

    # Bubble iterator for real timestamps.
    bubble_iter = iter(bubbles) if bubbles else iter([])
    next_bubble: Any = next(bubble_iter, None)

    record_idx = 0

    for rec in records:
        role = rec.get("role")
        content = (rec.get("message") or {}).get("content")
        if not isinstance(content, list):
            record_idx += 1
            continue

        for blk in content:
            if not isinstance(blk, dict):
                continue
            btype = blk.get("type")

            # Consume a matching bubble for timestamp.
            ts_us = session_start_us + record_idx * 1_000_000
            bubble_thinking_ms: int | None = None
            bubble_tool_call_id: str | None = None

            if next_bubble is not None:
                is_match = (
                    (role == "user" and btype == "text" and next_bubble.bubble_type == 1)
                    or (role == "assistant" and next_bubble.bubble_type == 2)
                )
                if is_match:
                    if next_bubble.created_at_us > 0:
                        ts_us = next_bubble.created_at_us
                    bubble_thinking_ms = next_bubble.thinking_duration_ms
                    bubble_tool_call_id = next_bubble.tool_call_id
                    next_bubble = next(bubble_iter, None)

            base: dict[str, Any] = {
                "source": "cursor",
                "session_id": session_id,
                "agent_id": None,
            }

            if btype == "text":
                text = blk.get("text", "")
                if not text.strip():
                    continue
                kind = EventKind.USER_TURN if role == "user" else EventKind.ASSISTANT_TEXT
                name = "user_turn" if role == "user" else "assistant_text"
                yield AgentEvent(
                    kind=kind, name=name, ts_start_us=ts_us,
                    payload={"text": _truncate_str(text, _MAX_TEXT_BYTES)},
                    **base,
                )

            elif btype == "tool_use" and role == "assistant":
                tool_name = blk.get("name", "?")
                raw_input = blk.get("input") or {}

                if bubble_thinking_ms is not None:
                    yield AgentEvent(
                        kind=EventKind.THINKING, name="thinking",
                        ts_start_us=ts_us,
                        payload={"text_len": 0, "duration_ms": bubble_thinking_ms},
                        **base,
                    )

                # Match Bash/Shell to terminal logs for real durations.
                ts_end_us: int | None = None
                exit_code: int | None = None
                if tool_name in ("Bash", "Shell") and isinstance(raw_input, dict):
                    candidates = terminal_by_cmd.get(raw_input.get("command", ""), [])
                    if candidates:
                        te = candidates.pop(0)
                        ts_us = te.started_at_us
                        ts_end_us = te.ended_at_us
                        exit_code = te.exit_code

                yield AgentEvent(
                    kind=EventKind.TOOL_CALL, name=str(tool_name),
                    ts_start_us=ts_us, ts_end_us=ts_end_us,
                    tool_use_id=bubble_tool_call_id, exit_code=exit_code,
                    payload={"input": _truncate_payload(raw_input, _MAX_INPUT_BYTES)},
                    **base,
                )

        record_idx += 1


__all__ = ["normalize_claude_session", "normalize_codex_session", "normalize_cursor_session"]
