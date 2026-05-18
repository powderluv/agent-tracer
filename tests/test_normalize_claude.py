"""Normalizer tests built from synthetic records that mirror real shapes.

Focus: tool_use ↔ tool_result pairing, streaming-chunk deduplication, user
rejections, thinking blocks, subagent metadata, orphan handling.
"""

from __future__ import annotations

from agent_tracer.events import EventKind
from agent_tracer.normalize import normalize_claude_session


def _rec(t: str, ts: str, **kw):
    base = {"type": t, "timestamp": ts, "sessionId": "sess1", "uuid": f"u-{ts}"}
    base.update(kw)
    return base


def _asst(ts: str, content, **kw):
    return _rec(
        "assistant",
        ts,
        message={
            "model": "claude-opus-4-7",
            "content": content,
            "usage": kw.pop("usage", None),
            "stop_reason": kw.pop("stop_reason", None),
        },
        **kw,
    )


def _user(ts: str, content, **kw):
    return _rec("user", ts, message={"role": "user", "content": content}, **kw)


def test_tool_call_pairs_use_and_result_via_id() -> None:
    records = [
        _asst(
            "2026-03-09T19:34:55.278Z",
            [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}}],
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
        _user(
            "2026-03-09T19:34:56.500Z",
            [{"type": "tool_result", "tool_use_id": "tu1", "content": "a\nb\n"}],
        ),
    ]
    events = list(normalize_claude_session(records))
    kinds = [e.kind for e in events]
    assert EventKind.ASSISTANT_MSG in kinds  # token counter event
    tool = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert tool.name == "Bash"
    assert tool.tool_use_id == "tu1"
    assert tool.ts_end_us > tool.ts_start_us
    assert tool.duration_us is not None and tool.duration_us > 0
    # Input is preserved as a structured dict so the categorizer can read fields.
    assert tool.payload["input"] == {"command": "ls"}
    assert "a\nb" in tool.payload["result"]


def test_streaming_chunk_dedups_tool_use_first_sighting_wins() -> None:
    # Same message.id appears across two assistant records; the tool_use's
    # span must start at the EARLIEST sighting, not the latest.
    records = [
        _asst(
            "2026-03-09T19:34:55.000Z",
            [{"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file": "x"}}],
        ),
        _asst(
            "2026-03-09T19:34:55.900Z",
            [{"type": "tool_use", "id": "tu1", "name": "Read", "input": {"file": "x"}}],
        ),
        _user(
            "2026-03-09T19:34:56.000Z",
            [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
        ),
    ]
    events = list(normalize_claude_session(records))
    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    # 1000ms (start at .000, end at .000 of next second = 1000000us)
    assert tools[0].duration_us == 1_000_000


def test_orphan_tool_result_is_recorded_as_error() -> None:
    records = [
        _user(
            "2026-03-09T19:34:56.000Z",
            [{"type": "tool_result", "tool_use_id": "missing", "content": "?"}],
        ),
    ]
    events = list(normalize_claude_session(records))
    assert any(e.kind == EventKind.ERROR and e.name == "orphan_tool_result" for e in events)


def test_unclosed_tool_use_emitted_as_orphan_at_end() -> None:
    records = [
        _asst(
            "2026-03-09T19:34:55.000Z",
            [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}}],
        ),
    ]
    events = list(normalize_claude_session(records))
    errs = [e for e in events if e.kind == EventKind.ERROR]
    assert any("orphan:Bash" in e.name for e in errs)


def test_user_rejection_record_becomes_error_instant() -> None:
    rec = {
        "type": "user",
        "timestamp": "2026-03-19T07:51:57.838Z",
        "sessionId": "sess1",
        "uuid": "u1",
        "toolUseResult": "User rejected tool use",
        "sourceToolAssistantUUID": "x",
    }
    events = list(normalize_claude_session([rec]))
    assert events and events[0].kind == EventKind.ERROR
    assert events[0].name == "user_rejected"
    assert events[0].is_error is True


def test_thinking_block_emits_length_proxies() -> None:
    rec = _asst(
        "2026-03-09T19:34:55.000Z",
        [{"type": "thinking", "thinking": "blah blah", "signature": "sig"}],
    )
    events = list(normalize_claude_session([rec]))
    th = next(e for e in events if e.kind == EventKind.THINKING)
    assert th.payload["text_len"] == len("blah blah")
    assert th.payload["signature_len"] == len("sig")


def test_agent_tool_extracts_subagent_type() -> None:
    records = [
        _asst(
            "2026-03-09T19:34:55.000Z",
            [
                {
                    "type": "tool_use",
                    "id": "tu1",
                    "name": "Agent",
                    "input": {"subagent_type": "Explore", "prompt": "find x"},
                }
            ],
        ),
        _user(
            "2026-03-09T19:34:56.000Z",
            [{"type": "tool_result", "tool_use_id": "tu1", "content": "result"}],
        ),
    ]
    events = list(normalize_claude_session(records))
    tool = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert tool.subagent_type == "Explore"


def test_unknown_top_level_types_are_skipped_quietly() -> None:
    records = [
        {"type": "attachment", "timestamp": "2026-03-09T19:34:55.000Z", "uuid": "x", "sessionId": "s"},
        {"type": "permission-mode", "timestamp": "2026-03-09T19:34:55.000Z", "uuid": "x", "sessionId": "s"},
        {"type": "pr-link", "timestamp": "2026-03-09T19:34:55.000Z", "uuid": "x", "sessionId": "s"},
    ]
    events = list(normalize_claude_session(records))
    assert events == []
