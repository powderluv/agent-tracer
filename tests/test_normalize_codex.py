"""Codex normalizer tests: tool-call pairing, runtime metadata, token counts."""

from __future__ import annotations

from agent_tracer.events import EventKind
from agent_tracer.normalize import normalize_codex_session


def _r(ts: str, rtype: str, payload: dict) -> dict:
    return {"timestamp": ts, "type": rtype, "payload": payload}


def test_function_call_pairs_with_function_call_output_via_call_id() -> None:
    recs = [
        _r("2026-04-27T07:34:44.799Z", "session_meta", {"id": "sess-abc", "cwd": "/x"}),
        _r(
            "2026-04-27T07:34:59.840Z",
            "response_item",
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "c1",
                "arguments": '{"cmd": "ls"}',
            },
        ),
        _r(
            "2026-04-27T07:34:59.885Z",
            "response_item",
            {"type": "function_call_output", "call_id": "c1", "output": "a\nb"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="sess-abc"))
    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    t = tools[0]
    assert t.name == "exec_command"
    assert t.tool_use_id == "c1"
    assert t.ts_end_us > t.ts_start_us
    assert t.payload["input"] and "ls" in t.payload["input"]
    assert "a\nb" in t.payload["output"]
    assert t.session_id == "sess-abc"


def test_exec_command_end_metadata_attaches_to_span() -> None:
    recs = [
        _r(
            "2026-04-27T07:34:59.840Z",
            "response_item",
            {"type": "function_call", "name": "exec_command", "call_id": "c1", "arguments": "{}"},
        ),
        _r(
            "2026-04-27T07:34:59.884Z",
            "event_msg",
            {
                "type": "exec_command_end",
                "call_id": "c1",
                "process_id": "1234",
                "parsed_cmd": [{"type": "unknown", "cmd": "git status"}],
                "aggregated_output": "ok",
            },
        ),
        _r(
            "2026-04-27T07:34:59.885Z",
            "response_item",
            {"type": "function_call_output", "call_id": "c1", "output": "x"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    t = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert t.payload["process_id"] == "1234"
    assert t.payload["parsed_cmd"][0]["cmd"] == "git status"
    assert "ok" in t.payload["aggregated_output"]


def test_custom_tool_call_pair_via_call_id_and_patch_apply_metadata() -> None:
    recs = [
        _r(
            "2026-04-27T07:42:08.736Z",
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "c2",
                "name": "apply_patch",
                "input": "*** Begin Patch ***",
            },
        ),
        _r(
            "2026-04-27T07:42:08.745Z",
            "event_msg",
            {
                "type": "patch_apply_end",
                "call_id": "c2",
                "success": True,
                "changes": {"/x/a.md": {"type": "modify"}, "/x/b.md": {"type": "modify"}},
            },
        ),
        _r(
            "2026-04-27T07:42:08.780Z",
            "response_item",
            {"type": "custom_tool_call_output", "call_id": "c2", "output": "Success"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    t = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert t.name == "apply_patch"
    assert t.is_error is False
    assert set(t.payload["changes"]) == {"/x/a.md", "/x/b.md"}


def test_patch_apply_failure_marks_span_as_error() -> None:
    recs = [
        _r(
            "2026-04-27T07:42:08.736Z",
            "response_item",
            {"type": "custom_tool_call", "call_id": "c3", "name": "apply_patch", "input": ""},
        ),
        _r(
            "2026-04-27T07:42:08.745Z",
            "event_msg",
            {"type": "patch_apply_end", "call_id": "c3", "success": False, "changes": {}},
        ),
        _r(
            "2026-04-27T07:42:08.780Z",
            "response_item",
            {"type": "custom_tool_call_output", "call_id": "c3", "output": "fail"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    t = next(e for e in events if e.kind == EventKind.TOOL_CALL)
    assert t.is_error is True


def test_token_count_with_info_emits_assistant_msg_with_counters() -> None:
    recs = [
        _r(
            "2026-04-27T07:34:59.882Z",
            "event_msg",
            {
                "type": "token_count",
                "info": {
                    "total_token_usage": {"input_tokens": 12000, "output_tokens": 400},
                    "last_token_usage": {
                        "input_tokens": 12776,
                        "cached_input_tokens": 11648,
                        "output_tokens": 454,
                        "reasoning_output_tokens": 197,
                    },
                    "model_context_window": 258400,
                },
            },
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    a = next(e for e in events if e.kind == EventKind.ASSISTANT_MSG)
    assert a.tokens_input == 12776
    assert a.tokens_output == 454
    assert a.cache_read == 11648
    assert a.payload["total_in"] == 12000


def test_user_and_agent_messages_become_instants() -> None:
    recs = [
        _r("2026-04-27T07:34:44.804Z", "event_msg", {"type": "user_message", "message": "hi"}),
        _r(
            "2026-04-27T07:34:59.828Z",
            "event_msg",
            {"type": "agent_message", "message": "hello", "phase": "commentary"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    kinds = [(e.kind, e.name) for e in events]
    assert (EventKind.USER_TURN, "user_turn") in kinds
    assert (EventKind.ASSISTANT_TEXT, "assistant_text") in kinds


def test_reasoning_emits_length_proxies() -> None:
    recs = [
        _r(
            "2026-04-27T07:34:55.244Z",
            "response_item",
            {
                "type": "reasoning",
                "summary": [],
                "content": None,
                "encrypted_content": "gAAA" * 100,
            },
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    th = next(e for e in events if e.kind == EventKind.THINKING)
    assert th.payload["encrypted_len"] == 400


def test_turn_context_updates_cwd_and_model_for_subsequent_events() -> None:
    recs = [
        _r("2026-04-27T07:00:00Z", "turn_context", {"cwd": "/proj", "model": "gpt-5"}),
        _r("2026-04-27T07:00:01Z", "event_msg", {"type": "user_message", "message": "hi"}),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    u = next(e for e in events if e.kind == EventKind.USER_TURN)
    assert u.cwd == "/proj"
    assert u.model == "gpt-5"


def test_orphan_function_call_emitted_at_stream_end() -> None:
    recs = [
        _r(
            "2026-04-27T07:00:00Z",
            "response_item",
            {"type": "function_call", "name": "exec_command", "call_id": "x", "arguments": "{}"},
        ),
    ]
    events = list(normalize_codex_session(recs, source_session_id="s"))
    errs = [e for e in events if e.kind == EventKind.ERROR]
    assert any("orphan:exec_command" in e.name for e in errs)
