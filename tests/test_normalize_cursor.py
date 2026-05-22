"""Normalizer tests for Cursor agent transcripts.

Focus: bubble-timestamp enrichment, instant vs Bash-span tool calls,
synthetic timestamp fallback, thinking events, blob-based token estimation.
"""

from __future__ import annotations

from agent_tracer.events import EventKind
from agent_tracer.normalize import normalize_cursor_session
from agent_tracer.parsers.cursor import BubbleInfo, ConversationBlob, TerminalEntry


def _user(text: str):
    return {"role": "user", "message": {"content": [{"type": "text", "text": text}]}}


def _assistant_text(text: str):
    return {"role": "assistant", "message": {"content": [{"type": "text", "text": text}]}}


def _assistant_tool(name: str, input_: dict | None = None):
    return {
        "role": "assistant",
        "message": {
            "content": [{"type": "tool_use", "name": name, "input": input_ or {}}],
        },
    }


def _bubble(bid: str, ts_us: int, btype: int, **kw):
    return BubbleInfo(
        bubble_id=bid, created_at_us=ts_us, bubble_type=btype,
        thinking_duration_ms=kw.get("thinking_ms"),
        tool_call_id=kw.get("tool_call_id"),
    )


def _blob(role: str, content_chars: int, **kw):
    return ConversationBlob(
        role=role, content_chars=content_chars,
        is_summary=kw.get("is_summary", False),
    )


# ---------------------------------------------------------------------------
# Basic record types
# ---------------------------------------------------------------------------

def test_user_text_becomes_user_turn():
    events = list(normalize_cursor_session([_user("hello")], session_start_us=1_000_000))
    turns = [e for e in events if e.kind == EventKind.USER_TURN]
    assert len(turns) == 1
    assert turns[0].source == "cursor"
    assert "hello" in turns[0].payload.get("text", "")


def test_assistant_text_becomes_assistant_text():
    events = list(normalize_cursor_session(
        [_assistant_text("thinking out loud")], session_start_us=2_000_000,
    ))
    texts = [e for e in events if e.kind == EventKind.ASSISTANT_TEXT]
    assert len(texts) == 1
    assert texts[0].source == "cursor"


def test_tool_use_emits_instant_tool_call():
    events = list(normalize_cursor_session(
        [_assistant_tool("Read", {"path": "/tmp/foo"})], session_start_us=3_000_000,
    ))
    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    assert tools[0].name == "Read"
    assert tools[0].ts_end_us is None
    assert tools[0].duration_us is None


def test_empty_text_skipped():
    events = list(normalize_cursor_session(
        [{"role": "assistant", "message": {"content": [{"type": "text", "text": "  "}]}}],
    ))
    assert len(events) == 0


def test_missing_content_skipped():
    events = list(normalize_cursor_session([{"role": "user", "message": {}}]))
    assert len(events) == 0


# ---------------------------------------------------------------------------
# Synthetic timestamps (no bubbles)
# ---------------------------------------------------------------------------

def test_synthetic_timestamps_without_bubbles():
    records = [_user("q1"), _assistant_text("a1"), _user("q2")]
    events = list(normalize_cursor_session(records, session_start_us=10_000_000))
    content = [e for e in events if e.kind in (EventKind.USER_TURN, EventKind.ASSISTANT_TEXT)]
    assert content[0].ts_start_us == 10_000_000
    assert content[1].ts_start_us == 11_000_000
    assert content[2].ts_start_us == 12_000_000


# ---------------------------------------------------------------------------
# Bubble-timestamp enrichment
# ---------------------------------------------------------------------------

def test_real_timestamps_from_bubbles():
    records = [_user("q"), _assistant_text("a")]
    bubbles = [_bubble("b1", 100_000_000, 1), _bubble("b2", 105_000_000, 2)]
    events = list(normalize_cursor_session(records, bubbles=bubbles, session_start_us=0))
    content = [e for e in events if e.kind in (EventKind.USER_TURN, EventKind.ASSISTANT_TEXT)]
    assert content[0].ts_start_us == 100_000_000
    assert content[1].ts_start_us == 105_000_000


def test_thinking_duration_from_bubbles():
    records = [_assistant_tool("Read", {"path": "/tmp/x"})]
    bubbles = [_bubble("b1", 300_000_000, 2, thinking_ms=1500, tool_call_id="tc1")]
    events = list(normalize_cursor_session(records, bubbles=bubbles))

    thinking = [e for e in events if e.kind == EventKind.THINKING]
    assert len(thinking) == 1
    assert thinking[0].payload.get("duration_ms") == 1500

    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    assert tools[0].tool_use_id == "tc1"


# ---------------------------------------------------------------------------
# Blob-based token estimation
# ---------------------------------------------------------------------------

def test_blob_token_events_emitted():
    records = [_user("q"), _assistant_text("a")]
    blobs = [
        _blob("user", 100),
        _blob("assistant", 200),
    ]
    events = list(normalize_cursor_session(records, blobs=blobs, session_start_us=0))
    msgs = [e for e in events if e.kind == EventKind.ASSISTANT_MSG]
    assert len(msgs) == 1
    # input = user blob (100 chars / 4 = 25 tokens)
    assert msgs[0].tokens_input == 25
    # output = assistant blob (200 chars / 4 = 50 tokens)
    assert msgs[0].tokens_output == 50
    assert msgs[0].payload.get("estimated") is True


def test_blob_summary_resets_context():
    blobs = [
        _blob("system", 400),          # 100 tokens
        _blob("user", 200),             # 50 tokens → cumulative input = 150
        _blob("assistant", 100),        # emits: input=150, output=25
        _blob("user", 40000, is_summary=True),  # condensation → resets to 10000
        _blob("user", 200),             # 50 tokens → input = 10000 + 50
        _blob("assistant", 100),        # emits: input=10050, output=25
    ]
    events = list(normalize_cursor_session([], blobs=blobs, session_start_us=0))
    msgs = [e for e in events if e.kind == EventKind.ASSISTANT_MSG]
    assert len(msgs) == 2
    # First assistant: input = system(100) + user(50) = 150
    assert msgs[0].tokens_input == 150
    # After summary (10000) + user(50) = 10050
    assert msgs[1].tokens_input == 10050


def test_no_blob_events_without_blobs():
    records = [_user("q"), _assistant_text("a")]
    events = list(normalize_cursor_session(records, session_start_us=0))
    msgs = [e for e in events if e.kind == EventKind.ASSISTANT_MSG]
    assert len(msgs) == 0  # no blobs → no token events


# ---------------------------------------------------------------------------
# Terminal-log enrichment (Bash commands get real spans)
# ---------------------------------------------------------------------------

def test_bash_tool_matched_to_terminal_log():
    records = [_assistant_tool("Bash", {"command": "git status"})]
    terminal_logs = [
        TerminalEntry(
            command="git status", started_at_us=400_000_000,
            ended_at_us=402_000_000, elapsed_ms=2000,
            exit_code=0, cwd="/tmp",
        ),
    ]
    events = list(normalize_cursor_session(
        records, terminal_logs=terminal_logs, session_start_us=0,
    ))
    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    assert tools[0].ts_start_us == 400_000_000
    assert tools[0].ts_end_us == 402_000_000
    assert tools[0].exit_code == 0


def test_bash_without_terminal_match_stays_instant():
    records = [_assistant_tool("Bash", {"command": "ls -la"})]
    events = list(normalize_cursor_session(
        records, terminal_logs=[], session_start_us=500_000_000,
    ))
    tools = [e for e in events if e.kind == EventKind.TOOL_CALL]
    assert len(tools) == 1
    assert tools[0].ts_end_us is None


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------

def test_session_id_flows_through():
    events = list(normalize_cursor_session([_user("hi")], source_session_id="abc-123"))
    assert events[0].session_id == "abc-123"


def test_source_is_cursor():
    events = list(normalize_cursor_session([_user("hi")]))
    assert events[0].source == "cursor"
