"""Perfetto emitter: pid/tid allocation, span shape, counter shape."""

from __future__ import annotations

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.perfetto import TraceBuilder, build_trace


def _ev(**kw) -> AgentEvent:
    base = {
        "source": "claude",
        "session_id": "abc12345",
        "kind": EventKind.TOOL_CALL,
        "name": "Bash",
        "ts_start_us": 1_000_000,
        "ts_end_us": 1_500_000,
    }
    base.update(kw)
    return AgentEvent(**base)


def test_single_tool_call_emits_complete_event_with_dur_and_metadata() -> None:
    trace = build_trace([_ev()])
    events = trace["traceEvents"]
    assert trace["displayTimeUnit"] == "ms"
    # process_name, process_sort, thread_name, thread_sort, the X event.
    kinds = [e.get("ph") for e in events]
    assert "M" in kinds and "X" in kinds
    x = next(e for e in events if e["ph"] == "X")
    assert x["name"] == "Bash"
    assert x["dur"] == 500_000
    assert x["ts"] == 1_000_000
    assert x["pid"] == 1
    assert x["tid"] == 0
    assert x["cat"] == "tool"


def test_distinct_sessions_get_distinct_pids() -> None:
    trace = build_trace(
        [
            _ev(session_id="aaa"),
            _ev(session_id="bbb"),
        ]
    )
    xs = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    assert {x["pid"] for x in xs} == {1, 2}


def test_subagent_gets_separate_tid_within_session() -> None:
    trace = build_trace(
        [
            _ev(agent_id=None, name="Read"),
            _ev(agent_id="a1234abcd", name="Bash"),
            _ev(agent_id="b9876defg", name="Grep"),
            _ev(agent_id="a1234abcd", name="Edit"),  # back to first subagent
        ]
    )
    xs = [e for e in trace["traceEvents"] if e["ph"] == "X"]
    tids_by_name = {x["name"]: x["tid"] for x in xs}
    # Main = 0, first subagent = 1, second subagent = 2.
    assert tids_by_name["Read"] == 0
    assert tids_by_name["Bash"] == 1
    assert tids_by_name["Grep"] == 2
    assert tids_by_name["Edit"] == 1


def test_zero_duration_span_gets_dur_one() -> None:
    trace = build_trace([_ev(ts_start_us=1_000_000, ts_end_us=1_000_000)])
    x = next(e for e in trace["traceEvents"] if e["ph"] == "X")
    assert x["dur"] == 1


def test_token_counter_event_emitted_on_main_lane() -> None:
    ev = AgentEvent(
        source="claude",
        session_id="abc",
        agent_id="sub1",  # even on a subagent, counter goes to main
        kind=EventKind.ASSISTANT_MSG,
        name="assistant_msg",
        ts_start_us=2_000_000,
        tokens_input=100,
        tokens_output=50,
        cache_read=20,
        cache_create=30,
    )
    # Plant a span first so the subagent lane exists.
    builder = TraceBuilder()
    builder.add(_ev(agent_id="sub1"))
    builder.add(ev)
    trace = builder.finalize()
    counters = [e for e in trace["traceEvents"] if e["ph"] == "C"]
    assert len(counters) == 1
    c = counters[0]
    assert c["tid"] == 0  # main lane regardless of subagent
    assert c["args"]["in"] == 100 and c["args"]["out"] == 50
    assert c["args"]["cache_r"] == 20 and c["args"]["cache_w"] == 30


def test_subagent_type_appears_in_span_name() -> None:
    trace = build_trace(
        [
            _ev(
                name="Agent",
                subagent_type="Explore",
            )
        ]
    )
    x = next(e for e in trace["traceEvents"] if e["ph"] == "X")
    assert x["name"] == "Agent:Explore"


def test_error_tool_call_produces_marker_instant() -> None:
    trace = build_trace([_ev(is_error=True, tool_use_id="tu1")])
    instants = [e for e in trace["traceEvents"] if e["ph"] == "i"]
    assert any(e["name"].startswith("!") for e in instants)


def test_metadata_events_precede_data_events() -> None:
    trace = build_trace([_ev()])
    events = trace["traceEvents"]
    # All ``M`` events come before any non-``M`` event.
    last_m = max(i for i, e in enumerate(events) if e["ph"] == "M")
    first_nonm = min(i for i, e in enumerate(events) if e["ph"] != "M")
    assert last_m < first_nonm
