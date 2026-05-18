"""Stats aggregator: per-session and overall totals."""

from __future__ import annotations

from agent_tracer.events import AgentEvent, EventKind
from agent_tracer.stats import _command_signature, compute_stats


def _tool(name, ts_start, ts_end, source="claude", session_id="s1", **kw):
    return AgentEvent(
        source=source,
        session_id=session_id,
        kind=EventKind.TOOL_CALL,
        name=name,
        ts_start_us=ts_start,
        ts_end_us=ts_end,
        **kw,
    )


def _msg(ts, *, source="claude", session_id="s1", **kw):
    return AgentEvent(
        source=source,
        session_id=session_id,
        kind=EventKind.ASSISTANT_MSG,
        name="assistant_msg",
        ts_start_us=ts,
        **kw,
    )


def test_session_summary_counts_tools_and_tokens() -> None:
    events = [
        _tool("Bash", 1_000_000, 1_500_000, payload={"input": {"command": "ls"}}),
        _tool("Read", 2_000_000, 2_010_000, payload={"input": {"file_path": "/x"}}),
        _msg(2_500_000, tokens_input=100, tokens_output=20, cache_read=80, cache_create=20),
        _msg(3_000_000, tokens_input=50, tokens_output=10, cache_read=50, cache_create=0),
    ]
    r = compute_stats(events)
    s = r.sessions[("claude", "s1")]
    assert s.tool_calls == 2
    assert s.tokens_input == 150
    assert s.tokens_output == 30
    assert s.cache_read == 130
    assert s.cache_create == 20
    # Hit rate = cache_read / (cache_read + cache_create)
    assert s.cache_hit_rate is not None and abs(s.cache_hit_rate - (130 / 150)) < 1e-6
    assert s.wallclock_s == 2.0  # 1.0s to 3.0s
    assert s.by_tool["Bash"] == 1
    assert s.by_tool["Read"] == 1
    assert s.tool_wall_us == 500_000 + 10_000


def test_overall_top_tools_and_wall() -> None:
    events = [
        _tool("Bash", 0, 2_000_000),
        _tool("Bash", 3_000_000, 3_500_000),
        _tool("Read", 4_000_000, 4_010_000),
    ]
    r = compute_stats(events)
    assert r.overall_by_tool["Bash"] == 2
    assert r.overall_by_tool["Read"] == 1
    assert r.overall_tool_wall_us["Bash"] == 2_500_000
    assert r.overall_tool_wall_us["Read"] == 10_000


def test_top_commands_extracts_signature_from_bash_input() -> None:
    events = [
        _tool("Bash", 0, 1, payload={"input": {"command": "ninja clr+dist"}}),
        _tool("Bash", 2, 3, payload={"input": {"command": "ninja clr+expunge"}}),
        _tool("Bash", 4, 5, payload={"input": {"command": "ninja clr+dist -j 32"}}),
        _tool(
            "exec_command",
            6,
            7,
            payload={"parsed_cmd": [{"type": "unknown", "cmd": "git status"}]},
        ),
    ]
    r = compute_stats(events)
    assert r.top_commands["ninja clr+dist"] == 2
    assert r.top_commands["ninja clr+expunge"] == 1
    assert r.top_commands["git status"] == 1


def test_command_signature_strips_sudo_and_sh_c() -> None:
    assert _command_signature("sudo pip install -e .") == "pip install"
    # Unquoted bash -lc form (matches what Codex emits via parsed_cmd[0].cmd).
    assert _command_signature("bash -lc ninja clr+dist") == "ninja clr+dist"
    assert _command_signature("ls -la") == "ls -la"
