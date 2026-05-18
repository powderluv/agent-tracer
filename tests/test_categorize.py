"""Categorizer: tool name + command classification."""

from __future__ import annotations

from agent_tracer.categorize import categorize
from agent_tracer.events import AgentEvent, EventKind


def _tool(source: str, name: str, input_=None, payload_extra=None, **kw) -> AgentEvent:
    payload = {"input": input_}
    if payload_extra:
        payload.update(payload_extra)
    return AgentEvent(
        source=source,  # type: ignore[arg-type]
        session_id="s",
        kind=EventKind.TOOL_CALL,
        name=name,
        ts_start_us=0,
        ts_end_us=1,
        payload=payload,
        **kw,
    )


def _text(source: str, text: str, kind: EventKind = EventKind.ASSISTANT_TEXT) -> AgentEvent:
    return AgentEvent(
        source=source,  # type: ignore[arg-type]
        session_id="s",
        kind=kind,
        name="assistant_text",
        ts_start_us=0,
        payload={"text": text},
    )


def test_claude_static_tool_lookup() -> None:
    assert categorize(_tool("claude", "Read")) == "fs"
    assert categorize(_tool("claude", "Grep")) == "fs"
    assert categorize(_tool("claude", "Edit")) == "editor"
    assert categorize(_tool("claude", "Write")) == "editor"
    assert categorize(_tool("claude", "Agent")) == "agent"
    assert categorize(_tool("claude", "WebFetch")) == "network"


def test_codex_static_tool_lookup() -> None:
    assert categorize(_tool("codex", "apply_patch")) == "editor"
    assert categorize(_tool("codex", "update_plan")) == "agent"
    assert categorize(_tool("codex", "web_search")) == "network"


def test_bash_command_classification() -> None:
    assert categorize(_tool("claude", "Bash", {"command": "rocm-smi --json"})) == "gpu_query"
    assert categorize(_tool("claude", "Bash", {"command": "nvidia-smi -q"})) == "gpu_query"
    assert categorize(_tool("claude", "Bash", {"command": "hipcc -O2 foo.cpp"})) == "gpu"
    assert categorize(_tool("claude", "Bash", {"command": "ninja -j 32 clr+dist"})) == "build"
    assert categorize(_tool("claude", "Bash", {"command": "cmake -B build -S ."})) == "build"
    assert categorize(_tool("claude", "Bash", {"command": "pytest -xvs"})) == "test"
    assert categorize(_tool("claude", "Bash", {"command": "git status --short"})) == "git"
    assert categorize(_tool("claude", "Bash", {"command": "gh pr view 123"})) == "git"
    assert (
        categorize(_tool("claude", "Bash", {"command": "sshpass -p x ssh nod@h"}))
        == "network"
    )
    assert categorize(_tool("claude", "Bash", {"command": "ls -la /tmp"})) == "fs"
    assert categorize(_tool("claude", "Bash", {"command": "echo hi"})) == "shell"


def test_codex_exec_command_uses_parsed_cmd_when_no_explicit_command_key() -> None:
    ev = _tool(
        "codex",
        "exec_command",
        input_=None,
        payload_extra={"parsed_cmd": [{"type": "unknown", "cmd": "ninja clr+dist"}]},
    )
    assert categorize(ev) == "build"


def test_codex_exec_command_classifies_argument_command() -> None:
    ev = _tool("codex", "exec_command", input_={"cmd": "rocminfo"})
    assert categorize(ev) == "gpu_query"


def test_gpu_text_terms_tag_assistant_text_as_gpu() -> None:
    # Discussing GPU concepts in assistant text tags the turn as gpu so it
    # surfaces in the trace's gpu category even without a shell command.
    assert categorize(_text("claude", "let me check the MES IC_BASE register")) == "gpu"
    assert categorize(_text("claude", "PSP GFX error 0xFFFF0006")) == "gpu"
    assert categorize(_text("claude", "hello world")) == "text"


def test_kinds_map_to_distinct_categories() -> None:
    assert (
        categorize(
            AgentEvent(
                source="claude",
                session_id="s",
                kind=EventKind.THINKING,
                name="thinking",
                ts_start_us=0,
            )
        )
        == "thinking"
    )
    assert (
        categorize(
            AgentEvent(
                source="codex",
                session_id="s",
                kind=EventKind.ASSISTANT_MSG,
                name="token_count",
                ts_start_us=0,
            )
        )
        == "model"
    )
    assert (
        categorize(
            AgentEvent(
                source="codex",
                session_id="s",
                kind=EventKind.COMPACTION,
                name="compacted",
                ts_start_us=0,
            )
        )
        == "meta"
    )
    assert (
        categorize(
            AgentEvent(
                source="claude",
                session_id="s",
                kind=EventKind.ERROR,
                name="user_rejected",
                ts_start_us=0,
                is_error=True,
            )
        )
        == "error"
    )


def test_unknown_tool_falls_back_to_tool() -> None:
    assert categorize(_tool("claude", "SomeFutureTool")) == "tool"
    assert categorize(_tool("codex", "some_new_function")) == "tool"
