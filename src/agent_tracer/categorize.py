"""Content miner: tag ``AgentEvent`` instances with a category string.

Categories drive Perfetto's category filter and feed downstream stats /
detectors. They are deliberately coarse — fine-grained detectors live in
``agent_tracer.hints``.

Cat values
----------
* ``gpu_query``  — rocm-smi, nvidia-smi, hipInfo, amd-smi, rocminfo
* ``gpu``        — hipcc/nvcc, rocprof, GPU benchmark binaries, any
                    command with the cwd/branch/text strongly implying GPU
                    work (MES, PSP, IC_BASE, gfx12 …)
* ``build``      — cmake, ninja, make, bazel, cargo build, gcc/g++/clang
                    invocations
* ``test``       — pytest, ctest, gtest, ``test_*`` executables, npm test
* ``git``        — git, gh
* ``network``    — ssh, sshpass, scp, rsync, curl, wget
* ``fs``         — Read/Write/Edit/Glob/Grep tools; ``ls``/``cat``/``find``/
                    ``head``/``tail`` shell commands; apply_patch
* ``agent``      — Agent (subagent dispatch), Task* tools, ScheduleWakeup,
                    Skill
* ``editor``     — apply_patch (Codex) and Edit/Write (Claude) — these
                    overlap with ``fs`` but are kept distinct because
                    detectors care about modification rate vs read rate
* ``text``       — assistant/user message bodies
* ``model``      — assistant_msg counters (token deltas)
* ``meta``       — session_meta, compaction, progress
* ``error``      — error / orphan / user_rejected events
* ``shell``      — Bash/exec_command that didn't match any of the above
"""

from __future__ import annotations

import re
from typing import Any

from agent_tracer.events import AgentEvent, EventKind

# --- regex tables ----------------------------------------------------------

# Match command verbs at a word boundary. Patterns are tried top-to-bottom;
# the first match wins.
_CMD_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("gpu_query", re.compile(r"\b(rocm-smi|nvidia-smi|amd-smi|rocminfo|hipInfo|nvidia-debugdump)\b")),
    ("gpu", re.compile(r"\b(hipcc|nvcc|rocprof|rocgdb|hipify|hipify-perl|hipify-clang|rocprofv\d?|rocm-bandwidth-test|nccl-tests?)\b")),
    ("build", re.compile(r"\b(cmake|ninja|make|bazel|meson|cargo\s+build|cargo\s+check|gcc|g\+\+|clang(\+\+)?|ld|ar|ranlib|nvc|nvc\+\+|ifort|hipfort|configure|autoreconf|setup\.py\s+build|pip\s+install|uv\s+pip|maturin)\b")),
    ("test", re.compile(r"\b(pytest|ctest|gtest|cargo\s+test|npm\s+test|hipTest|jest|vitest|tox|nox|hyperfine\s)\b")),
    ("git", re.compile(r"\b(git|gh)\b")),
    ("network", re.compile(r"\b(ssh|sshpass|scp|rsync|curl|wget|aria2c|axel)\b")),
    ("fs", re.compile(r"\b(ls|cat|head|tail|find|locate|stat|du|df|grep|rg|fd|tree|file|wc|sort|uniq|sed|awk|tar|zip|unzip|gzip|gunzip|mkdir|cp|mv|chmod|chown|touch|ln|readlink|realpath|basename|dirname|pwd)\b")),
    ("network", re.compile(r"\b(ping|traceroute|nslookup|dig|host|nmap|nc|netstat|ss|ip\s+addr|ip\s+route)\b")),
]

# Heuristics over freeform text — used on assistant_text/user_turn and as a
# bump for tool-call commands that mention these terms.
_GPU_TEXT_TERMS = re.compile(
    r"\b("
    r"MES|PSP|RLC|IMU|SDMA|GFXOFF|HQD|IC_BASE|GFX\d{2,}|"
    r"gfx1\d{3}|gfx9\d{2}|amdgpu|rocm|hsa|hsakmt|ROCr|hipcc|"
    r"kernel\s+launch|VRAM|GPU\d|GPU mem|VFIO|gfxoff"
    r")\b"
)


# --- tool-name → category --------------------------------------------------

_CLAUDE_TOOL_CATEGORY: dict[str, str] = {
    "Read": "fs",
    "Glob": "fs",
    "Grep": "fs",
    "Edit": "editor",
    "Write": "editor",
    "NotebookEdit": "editor",
    "MultiEdit": "editor",
    "Agent": "agent",
    "TaskCreate": "agent",
    "TaskUpdate": "agent",
    "TaskList": "agent",
    "TaskGet": "agent",
    "TaskOutput": "agent",
    "TaskStop": "agent",
    "ScheduleWakeup": "agent",
    "Skill": "agent",
    "Monitor": "agent",
    "WebFetch": "network",
    "WebSearch": "network",
    "AskUserQuestion": "agent",
    "ExitPlanMode": "agent",
    "EnterPlanMode": "agent",
    "ToolSearch": "agent",
    "ShareOnboardingGuide": "network",
    # Bash handled separately via command classifier.
}

_CODEX_TOOL_CATEGORY: dict[str, str] = {
    "apply_patch": "editor",
    "update_plan": "agent",
    "view_image": "fs",
    "web_search": "network",
    # exec_command handled separately.
}


# --- public API ------------------------------------------------------------


def categorize(event: AgentEvent) -> str | None:
    """Return a category string for ``event``, or ``None`` if uncategorized."""
    kind = event.kind
    if kind == EventKind.ERROR:
        return "error"
    if kind == EventKind.THINKING:
        return "thinking"
    if kind == EventKind.PROGRESS or kind == EventKind.COMPACTION or kind == EventKind.SESSION_META:
        return "meta"
    if kind == EventKind.ASSISTANT_MSG:
        return "model"
    if kind in (EventKind.ASSISTANT_TEXT, EventKind.USER_TURN):
        # Heuristic GPU tagging on freeform text — useful when the user is
        # discussing GPU work without invoking a tool.
        text = ""
        if isinstance(event.payload, dict):
            text = str(event.payload.get("text", ""))
        if text and _GPU_TEXT_TERMS.search(text):
            return "gpu"
        return "text"
    if kind == EventKind.TOOL_CALL:
        return _categorize_tool(event)
    return None


def categorize_in_place(event: AgentEvent) -> AgentEvent:
    """Set ``event.category`` and return it. Convenience for pipelines."""
    if event.category is None:
        event.category = categorize(event)
    return event


# --- internals -------------------------------------------------------------


def _categorize_tool(event: AgentEvent) -> str | None:
    name = event.name
    payload = event.payload if isinstance(event.payload, dict) else {}
    input_ = payload.get("input") if isinstance(payload, dict) else None

    # Static name lookup first.
    if event.source == "claude":
        cat = _CLAUDE_TOOL_CATEGORY.get(name)
        if cat:
            return cat
        # Bash → look at the command string.
        if name == "Bash":
            cmd = _extract_command_str(input_)
            return _classify_command(cmd) if cmd else "shell"
    elif event.source == "codex":
        cat = _CODEX_TOOL_CATEGORY.get(name)
        if cat:
            return cat
        if name == "exec_command":
            cmd = _extract_command_str(input_)
            # Codex also gives us parsed_cmd in payload — first parsed cmd is
            # often the most accurate verb.
            parsed = payload.get("parsed_cmd")
            if not cmd and isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict):
                    cmd = first.get("cmd") or ""
            return _classify_command(cmd) if cmd else "shell"
        if name == "write_stdin":
            # Sending input to a running exec — track as shell continuation.
            return "shell"
    # Unknown tool — fall back to ``tool``.
    return "tool"


def _extract_command_str(input_: Any) -> str:
    """Pull the shell command out of a tool input payload."""
    if input_ is None:
        return ""
    if isinstance(input_, str):
        return input_
    if isinstance(input_, dict):
        for k in ("command", "cmd", "script", "shell_command"):
            v = input_.get(k)
            if isinstance(v, str) and v:
                return v
        # Some Codex exec_command shapes nest under ``arguments``.
        a = input_.get("arguments")
        if isinstance(a, dict):
            return _extract_command_str(a)
        if isinstance(a, str):
            return a
    return ""


def _classify_command(cmd: str) -> str:
    cmd = cmd.strip()
    if not cmd:
        return "shell"
    # Take just the first few words; the rest is usually arguments.
    head = " ".join(cmd.split()[:6])
    for cat, pat in _CMD_RULES:
        if pat.search(head):
            return cat
    return "shell"
