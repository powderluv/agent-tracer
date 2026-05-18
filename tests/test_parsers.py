"""Smoke tests using synthetic fixtures that mirror the discovered shapes."""

from __future__ import annotations

import json
from pathlib import Path

from agent_tracer.parsers import claude, codex
from agent_tracer.parsers.discover import scan_claude, scan_codex


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_claude_iter_session_files(tmp_path: Path, monkeypatch) -> None:
    # Two sessions, one with a normal subagent and one compaction subagent.
    proj = tmp_path / "-home-user-foo"
    s1 = proj / "session-aaa"
    s2 = proj / "session-bbb"
    _write_jsonl(s1 / "session-aaa.jsonl", [{"type": "user", "uuid": "u1"}])
    _write_jsonl(
        s1 / "subagents" / "agent-a1234abcd.jsonl",
        [{"type": "user", "uuid": "u2", "agentId": "a1234abcd"}],
    )
    _write_jsonl(
        s2 / "subagents" / "agent-acompact-deadbeef.jsonl",
        [{"type": "system", "uuid": "u3"}],
    )

    files = list(claude.iter_session_files(root=tmp_path))
    kinds = {f.kind for f in files}
    assert kinds == {"main", "subagent", "subagent_compaction"}
    sub = next(f for f in files if f.kind == "subagent")
    assert sub.agent_id == "a1234abcd"
    comp = next(f for f in files if f.kind == "subagent_compaction")
    assert comp.is_compaction


def test_claude_iter_raw_records_resumes_from_offset(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    _write_jsonl(p, [{"i": 0}, {"i": 1}, {"i": 2}])

    offsets = []
    records = []
    for off, rec in claude.iter_raw_records(p):
        offsets.append(off)
        records.append(rec)
    assert [r["i"] for r in records] == [0, 1, 2]

    # Resuming from the offset after record 1 should yield only record 2.
    resumed = [r for _, r in claude.iter_raw_records(p, start_offset=offsets[1])]
    assert [r["i"] for r in resumed] == [2]


def test_claude_iter_raw_records_skips_bad_lines(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text('{"ok": 1}\nnot json\n{"ok": 2}\n')
    out = [r for _, r in claude.iter_raw_records(p)]
    assert out == [{"ok": 1}, {"ok": 2}]


def test_codex_iter_session_files_orders_by_start_time(tmp_path: Path) -> None:
    early = tmp_path / "2026/03/08"
    late = tmp_path / "2026/04/27"
    _write_jsonl(
        early / "rollout-2026-03-08T00-30-27-019ccc91-b87c-7321-bee1-c0262a5d648a.jsonl",
        [{"timestamp": "2026-03-08T00:30:27Z", "type": "session_meta", "payload": {}}],
    )
    _write_jsonl(
        late / "rollout-2026-04-27T00-28-25-019dcdd6-e273-7223-84a4-cd3000821fe2.jsonl",
        [{"timestamp": "2026-04-27T00:28:25Z", "type": "session_meta", "payload": {}}],
    )
    files = list(codex.iter_session_files(root=tmp_path))
    assert [f.session_id for f in files] == [
        "019ccc91-b87c-7321-bee1-c0262a5d648a",
        "019dcdd6-e273-7223-84a4-cd3000821fe2",
    ]


def test_discover_handles_realistic_records(tmp_path: Path) -> None:
    # Mirror the actual record shapes we saw in ~/.claude and ~/.codex.
    claude_root = tmp_path / "claude" / "-home-x" / "sess1"
    _write_jsonl(
        claude_root / "sess1.jsonl",
        [
            {
                "type": "assistant",
                "uuid": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {}},
                    ],
                },
            },
            {
                "type": "user",
                "uuid": "u1",
                "parentUuid": "a1",
                "message": {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
                },
            },
            {"type": "progress", "uuid": "p1", "data": {"type": "query_update"}},
        ],
    )
    codex_root = tmp_path / "codex" / "2026" / "04" / "27"
    _write_jsonl(
        codex_root / "rollout-2026-04-27T00-28-25-019dcdd6-aaaa-bbbb-cccc-ddddeeeeffff.jsonl",
        [
            {"timestamp": "t", "type": "session_meta", "payload": {}},
            {
                "timestamp": "t",
                "type": "response_item",
                "payload": {"type": "function_call", "call_id": "c1", "name": "exec_command"},
            },
            {
                "timestamp": "t",
                "type": "event_msg",
                "payload": {"type": "exec_command_end", "call_id": "c1"},
            },
        ],
    )

    # Use the public scanner against our synthetic roots by monkeypatching the
    # module-level constants.
    from agent_tracer.parsers import claude as claude_mod
    from agent_tracer.parsers import codex as codex_mod

    orig_claude = claude_mod.CLAUDE_PROJECTS_DIR
    orig_codex = codex_mod.CODEX_SESSIONS_DIR
    try:
        claude_mod.CLAUDE_PROJECTS_DIR = tmp_path / "claude"
        codex_mod.CODEX_SESSIONS_DIR = tmp_path / "codex"
        cr = scan_claude()
        xr = scan_codex()
    finally:
        claude_mod.CLAUDE_PROJECTS_DIR = orig_claude
        codex_mod.CODEX_SESSIONS_DIR = orig_codex

    assert cr.files == 1 and cr.records == 3
    assert cr.types["assistant"] == 1
    assert cr.block_types[("assistant", "tool_use")] == 1
    assert cr.block_types[("user", "tool_result")] == 1

    assert xr.files == 1 and xr.records == 3
    assert xr.payload_types[("response_item", "function_call")] == 1
    assert xr.payload_types[("event_msg", "exec_command_end")] == 1
