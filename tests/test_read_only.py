"""Enforce that no agent-tracer code mutates the upstream log directories.

We make ``~/.claude/projects`` and ``~/.codex/sessions`` (or test mirrors of
them) immutable in the test environment by checking that the source files
forbid write-mode access.

The check is intentionally text-based: we don't want any module under
``agent_tracer.parsers`` to ever open the upstream JSONLs with a writeable
mode (``"w"``, ``"a"``, ``"x"``, ``"r+"``, ``"w+"``, ``"a+"``, ``"x+"``) or
call any of: ``Path.write_text``, ``Path.write_bytes``, ``os.remove``,
``os.unlink``, ``shutil.copy*``, ``shutil.move``, ``Path.unlink``,
``Path.rename``, ``Path.chmod``. The CLI is allowed to write to
``~/.cache/agent-tracer/`` — but that's not under the parsers package.

A second runtime check actually opens a temporary tree mimicking the layout,
takes a snapshot of every file's mtime/size, runs every parser entry point,
and asserts nothing changed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

PARSERS_DIR = Path(__file__).resolve().parent.parent / "src" / "agent_tracer" / "parsers"

# Only flag patterns that are unambiguously write-capable. ``str.replace``
# is not forbidden; ``Path.replace`` (which renames) is rare and reviewed
# manually if it appears.
FORBIDDEN_WRITE_OPEN = re.compile(
    r"""
    \.open\(\s*['"][wax]\+?b?['"]\s*[,)] |        # path.open("w"/"a"/"x"/...)
    open\(\s*[^,]+,\s*['"][wax]\+?b?['"]\s*[,)] | # builtin open(p, "w"/...)
    \.write_text\(   |
    \.write_bytes\(  |
    \.unlink\(       |
    \.rename\(       |
    \.chmod\(        |
    os\.remove\(     |
    os\.unlink\(     |
    os\.rename\(     |
    os\.replace\(    |
    shutil\.copy     |
    shutil\.move\(
    """,
    re.VERBOSE,
)


def test_parsers_have_no_write_calls() -> None:
    """Static check: parser sources never call any write-capable filesystem API."""
    offenders: list[tuple[str, int, str]] = []
    for py in PARSERS_DIR.rglob("*.py"):
        for i, line in enumerate(py.read_text().splitlines(), 1):
            stripped = line.split("#", 1)[0]
            if FORBIDDEN_WRITE_OPEN.search(stripped):
                offenders.append((py.name, i, line.strip()))
    assert not offenders, (
        "Parser modules must be read-only. Offending lines:\n"
        + "\n".join(f"  {name}:{lineno}: {src}" for name, lineno, src in offenders)
    )


def _snapshot_tree(root: Path) -> dict[Path, tuple[int, int, bytes]]:
    """Return {path: (size, mtime_ns, sha-of-content)} for every file under root."""
    import hashlib

    out: dict[Path, tuple[int, int, bytes]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            data = p.read_bytes()
            st = p.stat()
            out[p] = (st.st_size, st.st_mtime_ns, hashlib.sha256(data).digest())
    return out


@pytest.fixture()
def fake_logs(tmp_path: Path) -> tuple[Path, Path]:
    """Build a tiny mirror of the Claude + Codex log layouts."""
    claude_root = tmp_path / "claude"
    codex_root = tmp_path / "codex"

    proj = claude_root / "-home-x"
    # Main session file: <project>/<sessionId>.jsonl
    (proj).mkdir(parents=True)
    (proj / "sess1.jsonl").write_text(
        json.dumps({"type": "user", "uuid": "u1", "timestamp": "t", "sessionId": "sess1"}) + "\n"
    )
    # Subagent file
    (proj / "sess1" / "subagents").mkdir(parents=True)
    (proj / "sess1" / "subagents" / "agent-aabbccdd.jsonl").write_text(
        json.dumps({"type": "user", "uuid": "u2", "timestamp": "t", "agentId": "aabbccdd"}) + "\n"
    )

    codex_day = codex_root / "2026/04/27"
    codex_day.mkdir(parents=True)
    (codex_day / "rollout-2026-04-27T00-28-25-019dcdd6-aaaa-bbbb-cccc-ddddeeeeffff.jsonl").write_text(
        json.dumps({"timestamp": "t", "type": "session_meta", "payload": {}}) + "\n"
    )
    return claude_root, codex_root


def test_parsers_do_not_mutate_log_tree(fake_logs: tuple[Path, Path]) -> None:
    """Runtime check: snapshot, run every parser entry point, snapshot again, diff."""
    from agent_tracer.parsers import claude as claude_mod
    from agent_tracer.parsers import codex as codex_mod
    from agent_tracer.parsers.discover import scan_claude, scan_codex

    claude_root, codex_root = fake_logs
    before = _snapshot_tree(claude_root) | _snapshot_tree(codex_root)

    orig_c, orig_x = claude_mod.CLAUDE_PROJECTS_DIR, codex_mod.CODEX_SESSIONS_DIR
    try:
        claude_mod.CLAUDE_PROJECTS_DIR = claude_root
        codex_mod.CODEX_SESSIONS_DIR = codex_root
        # Exercise every read path we ship.
        for sf in claude_mod.iter_session_files():
            list(claude_mod.iter_raw_records(sf.path))
        for sf in codex_mod.iter_session_files():
            list(codex_mod.iter_raw_records(sf.path))
        scan_claude()
        scan_codex()
    finally:
        claude_mod.CLAUDE_PROJECTS_DIR = orig_c
        codex_mod.CODEX_SESSIONS_DIR = orig_x

    after = _snapshot_tree(claude_root) | _snapshot_tree(codex_root)
    assert before == after, "Parser run modified the log tree"


def test_default_log_dirs_are_read_only_in_repo_layout() -> None:
    """The two module-level constants must point at the user's home, not the repo."""
    from agent_tracer.parsers.claude import CLAUDE_PROJECTS_DIR
    from agent_tracer.parsers.codex import CODEX_SESSIONS_DIR

    home = Path(os.path.expanduser("~")).resolve()
    assert CLAUDE_PROJECTS_DIR.resolve().is_relative_to(home)
    assert CODEX_SESSIONS_DIR.resolve().is_relative_to(home)
