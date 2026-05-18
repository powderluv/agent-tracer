"""Normalized event schema shared by all parsers and downstream consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class EventKind(StrEnum):
    USER_TURN = "user_turn"
    ASSISTANT_MSG = "assistant_msg"
    ASSISTANT_TEXT = "assistant_text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    SUBAGENT_SPAWN = "subagent_spawn"
    PROGRESS = "progress"
    ERROR = "error"
    COMPACTION = "compaction"
    SESSION_META = "session_meta"


Source = Literal["claude", "codex"]


@dataclass(slots=True)
class AgentEvent:
    """One normalized event. Tool calls are complete events (ts_end_us set);
    everything else is an instant (ts_end_us is None)."""

    source: Source
    session_id: str
    kind: EventKind
    name: str
    ts_start_us: int
    ts_end_us: int | None = None

    agent_id: str | None = None
    parent_uuid: str | None = None
    uuid: str | None = None

    category: str | None = None  # filled by content miner
    cwd: str | None = None
    git_branch: str | None = None
    model: str | None = None

    tokens_input: int | None = None
    tokens_output: int | None = None
    cache_read: int | None = None
    cache_create: int | None = None

    # tool-call extras
    tool_use_id: str | None = None
    subagent_type: str | None = None
    exit_code: int | None = None
    is_error: bool | None = None

    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_us(self) -> int | None:
        if self.ts_end_us is None:
            return None
        return self.ts_end_us - self.ts_start_us
