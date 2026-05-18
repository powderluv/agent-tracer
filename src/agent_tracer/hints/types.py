"""Shared types for hint detectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True, frozen=True)
class Anchor:
    """A concrete pointer into the source data so a hint is verifiable."""

    source: str          # "claude" | "codex"
    session_id: str
    ts_us: int
    detail: str          # one-line evidence: command snippet, file path, etc.


@dataclass(slots=True)
class Hint:
    detector: str        # stable id, e.g. "redundant_reads"
    category: str        # "agent" (P5) or "gpu" / "build" / ... (P6)
    title: str
    severity: Severity
    occurrences: int
    anchors: list[Anchor] = field(default_factory=list)
    remediation: str = ""
    # Optional cost estimates — None when we genuinely don't know.
    est_wall_saved_s: float | None = None
    est_tokens_saved: int | None = None
    # Free-form structured evidence dict for downstream tools / JSON output.
    evidence: dict = field(default_factory=dict)
