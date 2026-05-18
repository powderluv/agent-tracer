"""Timestamp helpers. Perfetto wants epoch microseconds."""

from __future__ import annotations

from datetime import UTC, datetime


def iso_to_us(ts: str) -> int:
    """Convert an ISO-8601 UTC string to epoch microseconds.

    Tolerates the trailing ``Z`` form and missing microseconds. Assumes UTC if
    no offset is present (Claude and Codex both emit ``Z``).
    """
    if not ts:
        raise ValueError("empty timestamp")
    # Python 3.11+ accepts the literal ``Z`` suffix in fromisoformat.
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000)
