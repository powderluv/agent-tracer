"""Redact credentials before they leak into derived artifacts.

Applied by the normalizer to every string flowing through ``_truncate_str``,
so all downstream consumers (Perfetto trace, hints, stats, report) see the
sanitized form. The original session JSONLs in ``~/.claude`` and
``~/.codex`` are never modified — redaction happens in-memory.

Patterns matched (case-sensitive on the verbs):

* ``sshpass -p PASSWORD``
* ``sshpass --password=PASSWORD`` / ``sshpass --password PASSWORD``
* ``SSHPASS=PASSWORD`` env var prefix
* ``--password=PASSWORD`` / ``--password PASSWORD`` for any tool
* ``--token=TOKEN`` / ``-token TOKEN`` for any tool
* ``(GITHUB|GH|API|OPENAI|ANTHROPIC|HF|HUGGINGFACE)_(TOKEN|KEY|SECRET)=…``
* ``printf "%s\\n" PASSWORD | sudo`` — the standard sudo-piping pattern.

We also collect the actual captured password values during a single call
and replace any later occurrence of the same value within the string, so
when a credential appears twice (e.g., once as ``sshpass -p PWD`` and
again inside a remote command) the second copy is also redacted.
"""

from __future__ import annotations

import re

REDACTED = "<REDACTED>"

# (description, regex, replacement). The regex must use group(1) for the
# prefix to keep and group(2) for the value to redact.
_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "sshpass -p",
        re.compile(r"(sshpass\s+-p\s+)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "sshpass --password=",
        re.compile(r"(sshpass\s+--password=)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "sshpass --password VAL",
        re.compile(r"(sshpass\s+--password\s+)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "SSHPASS=",
        re.compile(r"(\bSSHPASS=)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "--password=",
        re.compile(r"(--password=)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "--password VAL",
        # `\b` after password keeps `--password-store` from matching.
        re.compile(r"(--password)(\s+)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1\2{REDACTED}",
    ),
    (
        "--token=",
        re.compile(r"(--token=)('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"),
        rf"\1{REDACTED}",
    ),
    (
        "API_KEY/TOKEN/SECRET env var",
        re.compile(
            # Vendor prefix, optionally followed by more uppercase words
            # (e.g., OPENAI_API_KEY = OPENAI + _API + _KEY).
            r"\b((?:GITHUB|GH|API|OPENAI|ANTHROPIC|HF|HUGGINGFACE|HUGGING_FACE|AWS|GCP|AZURE)"
            r"(?:_[A-Z0-9]+)*_(?:TOKEN|KEY|SECRET|PASSWORD|APIKEY|ACCESS_KEY))="
            r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)"
        ),
        rf"\1={REDACTED}",
    ),
    (
        "printf %s | sudo",
        # printf "%s\n" PASSWORD | sudo -S …
        re.compile(
            r"""(printf\s+["']%s\\?n["']\s+)"""        # group 1: printf "%s\n"
            r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)"""  # group 2: password value
            r"""(\s*\|\s*sudo\s+-S)"""                  # group 3: | sudo -S
        ),
        rf"\1{REDACTED}\3",
    ),
]


def _strip_quotes(s: str) -> str:
    """Return ``s`` without one matched layer of surrounding quotes, if present."""
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def redact_secrets(text: str) -> str:
    """Apply all known credential patterns to ``text``.

    Also performs a second pass: any value captured from a primary pattern
    (e.g., the password from ``sshpass -p X``) is also replaced wherever
    else it appears in ``text``, so paired occurrences (sudo piping using
    the same password) are caught too. Values shorter than 4 characters
    are skipped from the second pass to avoid clobbering the rest of the
    string.
    """
    if not text or not isinstance(text, str):
        return text

    out = text
    captured: set[str] = set()
    for _name, pat, repl in _PATTERNS:
        # Sub once; while we're at it, harvest the captured group(2) values
        # so we can globally redact them in the second pass.
        for m in pat.finditer(out):
            if m.lastindex and m.lastindex >= 2:
                val = _strip_quotes(m.group(2))
                if len(val) >= 4 and val != REDACTED:
                    captured.add(val)
        out = pat.sub(repl, out)

    if captured:
        # Sort by length desc so longer values get replaced before shorter
        # ones (avoids partially overwriting a longer secret).
        for val in sorted(captured, key=len, reverse=True):
            out = out.replace(val, REDACTED)

    return out
