"""Credential redaction patterns and second-pass value sweep."""

from __future__ import annotations

from agent_tracer.redact import REDACTED, redact_secrets


def test_sshpass_dash_p_double_quoted() -> None:
    s = 'sshpass -p "secretvalue" ssh nod@host'
    out = redact_secrets(s)
    assert "secretvalue" not in out
    assert REDACTED in out
    assert "sshpass -p" in out and "ssh nod@host" in out


def test_sshpass_dash_p_single_quoted() -> None:
    s = "sshpass -p '$harktank2Go' ssh nod@192.168.122.16 'cmd'"
    out = redact_secrets(s)
    assert "$harktank2Go" not in out
    assert REDACTED in out


def test_sshpass_dash_p_unquoted() -> None:
    s = "sshpass -p gpu-test ssh ubuntu@192.168.122.170 ls"
    out = redact_secrets(s)
    assert "gpu-test" not in out
    assert REDACTED in out


def test_sshpass_long_form_password_flag() -> None:
    assert "verysecret" not in redact_secrets("sshpass --password=verysecret ssh x")
    assert "verysecret" not in redact_secrets("sshpass --password verysecret ssh x")


def test_env_prefix_SSHPASS() -> None:
    s = "SSHPASS='$harktank2Go' sshpass ssh nod@host"
    out = redact_secrets(s)
    assert "$harktank2Go" not in out
    assert "SSHPASS=" in out and REDACTED in out


def test_generic_password_flag() -> None:
    s = "mytool --password=hunter2 --other=ok"
    assert "hunter2" not in redact_secrets(s)
    s2 = "mytool --password hunter2 --other ok"
    assert "hunter2" not in redact_secrets(s2)


def test_password_store_not_redacted() -> None:
    s = "git --password-store=plaintext clone url"
    assert redact_secrets(s) == s


def test_token_flag_redacted() -> None:
    s = "gh auth --token=ghp_VeryLongTokenABCDEFGHIJ status"
    assert "ghp_VeryLongTokenABCDEFGHIJ" not in redact_secrets(s)


def test_api_key_envvar() -> None:
    for name in (
        "GITHUB_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "AWS_SECRET",
    ):
        s = f"{name}=verylongvalue1234 ./run.sh"
        out = redact_secrets(s)
        assert "verylongvalue1234" not in out, f"failed for {name}: {out}"
        assert name in out


def test_printf_sudo_pipe() -> None:
    s = (
        'ssh ubuntu@host \'printf "%s\\n" supersecretpw | sudo -S '
        "apt install foo'"
    )
    out = redact_secrets(s)
    assert "supersecretpw" not in out


def test_value_sweep_catches_password_repeated_in_string() -> None:
    # password appears in sshpass AND inside the remote command.
    s = (
        "sshpass -p gpu-test ssh ubuntu@h "
        "'printf \"%s\\n\" gpu-test | sudo -S ls'"
    )
    out = redact_secrets(s)
    assert "gpu-test" not in out


def test_short_values_are_not_globally_swept() -> None:
    # Three-char password is sketchy to globally sweep — we keep the
    # primary redaction but skip the second pass.
    s = "sshpass -p abc ssh user@host abcdef"
    out = redact_secrets(s)
    assert "sshpass -p <REDACTED>" in out
    # `abcdef` later in the string should remain (it isn't a credential
    # we've identified; just a coincidental substring).
    assert "abcdef" in out


def test_non_string_inputs_pass_through() -> None:
    assert redact_secrets("") == ""  # type: ignore[arg-type]
    assert redact_secrets(None) is None  # type: ignore[arg-type]


def test_normalizer_pipes_redaction_through(monkeypatch) -> None:
    """End-to-end: a Bash tool_use with sshpass arrives redacted in the AgentEvent."""
    from agent_tracer.events import EventKind
    from agent_tracer.normalize import normalize_claude_session

    records = [
        {
            "type": "assistant",
            "uuid": "a1",
            "timestamp": "2026-04-01T00:00:00.000Z",
            "sessionId": "s",
            "message": {
                "model": "x",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu1",
                        "name": "Bash",
                        "input": {
                            "command": "sshpass -p '$harktank2Go' ssh nod@host pwd"
                        },
                    }
                ],
            },
        },
        {
            "type": "user",
            "uuid": "u1",
            "parentUuid": "a1",
            "timestamp": "2026-04-01T00:00:01.000Z",
            "sessionId": "s",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}
                ],
            },
        },
    ]
    tool = next(
        e
        for e in normalize_claude_session(records)
        if e.kind == EventKind.TOOL_CALL
    )
    assert "$harktank2Go" not in str(tool.payload)
    assert REDACTED in tool.payload["input"]["command"]
