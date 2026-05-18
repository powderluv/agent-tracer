# agent-tracer

Analyze Claude Code and Codex session logs on this machine. Emit unified
Perfetto traces, agent + system telemetry, and ranked optimization hints
for both the agent workflow itself and the CPU/GPU work it drives.

See [PLAN.md](PLAN.md) for the full design.

## Status

P4 — telemetry sampler daemon shipped. P6 (telemetry-driven GPU/build
hints) is the remaining phase.

## Read-only access

The tool *never* writes inside `~/.claude/projects` or `~/.codex/sessions`.
Parser modules open files with `"rb"` and a static+runtime test enforces
that no write-capable API ever enters those packages.

## Quick start

```
pip install -e .

# Sanity-check that we can read your local Claude/Codex logs
agent-tracer discover

# Build a unified Perfetto trace (Claude + Codex) for the last few weeks
agent-tracer build --since 2026-05-01 -o trace.json
# Open trace.json in https://ui.perfetto.dev

# Per-session tables: wall-clock, tools, tokens, cache hit rate, top commands
agent-tracer stats --since 2026-05-01

# Ranked optimization hints (markdown or --json)
agent-tracer hints --since 2026-05-01

# Restrict to one source / project / set of sessions
agent-tracer hints --since 2026-05-01 --source codex
agent-tracer build  --project-slug=-home-nod-github-claude-rocm-workspace -o trace.json

# Telemetry sampler (1Hz to LanceDB; needs [store] extras)
pip install -e '.[store]'
agent-tracer sample --interval 1
# Or one-shot to verify it works
agent-tracer sample --once
```

## Telemetry sampler

`agent-tracer sample` polls `rocm-smi`, `nvidia-smi`, and `/proc` and writes
`gpu_telemetry` + `system_telemetry` tables to
`~/.cache/agent-tracer/telemetry.lance`. Missing/erroring tools are silently
skipped; the daemon still records what's available.

Binary search paths (env override > venv `bin/` > `/opt/rocm/bin` >
`/opt/rocm-*/bin` > `$PATH`):

```
AGENT_TRACER_ROCM_SMI=/opt/rocm-6.4/bin/rocm-smi agent-tracer sample
AGENT_TRACER_NVIDIA_SMI=/usr/bin/nvidia-smi      agent-tracer sample
```

Writes are batched (≥256 rows or 60s) to avoid fragmenting the Lance
dataset. SIGINT/SIGTERM flushes cleanly.

## Detectors shipped (P5, agent-side only)

- **redundant_reads** — same file Read ≥3× in one session.
- **repeated_bash** — identical Bash/exec_command ≥3× in one session
  (filters trivial pwd/ls/cd).
- **compaction_frequency** — context-compaction firing ≥3× per session.
- **hot_tool_time** — one tool kind dominating ≥50% of session wall-clock.

Each hint carries concrete anchors (session id, timestamp, command snippet)
and a remediation string. Min-evidence thresholds suppress noise.

## Data sources

- `~/.claude/projects/<cwd-slug>/<sessionId>/*.jsonl` (main)
- `~/.claude/projects/<cwd-slug>/<sessionId>/subagents/agent-*.jsonl`
- `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`

## Layout

```
src/agent_tracer/
├── events.py            # normalized AgentEvent dataclass
├── parsers/
│   ├── claude.py        # ~/.claude JSONL → raw records (read-only)
│   ├── codex.py         # ~/.codex JSONL → raw records (read-only)
│   └── discover.py      # schema/shape sanity report
├── normalize.py         # raw records → AgentEvent stream
├── perfetto.py          # AgentEvent stream → Chrome/Perfetto trace JSON
├── timeutil.py          # ISO-8601 → epoch microseconds
├── cli.py               # argparse entry point
├── hints/               # detector modules (P5/P6)
└── telemetry/           # sampler daemon (P4)
```
