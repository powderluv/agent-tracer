# agent-tracer

Analyze Claude Code and Codex session logs on this machine. Emit unified
Perfetto traces, agent + system telemetry, and ranked optimization hints
for both the agent workflow itself and the CPU/GPU work it drives.

See [PLAN.md](PLAN.md) for the full design.

## Status

P5 — unified Claude+Codex Perfetto trace + categorized events + per-session
stats + agent-side optimization hints. P4 (telemetry sampler) and P6
(telemetry-driven GPU/build hints) are the remaining phases.

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
```

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
