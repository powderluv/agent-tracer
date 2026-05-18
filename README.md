# agent-tracer

Analyze Claude Code and Codex session logs on this machine. Emit unified
Perfetto traces, agent + system telemetry, and ranked optimization hints
for both the agent workflow itself and the CPU/GPU work it drives.

See [PLAN.md](PLAN.md) for the full design.

## Status

P1 — Claude normalizer + Perfetto trace emitter. Codex parser/normalizer
lands in P2.

## Read-only access

The tool *never* writes inside `~/.claude/projects` or `~/.codex/sessions`.
Parser modules open files with `"rb"` and a static+runtime test enforces
that no write-capable API ever enters those packages.

## Quick start

```
pip install -e .

# Sanity-check that we can read your local Claude/Codex logs
agent-tracer discover

# Stream raw events from a source
agent-tracer list --source claude --limit 20
agent-tracer list --source codex --limit 20

# Build a Perfetto trace from the last few weeks of Claude sessions
agent-tracer build --since 2026-05-01 -o trace.json
# Open trace.json in https://ui.perfetto.dev
```

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
