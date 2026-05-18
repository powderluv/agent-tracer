# agent-tracer

Analyze Claude Code and Codex session logs on this machine. Emit unified
Perfetto traces, agent + system telemetry, and ranked optimization hints
for both the agent workflow itself and the CPU/GPU work it drives.

See [PLAN.md](PLAN.md) for the full design.

## Status

P0 — repo scaffold, raw JSONL parsers, schema discovery.

## Quick start

```
pip install -e .

# Sanity-check that we can read your local Claude/Codex logs
agent-tracer discover

# Stream raw events from both sources, newest first
agent-tracer list --source claude --limit 20
agent-tracer list --source codex --limit 20
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
│   ├── claude.py        # ~/.claude JSONL → raw records
│   ├── codex.py         # ~/.codex JSONL → raw records
│   └── discover.py      # schema/shape sanity report
├── cli.py               # argparse entry point
├── hints/               # detector modules (P5/P6)
└── telemetry/           # sampler daemon (P4)
```
