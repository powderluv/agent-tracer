# agent-tracer

Analyze Claude Code and Codex session logs on this machine. Emit unified
Perfetto traces, agent + system telemetry, and ranked optimization hints
for both the agent workflow itself and the CPU/GPU work it drives.

See [PLAN.md](PLAN.md) for the full design.

## Status

All phases (P0–P6) shipped. The tool ingests Claude + Codex session logs,
unifies them into a Perfetto trace, runs agent-side / build-pattern /
telemetry-correlated optimization detectors, and can embed each hint as
an instant in the trace itself.

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

## Detectors

### Agent-side (no telemetry needed)
- **redundant_reads** — same file Read ≥3× in one session.
- **repeated_bash** — identical Bash/exec_command ≥3× in one session
  (filters trivial pwd/ls/cd).
- **compaction_frequency** — context-compaction firing ≥3× per session.
- **hot_tool_time** — one tool kind dominating ≥50% of session wall-clock.

### Build pattern (no telemetry needed)
- **repeated_rebuilds** — same `ninja <target>` ≥4× in one calendar day.
- **expunge_chain** — `<target>+expunge && <target>+dist` patterns.
- **ssh_overhead** — sum of ssh/scp/sshpass/rsync wall-time over a session.

### Telemetry-correlated (needs `agent-tracer sample` running)
- **gpu_idle_build** — `cat:build` span >30s with mean GPU util <5%
  (build is CPU-bound).
- **host_bound_gpu** — GPU util ≥70% AND host CPU ≥90% (host-side
  bottleneck).
- **vram_pressure** — VRAM used ≥90% of total during a span.

Each hint carries concrete anchors (session id, timestamp, command snippet)
and a remediation string. Min-evidence thresholds suppress noise.

### Embedding hints in the trace

```
agent-tracer build --since 2026-05-01 --annotate-hints -o trace.json
```

This runs every detector and adds each hint anchor as an instant event
in the trace (cat=`opt-hint`) at the anchor's timestamp, so the hints
appear inline next to the spans they refer to.

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
