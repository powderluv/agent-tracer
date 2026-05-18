# agent-tracer — Design Plan

Analyze Claude Code and Codex session logs to produce unified Perfetto traces,
agent/system telemetry, and concrete optimization hints.

## Goals

1. **Trace** — emit a Chrome/Perfetto trace JSON of agent activity (tool calls,
   subagent fan-out, token usage) across both Claude and Codex sessions, viewable
   in `ui.perfetto.dev`.
2. **Correlate** — overlay system CPU/GPU telemetry (rocm-smi / nvidia-smi /
   procfs) sampled going forward.
3. **Mine** — tag spans by content (gpu / build / test / git / network / fs) so
   the trace is filterable.
4. **Diagnose** — run detectors over the normalized event store and surface
   ranked optimization hints with concrete anchors (session + timestamp +
   evidence). Hints cover both *agent workflow* (tokens, redundant work,
   serialization) and *underlying CPU/GPU work* (idle builds, host-bound
   kernels, VRAM pressure, repeated rebuilds, FLR/brick events).

## Non-goals

- Not a UI. Perfetto is the viewer; we emit JSON.
- Not a code-quality judge of the agent. We measure observable cost; we don't
  claim a Read was "unnecessary" without evidence of duplication.
- Not auto-fixing anything. Hints are suggestions with anchors.
- Not a replacement for `claude-trace` or proprietary observability. This is a
  local-first analysis tool over on-disk session logs.

## Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Claude JSONL │   │ Codex JSONL  │   │ telemetry sampler│
│  ~/.claude/  │   │  ~/.codex/   │   │ (rocm-smi/etc)   │
└──────┬───────┘   └──────┬───────┘   └────────┬─────────┘
       │                  │                    │
       ▼                  ▼                    ▼
       parsers ──► normalizer ──► events.lance     telemetry.lance
                                       │                    │
                                       ▼                    │
                                content miner               │
                                       │                    │
              ┌────────────────────────┴────────────────────┘
              ▼                                  ▼
       Perfetto trace builder ──► trace.json   detectors ──► hints.md
```

## Storage

```
~/.cache/agent-tracer/
├── manifest.json          # (file_path, mtime, last_byte_offset) — point lookups
├── events.lance/          # normalized AgentEvent rows
├── telemetry.lance/       # 1Hz sampler output
└── hints.lance/           # hint history (track whether suggestions were acted on)
```

- **LanceDB** for the two analytical tables (columnar, fast scans, append-
  friendly, optional vector search later).
- **JSON manifest** for ingest bookkeeping (wrong shape for Lance).
- **Don't touch `~/.codex/logs_2.sqlite`** — owned by the live Codex TUI (WAL
  contention, unstable schema). We parse rollout JSONLs.
- Batch writes (≥1000 rows or 60s) and periodic `compact_files()` to keep Lance
  fragments healthy.

## Normalized event schema

```python
@dataclass
class AgentEvent:
    source: Literal["claude", "codex"]
    session_id: str
    agent_id: str | None            # Claude subagent id; None for Codex top-level
    parent_uuid: str | None         # Claude only — chains tool_result to tool_use
    kind: EventKind                 # see below
    name: str                       # tool name, "user_turn", "assistant_text", …
    category: str | None            # filled by content miner: gpu/build/test/...
    ts_start_us: int                # epoch microseconds
    ts_end_us: int | None           # complete events only
    cwd: str | None
    git_branch: str | None
    model: str | None
    tokens_input: int | None
    tokens_output: int | None
    cache_read: int | None
    cache_create: int | None
    payload: dict                   # full original record (truncated for huge stdout)
```

`EventKind`: `user_turn`, `assistant_msg`, `assistant_text`, `thinking`,
`tool_call`, `subagent_spawn`, `progress`, `error`, `compaction`.

Tool calls become **complete** events (`ts_start_us`, `ts_end_us`) by pairing
the call with its result. Everything else is an **instant** event.

### Pairing

- **Claude**: `assistant.tool_use.id` → `user.tool_result.tool_use_id` (and
  `tool_result.parentUuid == tool_use.uuid` as a sanity check). Start =
  `tool_use` record timestamp, end = `tool_result` record timestamp. Sub-tool
  progress events (`type:"progress"`) attach to the enclosing tool span via
  `parentToolUseID`.
- **Codex**: `response_item.function_call.call_id` → `event_msg.exec_command_end.call_id`
  (or `response_item.function_call_output.call_id` for non-exec tools). Codex
  conveniently provides `process_id` and `parsed_cmd` in `exec_command_end`.

### Subagents

- **Claude** writes one JSONL per subagent at
  `~/.claude/projects/<cwd>/<sessionId>/subagents/agent-<agentId>.jsonl` with a
  matching `.meta.json`. Glue to parent session by `sessionId`; subagent's first
  `user` record's parent tool_use lives in the main session log.
  `agent-acompact-*` are compaction agents — tagged separately so they don't
  dominate the lane view.
- **Codex** is single-agent (no subagents); fan-out happens via parallel
  function calls within one turn.

## Perfetto mapping

- **pid per session**: `claude:<sessionId>` / `codex:<sessionId>`, named via
  metadata events.
- **tid lanes within a session**: main turn lane, one per Claude subagent
  (named by `slug`), one telemetry lane.
- **`ph:'X'` complete events**: tool calls with `dur`.
- **`ph:'i'` instants**: user prompts, thinking blocks, errors, hints.
- **`ph:'C'` counters**: per-session token counters (`input`, `output`,
  `cache_read`, `cache_create`). Global lane: `gpu_util%`, `vram_used`,
  `gpu_power_w`, `cpu_util%`.
- **`cat`** carries the content-miner tag.
- **`args`** carries cwd, model, subagent_type, truncated command snippet,
  tool_use_id.

## Content miner

Regex/keyword classifier over Bash/shell `command` strings:

| Pattern | Category |
|---|---|
| `rocm-smi`, `nvidia-smi`, `hipInfo`, `amd-smi`, `rocminfo` | `gpu,query` |
| `hipcc`, `nvcc`, `cmake`, `ninja`, `make\b` | `build` |
| `pytest`, `ctest`, `hipTest`, `gtest`, `test_*` | `test` |
| `git\b`, `gh\s` | `git` |
| `ssh`, `sshpass`, `scp`, `rsync` | `network` |
| Read/Edit/Write/Glob/Grep tools | `fs` |

Assistant-text keyword pass for high-signal GPU terms (`MES`, `PSP`,
`IC_BASE`, `gfx12`, `KFD`, `SDMA`) tags the enclosing turn as GPU-related
even when no command ran.

## Optimization detectors

Pure functions over `events.lance` (+ `telemetry.lance` for the second group).
Each returns `Iterable[Hint]` with **concrete anchors** (session + timestamp +
command). Min-evidence thresholds suppress noise.

### Agent-workflow

| Detector | Signal | Suggests |
|---|---|---|
| Serial tool calls | Back-to-back single-tool turns with no data dep | Batch into one turn |
| Redundant Reads | Same `file_path` Read ≥3× in one session | Earlier context retention |
| Repeated Bash output | Identical Bash ≥3× in one session | Cache result |
| Cache miss churn | `cache_read / (cache_read+cache_create)` drops | Avoid large rewrites mid-session |
| Subagent serialization | Adjacent independent Agent dispatches | One message with parallel Agent calls |
| Over-thinking | Long `thinking` then tiny Edit | Lower thinking budget for this class |
| Compaction frequency | `agent-acompact-*` fires >N×/session | Split into separate sessions |
| Hot tool-call types | Top-N tool kinds by wall-clock | Surfaces where time goes |

### Underlying CPU/GPU

| Detector | Signal | Suggests |
|---|---|---|
| GPU-idle build | `cat:build` span >30s, mean `gpu_util<5%` | More `-j`, ccache, distcc |
| Host-bound GPU run | `gpu_util>70%` AND single CPU core pegged | Threading on host side |
| VRAM pressure | `vram_used>90%` during a span | OOM risk, batch-size tuning |
| Repeated full rebuilds | Same `ninja <target>` ≥N×/day, often post-`expunge` | Incremental build, ccache |
| Expunge chain | `ninja X+expunge && ninja X+dist` pattern | Lift to a script with conditional expunge |
| SSH/SCP overhead | Sum of `sshpass ssh <host>` wall-time/session | `ControlMaster`, persistent tmux |
| Test prep gap | Wall time between consecutive test runs minus runtime | Persistent build/test dir |
| GPU brick events | Bash spans matching `power cycle`, `lspci.*7F`, vfio-pci bind failures | Quantify lost hours |
| Power throttle | `gpu_power≈TDP` with `gpu_util` headroom or temp at throttle | Cooling/power-limit |
| Cold-cache compile | `hipcc`/`nvcc` without prior matching args within window | ccache config |

Severity = measured cost (wall-time or tokens). No subjective severities.

Hints are emitted **both** as `ph:'i'` instants in the trace (category
`opt-hint`, distinct color) and as a markdown report. Hint history persists in
`hints.lance` so we can tell when a suggestion stops firing.

## CLI

```
agent-tracer index                           # rebuild normalized event cache
agent-tracer list [--source claude|codex] [--since DATE]
agent-tracer discover                        # dump schema stats (P0 sanity)
agent-tracer build --since 2026-05-01 [--session ID...] -o trace.json
agent-tracer stats --since 2026-05-01        # Read:Edit, cost, GPU-touching turns
agent-tracer hints --since 2026-05-01 [--category gpu|agent|build]
agent-tracer sample --interval 1s            # telemetry daemon
```

## Phasing

1. **P0** — repo scaffold, plan, raw parsers, schema discovery, fixtures.
2. **P1** — Claude normalizer → Perfetto trace. Validate visually in Perfetto.
3. **P2** — Codex parser, unify into one trace.
4. **P3** — Content miner + category tagging.
5. **P5** — Stats CLI + agent-side hints (no telemetry needed).
6. **P4** — Telemetry sampler daemon.
7. **P6** — GPU/build-side hints + trace annotations.

P5 ships before P4 because agent-side detectors don't need telemetry and give
immediate value.

## Risks / gotchas

- Claude assistant messages carry one `timestamp`, so parallel tool calls in
  one turn collapse to the same start; render them stacked, not as zero-width
  spikes. End = next user message ts.
- Codex rollouts contain very long `aggregated_output` — truncate `args` to
  ~2KB before encoding or the trace file balloons.
- Thinking is largely redacted post-March; use `signature` length as a depth
  proxy (per the Anthropic issue #42796 analysis).
- `agent-acompact-*` subagents must be tagged distinctly.
- Sub-tool `progress` records (e.g., a subagent's WebSearch query updates) are
  useful for visualizing intra-tool work but shouldn't be counted as user turns.
- Telemetry sampler must batch writes — per-sample inserts will fragment
  Lance datasets.
- `manifest.json` resumes ingest from byte offsets — must handle truncation
  (file got smaller than last offset → re-ingest from 0).

## Alternatives considered

- **Web dashboard (FastAPI+React)** — more flexible filtering, but Perfetto
  gives zoom/pan/categories/search for free and is the standard for trace
  analysis. We keep the option open behind the Lance store.
- **Protobuf Perfetto trace** — smaller files, but adds a build dep; JSON
  loads identically in the viewer at this scale.
- **OpenTelemetry → Jaeger/Tempo** — overkill for a local tool and worse at
  swim-lane visualization than Perfetto.
- **SQLite for everything** — simpler dep story, but worse at the analytical
  scans the stats/hints layer needs over millions of rows.
- **Reuse `~/.codex/logs_2.sqlite`** — would skip the Codex parser but the
  file is owned by the live TUI (WAL contention, unstable schema).
- **Hint auto-fix (codemod / PR generator)** — out of scope; high blast radius,
  low confidence. Hints stay advisory.
