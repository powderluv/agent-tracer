"""SVG chart generators for the markdown / PDF report.

Charts use matplotlib (no seaborn, no plotly) and write SVG files. The
report module references them as ``![](relative/path.svg)`` so both
markdown viewers and weasyprint's PDF renderer pick them up.

Each generator takes the data it needs (already aggregated) plus an
output ``Path`` and returns the relative filename (for embedding).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Matplotlib is imported lazily so the rest of the package stays importable
# without the optional ``[pdf]`` extra.
_BACKGROUND = "white"


def _setup_axes(figsize=(8.5, 4.5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, facecolor=_BACKGROUND, dpi=100)
    ax.set_facecolor(_BACKGROUND)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    return fig, ax


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=_BACKGROUND)
    import matplotlib.pyplot as plt

    plt.close(fig)


def chart_activity_by_day(activity: dict[str, dict[str, int]], out_path: Path) -> Path:
    """Vertical bar chart: tool calls per day. ``activity`` is the
    ``extras['activity_by_day']`` dict from report._compute_extras."""
    days = list(activity.keys())
    tool_calls = [activity[d]["tool_calls"] for d in days]
    fig, ax = _setup_axes(figsize=(11, 4))
    xs = range(len(days))
    ax.bar(xs, tool_calls, color="#3b82f6", width=0.85, zorder=2)
    ax.set_title("Tool calls per day", loc="left", fontsize=12, pad=10)
    ax.set_ylabel("Tool calls")
    # Show every Nth label so a 70-day axis stays readable.
    step = max(1, len(days) // 14)
    ax.set_xticks(list(xs)[::step])
    ax.set_xticklabels(days[::step], rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(days) - 0.5)
    _save(fig, out_path)
    return out_path


def chart_tool_calls_by_category(tool_cat_count, tool_cat_wall_us, out_path: Path) -> Path:
    """Two stacked horizontal bars: count vs wall-clock share by category."""
    cats = [c for c, _ in tool_cat_count.most_common()]
    counts = [tool_cat_count[c] for c in cats]
    walls = [tool_cat_wall_us[c] for c in cats]
    total_c = sum(counts) or 1
    total_w = sum(walls) or 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 1.7), facecolor=_BACKGROUND, dpi=100)
    ax.set_facecolor(_BACKGROUND)

    palette = [
        "#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899",
        "#06b6d4", "#84cc16", "#f97316", "#64748b",
    ]
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(cats)}

    def stacked(ax, y, values, total):
        left = 0
        for cat, v in zip(cats, values, strict=True):
            share = v / total
            ax.barh(y, share, left=left, color=color_map[cat], edgecolor="white", linewidth=0.5)
            if share >= 0.04:
                ax.text(
                    left + share / 2,
                    y,
                    cat,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                )
            left += share

    stacked(ax, 1, counts, total_c)
    stacked(ax, 0, walls, total_w)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["wall-clock share", "count share"], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    ax.set_title("Tool calls by category — count vs wall-clock", loc="left", fontsize=12, pad=10)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0)
    _save(fig, out_path)
    return out_path


def chart_top_tools(by_count, wall_us_by_tool: dict[str, int], out_path: Path, *, top_n: int = 15) -> Path:
    """Horizontal bar — top N tools by count, with wall-clock annotated."""
    top = by_count.most_common(top_n)
    names = [n for n, _ in top]
    counts = [c for _, c in top]
    walls = [wall_us_by_tool.get(n, 0) / 1_000_000 for n in names]

    fig, ax = _setup_axes(figsize=(9, max(4, 0.32 * top_n)))
    ys = range(len(names))[::-1]
    ax.barh(list(ys), counts, color="#3b82f6", zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"Top {top_n} tools by count (wall-clock annotated)", loc="left", fontsize=12, pad=10)
    # Annotate each bar with wall time.
    max_count = max(counts) if counts else 1
    for y, c, w in zip(ys, counts, walls, strict=True):
        ax.text(
            c + max_count * 0.015,
            y,
            _humanize_seconds(w),
            va="center",
            fontsize=8,
            color="#475569",
        )
    ax.set_xlim(0, max_count * 1.18)
    _save(fig, out_path)
    return out_path


def chart_tools_by_wall(wall_us_by_tool: dict[str, int], by_count, out_path: Path, *, top_n: int = 15) -> Path:
    """Horizontal bar — top N tools by wall-clock, with call-count annotated."""
    top = sorted(wall_us_by_tool.items(), key=lambda kv: -kv[1])[:top_n]
    names = [n for n, _ in top]
    walls_s = [us / 1_000_000 for _, us in top]
    counts = [by_count.get(n, 0) for n in names]

    fig, ax = _setup_axes(figsize=(9, max(4, 0.32 * top_n)))
    ys = range(len(names))[::-1]
    ax.barh(list(ys), walls_s, color="#10b981", zorder=2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Wall-clock (seconds, log scale)")
    if walls_s and max(walls_s) / max(1e-9, min(w for w in walls_s if w > 0) or 1e-9) > 100:
        ax.set_xscale("log")
    ax.set_title(f"Top {top_n} tools by wall-clock (call count annotated)", loc="left", fontsize=12, pad=10)
    max_w = max(walls_s) if walls_s else 1
    for y, w, c in zip(ys, walls_s, counts, strict=True):
        ax.text(
            w * 1.05 if ax.get_xscale() == "log" else w + max_w * 0.015,
            y,
            f"{c:,}×",
            va="center",
            fontsize=8,
            color="#475569",
        )
    _save(fig, out_path)
    return out_path


def chart_sessions_timeline(sessions: Iterable, out_path: Path) -> Path:
    """Gantt-style: each session as a horizontal span, color by source."""
    rows = sorted(sessions, key=lambda s: s.ts_start_us)
    if not rows:
        return out_path
    fig, ax = _setup_axes(figsize=(11, max(2.5, 0.32 * len(rows))))
    color_for = {"claude": "#3b82f6", "codex": "#10b981"}
    for i, s in enumerate(rows):
        y = len(rows) - 1 - i
        start_d = s.ts_start_us / 1_000_000 / 86400  # days since epoch
        dur_d = max(s.wallclock_s, 1) / 86400
        ax.barh(
            y,
            dur_d,
            left=start_d,
            height=0.7,
            color=color_for.get(s.source, "#64748b"),
            zorder=2,
        )
        ax.text(
            start_d,
            y,
            f"  {s.source}:{s.session_id[:8]}",
            va="center",
            ha="left",
            fontsize=7,
            color="white",
            fontweight="bold",
        )
    # Format x as YYYY-MM-DD
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    def fmt_days_since_epoch(x, _pos):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(x * 86400, tz=UTC).strftime("%Y-%m-%d")

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_days_since_epoch))
    ax.set_yticks([])
    ax.set_title("Session timeline (color = source)", loc="left", fontsize=12, pad=10)
    ax.grid(axis="x", linestyle=":", alpha=0.4, zorder=0)
    fig.autofmt_xdate(rotation=30, ha="right")
    _ = mdates  # silence unused-import lint
    _save(fig, out_path)
    return out_path


# --- helpers ---------------------------------------------------------------


def _humanize_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    if s < 86400:
        return f"{s / 3600:.1f}h"
    return f"{s / 86400:.1f}d"
