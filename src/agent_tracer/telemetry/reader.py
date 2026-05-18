"""Read telemetry samples from LanceDB and align them to event windows.

The reader is intentionally thin: it loads a time-windowed slice of the
``gpu_telemetry`` and ``system_telemetry`` tables into Arrow, then exposes
``window_stats(ts_start_us, ts_end_us)`` for cheap repeated lookups during
detector runs.

``TelemetryReader`` returns ``None`` from every method when the dataset
doesn't exist, so detectors can treat absence of telemetry as "skip this
detector" without a hard failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa


DEFAULT_DATASET = Path.home() / ".cache" / "agent-tracer" / "telemetry.lance"


@dataclass(slots=True)
class WindowStats:
    """Aggregates over a [ts_start_us, ts_end_us) telemetry slice."""

    n_gpu_samples: int = 0
    n_system_samples: int = 0
    # Per-metric: (mean, max, min). None entries mean no data for the metric.
    gpu_util_mean: float | None = None
    gpu_util_max: float | None = None
    vram_used_max_mb: float | None = None
    vram_total_mb: float | None = None
    gpu_power_max_w: float | None = None
    gpu_temp_max_c: float | None = None
    cpu_util_mean: float | None = None
    cpu_util_max: float | None = None
    load_1m_max: float | None = None
    mem_used_max_mb: float | None = None


@dataclass(slots=True)
class TelemetryReader:
    """Read GPU + system telemetry from a Lance dataset, sliced to a window."""

    dataset: Path = DEFAULT_DATASET
    _loaded: bool = False
    _gpu: pa.Table | None = None       # type: ignore[name-defined]
    _system: pa.Table | None = None    # type: ignore[name-defined]
    _gpu_ts: list[int] = field(default_factory=list)
    _system_ts: list[int] = field(default_factory=list)

    def exists(self) -> bool:
        """Quick check without forcing a load."""
        return Path(self.dataset).is_dir()

    def load(self) -> bool:
        """Materialize both tables into Arrow. Idempotent. Returns True on success."""
        if self._loaded:
            return self._gpu is not None or self._system is not None
        if not self.exists():
            self._loaded = True
            return False
        try:
            import lancedb
        except ImportError:
            self._loaded = True
            return False
        try:
            db = lancedb.connect(self.dataset)
            tables = getattr(db.list_tables(), "tables", None) or list(db.list_tables())
            if "gpu_telemetry" in tables:
                self._gpu = db.open_table("gpu_telemetry").to_arrow().sort_by("ts_us")
                self._gpu_ts = self._gpu.column("ts_us").to_pylist()
            if "system_telemetry" in tables:
                self._system = (
                    db.open_table("system_telemetry").to_arrow().sort_by("ts_us")
                )
                self._system_ts = self._system.column("ts_us").to_pylist()
        except (OSError, ValueError):
            self._loaded = True
            return False
        self._loaded = True
        return self._gpu is not None or self._system is not None

    def window_stats(self, ts_start_us: int, ts_end_us: int) -> WindowStats:
        """Return aggregates over [ts_start_us, ts_end_us). Empty WindowStats if no data."""
        if not self._loaded:
            self.load()
        stats = WindowStats()
        if ts_end_us <= ts_start_us:
            return stats

        if self._gpu is not None and self._gpu_ts:
            lo, hi = _bisect_range(self._gpu_ts, ts_start_us, ts_end_us)
            if lo < hi:
                slice_ = self._gpu.slice(lo, hi - lo)
                stats.n_gpu_samples = slice_.num_rows
                stats.gpu_util_mean = _col_mean(slice_, "util_pct")
                stats.gpu_util_max = _col_max(slice_, "util_pct")
                stats.vram_used_max_mb = _col_max(slice_, "vram_used_mb")
                stats.vram_total_mb = _col_max(slice_, "vram_total_mb")
                stats.gpu_power_max_w = _col_max(slice_, "power_w")
                stats.gpu_temp_max_c = _col_max(slice_, "temp_c")

        if self._system is not None and self._system_ts:
            lo, hi = _bisect_range(self._system_ts, ts_start_us, ts_end_us)
            if lo < hi:
                slice_ = self._system.slice(lo, hi - lo)
                stats.n_system_samples = slice_.num_rows
                stats.cpu_util_mean = _col_mean(slice_, "cpu_util_pct")
                stats.cpu_util_max = _col_max(slice_, "cpu_util_pct")
                stats.load_1m_max = _col_max(slice_, "load_1m")
                stats.mem_used_max_mb = _col_max(slice_, "mem_used_mb")
        return stats


def _bisect_range(ts: list[int], lo_v: int, hi_v: int) -> tuple[int, int]:
    """Half-open [lo_v, hi_v) → indices into a sorted ``ts`` list."""
    import bisect

    return bisect.bisect_left(ts, lo_v), bisect.bisect_left(ts, hi_v)


def _col_mean(table: pa.Table, name: str) -> float | None:  # type: ignore[name-defined]
    import pyarrow.compute as pc

    if name not in table.column_names:
        return None
    out = pc.mean(table.column(name))
    return float(out.as_py()) if out.as_py() is not None else None


def _col_max(table: pa.Table, name: str) -> float | None:  # type: ignore[name-defined]
    import pyarrow.compute as pc

    if name not in table.column_names:
        return None
    out = pc.max(table.column(name))
    return float(out.as_py()) if out.as_py() is not None else None
