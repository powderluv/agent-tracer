"""LanceDB writer for telemetry samples.

Two tables under one dataset directory:

* ``gpu_telemetry``    — one row per GPU per tick.
* ``system_telemetry`` — one row per tick (system-wide).

Writes are batched (``flush_every`` rows or ``flush_seconds`` — whichever
fires first) so per-tick inserts don't fragment the Lance dataset. Call
``flush()`` on graceful shutdown; ``close()`` flushes and closes.

We import lancedb lazily so the rest of the package stays import-clean
without the optional ``[store]`` extra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent_tracer.telemetry.samplers import GpuSample, SystemSample

if TYPE_CHECKING:  # avoid hard runtime dep at import time
    import lancedb
    import pyarrow as pa


_GPU_SCHEMA_FIELDS = [
    ("ts_us", "int64"),
    ("host", "string"),
    ("vendor", "string"),
    ("gpu_idx", "int32"),
    ("gpu_name", "string"),
    ("util_pct", "float32"),
    ("mem_util_pct", "float32"),
    ("vram_used_mb", "float32"),
    ("vram_total_mb", "float32"),
    ("power_w", "float32"),
    ("temp_c", "float32"),
]

_SYS_SCHEMA_FIELDS = [
    ("ts_us", "int64"),
    ("host", "string"),
    ("cpu_util_pct", "float32"),
    ("mem_used_mb", "float32"),
    ("mem_total_mb", "float32"),
    ("load_1m", "float32"),
    ("load_5m", "float32"),
    ("load_15m", "float32"),
]


@dataclass(slots=True)
class TelemetryWriter:
    path: Path
    flush_every: int = 256
    flush_seconds: float = 60.0
    _gpu_buf: list[dict[str, Any]] = field(default_factory=list)
    _sys_buf: list[dict[str, Any]] = field(default_factory=list)
    _last_flush: float = field(default_factory=time.monotonic)
    _db: lancedb.DBConnection | None = None  # type: ignore[name-defined]
    _gpu_schema: pa.Schema | None = None     # type: ignore[name-defined]
    _sys_schema: pa.Schema | None = None     # type: ignore[name-defined]

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    # --- public ----------------------------------------------------------

    def add_gpu(self, ts_us: int, host: str, sample: GpuSample) -> None:
        self._gpu_buf.append(
            {
                "ts_us": ts_us,
                "host": host,
                "vendor": sample.vendor,
                "gpu_idx": sample.gpu_idx,
                "gpu_name": sample.gpu_name,
                "util_pct": _f32(sample.util_pct),
                "mem_util_pct": _f32(sample.extras.get("mem_util_pct")),
                "vram_used_mb": _f32(sample.vram_used_mb),
                "vram_total_mb": _f32(sample.vram_total_mb),
                "power_w": _f32(sample.power_w),
                "temp_c": _f32(sample.temp_c),
            }
        )
        self._maybe_flush()

    def add_system(self, ts_us: int, sample: SystemSample) -> None:
        self._sys_buf.append(
            {
                "ts_us": ts_us,
                "host": sample.host,
                "cpu_util_pct": _f32(sample.cpu_util_pct),
                "mem_used_mb": _f32(sample.mem_used_mb),
                "mem_total_mb": _f32(sample.mem_total_mb),
                "load_1m": _f32(sample.load_1m),
                "load_5m": _f32(sample.load_5m),
                "load_15m": _f32(sample.load_15m),
            }
        )
        self._maybe_flush()

    def flush(self) -> tuple[int, int]:
        """Force-write any buffered rows. Returns (gpu_rows, sys_rows) flushed."""
        gpu_n = self._write_buf("gpu_telemetry", self._gpu_buf, _GPU_SCHEMA_FIELDS)
        sys_n = self._write_buf("system_telemetry", self._sys_buf, _SYS_SCHEMA_FIELDS)
        self._gpu_buf.clear()
        self._sys_buf.clear()
        self._last_flush = time.monotonic()
        return gpu_n, sys_n

    def close(self) -> None:
        self.flush()
        # lancedb.DBConnection has no explicit close; the wal flushes on flush().

    # --- internals -------------------------------------------------------

    def _maybe_flush(self) -> None:
        if (
            len(self._gpu_buf) + len(self._sys_buf) >= self.flush_every
            or (time.monotonic() - self._last_flush) >= self.flush_seconds
        ):
            self.flush()

    def _db_conn(self) -> lancedb.DBConnection:  # type: ignore[name-defined]
        if self._db is None:
            import lancedb

            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.path)
        return self._db

    def _schema(self, fields: list[tuple[str, str]]) -> pa.Schema:  # type: ignore[name-defined]
        import pyarrow as pa

        type_map = {
            "int64": pa.int64(),
            "int32": pa.int32(),
            "float32": pa.float32(),
            "string": pa.string(),
        }
        return pa.schema([(n, type_map[t]) for n, t in fields])

    def _write_buf(self, table: str, buf: list[dict[str, Any]], fields: list[tuple[str, str]]) -> int:
        if not buf:
            return 0
        import pyarrow as pa

        db = self._db_conn()
        schema = self._schema(fields)
        arr_table = pa.Table.from_pylist(buf, schema=schema)
        # lancedb.list_tables() returns a ListTablesResponse object with the
        # actual list under ``.tables``; iterating it yields Pydantic items.
        existing = getattr(db.list_tables(), "tables", []) or list(db.list_tables())
        if table in existing:
            db.open_table(table).add(arr_table)
        else:
            db.create_table(table, data=arr_table)
        return len(buf)


def _f32(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
