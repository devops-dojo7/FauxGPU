"""Job-queue trace replay: parses a simple CSV job trace into
engine.scheduler.SchedJob objects, so a trace can be run through the
existing multi-tenant scheduler (engine/scheduler.py) unchanged — replay is
just "load jobs from a file" feeding the same simulate_scheduler already
used for hand-entered jobs.

CSV schema: job_id,team,priority,gpu_count,submit_time,duration — matching
SchedJob's fields exactly. This is a simplified schema of this project's own
design, not a parser for any specific real published trace format (Philly,
Alibaba PAI, MLPerf, etc.), which have far more columns than this simulator
models anyway. The bundled sample trace (data/sample_trace.csv) is an
illustrative, synthetic trace with a bursty arrival pattern and a mix of
small/large jobs inspired by publicly-known characteristics of ML cluster
traces — it is not a literal excerpt of any specific published trace.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from engine.scheduler import SchedJob

_REQUIRED_COLUMNS = ("job_id", "team", "priority", "gpu_count", "submit_time", "duration")
_SAMPLE_TRACE_PATH = Path(__file__).parent / "data" / "sample_trace.csv"


def parse_trace_csv(csv_text: str) -> list[SchedJob]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise ValueError("Trace CSV is empty")
    missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"Trace CSV is missing required column(s): {missing}. Expected: {list(_REQUIRED_COLUMNS)}")

    jobs: list[SchedJob] = []
    for i, row in enumerate(reader, start=2):  # row 1 is the header
        try:
            jobs.append(
                SchedJob(
                    job_id=row["job_id"],
                    team=row["team"],
                    priority=int(row["priority"]),
                    gpu_count=int(row["gpu_count"]),
                    submit_time=float(row["submit_time"]),
                    duration=float(row["duration"]),
                )
            )
        except (KeyError, ValueError) as e:
            raise ValueError(f"Trace CSV row {i} is malformed: {e}") from e

    if not jobs:
        raise ValueError("Trace CSV has no job rows")
    return jobs


def load_sample_trace_csv() -> str:
    return _SAMPLE_TRACE_PATH.read_text()
