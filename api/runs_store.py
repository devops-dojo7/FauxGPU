"""In-memory store for live training-run progress reported by trainer Jobs.

Single-process, in-memory by design — this is a teaching tool watching one
demo run at a time, not a metrics system. If the API restarts, run history
is gone; that's an acceptable tradeoff for the scope here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class RunState:
    run_id: str
    status: str = "running"  # "running" | "done"
    meta: dict | None = None
    steps: list[dict] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class RunsStore:
    def __init__(self, max_runs: int = 20, max_steps_per_run: int = 2000):
        self._lock = threading.Lock()
        self._runs: dict[str, RunState] = {}
        self._max_runs = max_runs
        self._max_steps_per_run = max_steps_per_run

    def start(self, run_id: str, meta: dict) -> RunState:
        with self._lock:
            run = RunState(run_id=run_id, meta=meta)
            self._runs[run_id] = run
            if len(self._runs) > self._max_runs:
                oldest_id = min(self._runs, key=lambda k: self._runs[k].started_at)
                if oldest_id != run_id:
                    del self._runs[oldest_id]
            return run

    def add_step(self, run_id: str, step: dict) -> RunState | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run.steps.append(step)
            if len(run.steps) > self._max_steps_per_run:
                run.steps = run.steps[-self._max_steps_per_run :]
            run.updated_at = time.time()
            return run

    def finish(self, run_id: str) -> RunState | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run.status = "done"
            run.updated_at = time.time()
            return run

    def stop(self, run_id: str) -> RunState | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            run.status = "stopped"
            run.updated_at = time.time()
            return run

    def get(self, run_id: str) -> RunState | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[RunState]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.started_at, reverse=True)


store = RunsStore()
