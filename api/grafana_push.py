"""Optional Grafana Cloud integration: pushes live run metrics via
Prometheus remote-write, and posts start/finish annotations via the
standard Grafana HTTP API. Both are entirely optional — every function here
is a no-op (or returns False) if the relevant env vars aren't set, so the
simulator works standalone with zero Grafana connection required.

Two separate credentials, deliberately not conflated:
- GRAFANA_REMOTE_WRITE_URL / GRAFANA_PROM_USERNAME / GRAFANA_PROM_TOKEN:
  a Grafana Cloud Access Policy token scoped to metrics:write, for pushing
  time series into the Prometheus/Mimir datasource.
- GRAFANA_URL / GRAFANA_API_KEY: a Grafana service account token, for the
  regular Grafana HTTP API (annotations, dashboards).
"""

from __future__ import annotations

import os
import time

import requests
from prometheus_remote_writer import RemoteWriter

from api.dcgm_metrics import dcgm_metric_items
from api.live_phase import instantaneous_power_watts
from api.runs_store import store
from engine.gpu_specs import get_gpu

_REMOTE_WRITE_URL = os.environ.get("GRAFANA_REMOTE_WRITE_URL")
_PROM_USERNAME = os.environ.get("GRAFANA_PROM_USERNAME")
_PROM_TOKEN = os.environ.get("GRAFANA_PROM_TOKEN")

_GRAFANA_URL = os.environ.get("GRAFANA_URL")
_GRAFANA_API_KEY = os.environ.get("GRAFANA_API_KEY")

_writer: RemoteWriter | None = None
if _REMOTE_WRITE_URL and _PROM_USERNAME and _PROM_TOKEN:
    _writer = RemoteWriter(
        url=_REMOTE_WRITE_URL,
        auth={"username": _PROM_USERNAME, "password": _PROM_TOKEN},
        user_agent="gpu-cluster-simulator/1.0",
    )


def push_available() -> bool:
    return _writer is not None


def annotations_available() -> bool:
    return bool(_GRAFANA_URL and _GRAFANA_API_KEY)


def push_run_metrics() -> int:
    """Push current gauges for every tracked run. Returns samples sent."""
    if _writer is None:
        return 0

    now = time.time()
    items = []
    for run in store.list():
        meta = run.meta or {}
        gpu_id = str(meta.get("gpu", ""))
        try:
            gpu = get_gpu(gpu_id)
        except ValueError:
            gpu = None
        labels = {
            "run_id": run.run_id,
            "model": str(meta.get("model", "unknown")),
            "gpu": gpu_id,
            "gpu_name": gpu.name if gpu else "unknown",
            "topology": str(meta.get("topology", "unknown")),
        }
        items.append(
            {
                "metric": {"__name__": "simgpu_run_status", **labels},
                "values": [1.0 if run.status == "running" else 0.0],
                "timestamps": [now],
            }
        )
        latest = run.steps[-1] if run.steps else None
        if latest:
            items.append({"metric": {"__name__": "simgpu_run_step", **labels}, "values": [latest["step"]], "timestamps": [now]})
            items.append(
                {"metric": {"__name__": "simgpu_run_tokens_seen", **labels}, "values": [latest["tokens_seen"]], "timestamps": [now]}
            )
        if gpu is not None:
            power, _ = instantaneous_power_watts(run, gpu, utilization=0.35)
            items.append({"metric": {"__name__": "simgpu_run_power_watts", **labels}, "values": [power], "timestamps": [now]})

    items.extend(dcgm_metric_items(store.list()))

    if not items:
        return 0
    result = _writer.send(items)
    return result.samples_sent


def post_annotation(text: str, tags: list[str]) -> None:
    """Best-effort: a Grafana outage should never fail a simulator run."""
    if not annotations_available():
        return
    try:
        requests.post(
            f"{_GRAFANA_URL.rstrip('/')}/api/annotations",
            headers={"Authorization": f"Bearer {_GRAFANA_API_KEY}", "Content-Type": "application/json"},
            json={"text": text, "tags": tags, "time": int(time.time() * 1000)},
            timeout=5,
        )
    except requests.RequestException:
        pass
