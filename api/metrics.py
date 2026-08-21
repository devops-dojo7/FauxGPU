"""Prometheus-format /metrics endpoint exposing the current state of tracked
training runs — the bridge artifact for pointing a real Grafana/Prometheus
at this simulator, once it's network-reachable from wherever Grafana runs
(e.g. both deployed in the same cluster, or via a tunnel). Grafana/Prometheus
pull this on their own schedule; nothing here pushes anywhere.
"""

from __future__ import annotations

from api.dcgm_metrics import render_dcgm_prometheus_text
from api.live_phase import instantaneous_power_watts
from api.runs_store import store
from engine.gpu_specs import get_gpu


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_prometheus_metrics() -> str:
    lines: list[str] = []

    lines.append("# HELP simgpu_run_status Run status (1=running, 0=done)")
    lines.append("# TYPE simgpu_run_status gauge")
    lines.append("# HELP simgpu_run_step Latest step number reported by the run")
    lines.append("# TYPE simgpu_run_step gauge")
    lines.append("# HELP simgpu_run_total_steps Total steps configured for the run")
    lines.append("# TYPE simgpu_run_total_steps gauge")
    lines.append("# HELP simgpu_run_tokens_seen Cumulative tokens processed by the run")
    lines.append("# TYPE simgpu_run_tokens_seen gauge")
    lines.append("# HELP simgpu_run_power_watts Estimated per-GPU power draw for the run")
    lines.append("# TYPE simgpu_run_power_watts gauge")

    for run in store.list():
        meta = run.meta or {}
        model = _escape(str(meta.get("model", "unknown")))
        gpu_id = str(meta.get("gpu", "unknown"))
        topology = _escape(str(meta.get("topology", "unknown")))
        labels = f'run_id="{_escape(run.run_id)}",model="{model}",gpu="{_escape(gpu_id)}",topology="{topology}"'

        lines.append(f"simgpu_run_status{{{labels}}} {1 if run.status == 'running' else 0}")

        latest_step = run.steps[-1] if run.steps else None
        if latest_step:
            lines.append(f"simgpu_run_step{{{labels}}} {latest_step['step']}")
            lines.append(f"simgpu_run_tokens_seen{{{labels}}} {latest_step['tokens_seen']}")
        if "total_steps" in meta:
            lines.append(f"simgpu_run_total_steps{{{labels}}} {meta['total_steps']}")

        try:
            gpu = get_gpu(gpu_id)
            power, _ = instantaneous_power_watts(run, gpu, utilization=0.35)
            lines.append(f"simgpu_run_power_watts{{{labels}}} {power:.2f}")
        except ValueError:
            pass  # unknown gpu id, skip the power gauge for this run

    return "\n".join(lines) + "\n" + render_dcgm_prometheus_text(store.list())
