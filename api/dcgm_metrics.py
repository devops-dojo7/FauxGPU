"""DCGM-exporter-compatible metric names, so a standard NVIDIA DCGM Grafana
dashboard (e.g. grafana.com/grafana/dashboards/23382, the same one real
nvidia-dcgm-exporter deployments feed) renders simulator data directly —
same metric names and label shape (Hostname/gpu/UUID/modelName) real
clusters publish. Approximated from the simulator's own compute/power
model, not a byte-for-byte reproduction of dcgm-exporter's internals:
- GPU/memory temperature: linear interpolation from power draw (idle→TDP
  maps to a plausible ~30C→85C range), since the simulator has no thermal
  model of its own.
- Framebuffer used/free: a fixed 70%-utilized assumption against the GPU's
  VRAM capacity, since per-run VRAM usage isn't threaded through run meta.
- SM/mem clocks: fixed representative values per GPU tier, not live values.
- Power, temperature, and GR/tensor/DRAM "active" fractions all track which
  phase of the current step the run is actually in right now (see
  api/live_phase.py) — compute phase reads as high GR/tensor activity and
  power near the achievable-utilization draw; communication phase reads as
  low GR/tensor activity, high DRAM activity, and lower (network-bound)
  power. This is what makes a live panel show real pulsing load instead of
  one flat time-weighted average for the whole run. Small per-GPU jitter is
  added so 8 simulated GPUs don't read as one identical flat trace.
- Total energy consumption: power × the run's elapsed seconds so far —
  monotonically increasing within a run, like the real counter.
- XID errors: nonzero (a fixed representative code, 79 — "GPU has fallen off
  the bus") while an engine.chaos "xid_error" event injected via
  POST /runs/{run_id}/inject is active for the run's current step; 0
  otherwise. PCIe replay counter is still always 0 — no model for it.
NVLINK_BANDWIDTH_TOTAL is deliberately omitted — it's a real counter with no
clean source in this simulator, and a fabricated rate() would mislead more
than an empty panel.
"""

from __future__ import annotations

import math
import time

from api.live_phase import instantaneous_power_watts
from api.runs_store import RunState
from engine.chaos import ChaosEvent, is_active
from engine.gpu_specs import get_gpu

_XID_FALLEN_OFF_BUS = 79  # a representative real DCGM Xid code, for display only


def compute_dcgm_series(run: RunState) -> list[dict]:
    """Returns [{"name": ..., "labels": {...}, "value": ...}, ...] — one set
    per simulated GPU in the run (meta["total_gpus"]), only for runs that
    are currently running (a finished run has no "live" GPU to report on).
    """
    if run.status != "running":
        return []

    meta = run.meta or {}
    gpu_id = str(meta.get("gpu", ""))
    try:
        gpu = get_gpu(gpu_id)
    except ValueError:
        return []

    total_gpus = int(meta.get("total_gpus", 1)) or 1

    base_power, in_compute_phase = instantaneous_power_watts(run, gpu, utilization=0.35)

    latest = run.steps[-1] if run.steps else None
    elapsed_s = latest["elapsed_s"] if latest else 0.0
    latest_step_num = latest["step"] if latest else 0

    xid_active = any(
        is_active(ChaosEvent(**e), latest_step_num) for e in run.chaos_events if e["kind"] == "xid_error"
    )
    xid_value = _XID_FALLEN_OFF_BUS if xid_active else 0

    sm_clock = 1830 if gpu.tdp_watts >= 700 else 1500
    mem_clock = 2619 if gpu.mem_bandwidth_gbps > 2000 else 1215

    if in_compute_phase:
        gr_active, tensor_active, dram_active = 0.92, 0.85, 0.15
    else:
        gr_active, tensor_active, dram_active = 0.10, 0.05, 0.90

    series = []
    for i in range(total_gpus):
        # Small deterministic per-GPU jitter so a multi-GPU run doesn't read
        # as 8 perfectly identical overlapping lines.
        jitter = 1 + 0.04 * math.sin(i * 1.9 + time.time() * 0.7)
        power = base_power * jitter
        temp_c = 30 + (power / gpu.tdp_watts) * 55
        mem_temp_c = max(25.0, temp_c - 5)
        fb_used_mib = gpu.vram_gb * 1024 * 0.7 * (1 + 0.02 * math.sin(i * 2.3))
        fb_free_mib = gpu.vram_gb * 1024 - fb_used_mib
        energy_mj = power * elapsed_s * 1000

        labels = {
            "Hostname": f"simgpu-{meta.get('topology', 'node')}",
            "gpu": str(i),
            "UUID": f"GPU-{run.run_id}-{i}",
            "modelName": gpu.name,
        }

        def add(name: str, value: float) -> None:
            series.append({"name": name, "labels": labels, "value": value})

        add("DCGM_FI_DEV_POWER_USAGE", power)
        add("DCGM_FI_DEV_GPU_TEMP", temp_c)
        add("DCGM_FI_DEV_MEMORY_TEMP", mem_temp_c)
        add("DCGM_FI_DEV_FB_USED", fb_used_mib)
        add("DCGM_FI_DEV_FB_FREE", fb_free_mib)
        add("DCGM_FI_DEV_SM_CLOCK", sm_clock)
        add("DCGM_FI_DEV_MEM_CLOCK", mem_clock)
        add("DCGM_FI_PROF_GR_ENGINE_ACTIVE", gr_active * jitter)
        add("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", tensor_active * jitter)
        add("DCGM_FI_PROF_DRAM_ACTIVE", dram_active * jitter)
        add("DCGM_FI_DEV_XID_ERRORS", xid_value)
        add("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", 0)
        add("DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION", energy_mj)

    return series


def render_dcgm_prometheus_text(runs: list[RunState]) -> str:
    lines = ["# HELP simgpu note: DCGM_FI_* metrics below are simulated, not from real dcgm-exporter."]
    for run in runs:
        for s in compute_dcgm_series(run):
            label_str = ",".join(f'{k}="{v}"' for k, v in s["labels"].items())
            lines.append(f'{s["name"]}{{{label_str}}} {s["value"]}')
    return "\n".join(lines) + ("\n" if lines else "")


def dcgm_metric_items(runs: list[RunState]) -> list[dict]:
    now = time.time()
    items = []
    for run in runs:
        for s in compute_dcgm_series(run):
            items.append({"metric": {"__name__": s["name"], **s["labels"]}, "values": [s["value"]], "timestamps": [now]})
    return items
