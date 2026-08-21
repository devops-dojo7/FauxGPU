"""Determines which phase (compute vs. communication) a run's simulated
GPU is "currently" in, based on the run's own observed pace so far — lets
pushed telemetry oscillate in real time with the run's actual step cadence
(whatever speedup was used) instead of reporting one flat time-weighted
average for the whole run. This is what makes a live Grafana panel show an
actual pulsing load while a test is running, instead of a dead flat line.
"""

from __future__ import annotations

import time

from api.runs_store import RunState
from engine.gpu_specs import GpuSpec
from engine.power import COMMUNICATION_POWER_FRACTION, power_watts, training_step_power_watts


def phase_fraction(run: RunState) -> float | None:
    """Fraction (0..1) of the way through the *current* simulated step,
    estimated from the run's average observed real-seconds-per-step so far.
    None if there isn't at least one reported step yet to estimate a pace from.
    """
    latest = run.steps[-1] if run.steps else None
    if not latest or latest["step"] <= 0:
        return None
    real_seconds_per_step = latest["elapsed_s"] / latest["step"]
    if real_seconds_per_step <= 0:
        return None
    time_since_start = time.time() - run.started_at
    return (time_since_start % real_seconds_per_step) / real_seconds_per_step


def instantaneous_power_watts(run: RunState, gpu: GpuSpec, utilization: float = 0.35) -> tuple[float, bool]:
    """Returns (power_watts, in_compute_phase). Falls back to the
    time-weighted average (in_compute_phase=True) when there isn't enough
    step data yet to place the run within its current step.
    """
    meta = run.meta or {}
    compute_s = meta.get("compute_s_per_step", 0.0)
    comm_s = meta.get("communication_s_per_step", 0.0)
    total_s = compute_s + comm_s

    frac = phase_fraction(run)
    if frac is None or total_s <= 0:
        return training_step_power_watts(gpu, compute_s, comm_s, utilization), True

    compute_share = compute_s / total_s
    in_compute = frac < compute_share
    power = power_watts(gpu, utilization) if in_compute else power_watts(gpu, COMMUNICATION_POWER_FRACTION)
    return power, in_compute
