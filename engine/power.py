"""GPU power draw estimation — linear interpolation between idle and TDP based
on how "busy" the GPU is in a given phase, the same shape real power meters
show: near-idle when stalled on I/O, climbing toward TDP under load.

Not a substitute for real telemetry (nvidia-smi / DCGM) — a teaching
approximation so the simulator's monitoring view has believable numbers that
react to the same utilization drivers as the rest of the app (compute
utilization for training, and a fixed HBM-traffic-heavy estimate for decode,
which stays power-hungry even though SMs aren't compute-bound).
"""

from __future__ import annotations

from engine.gpu_specs import GpuSpec

DECODE_POWER_FRACTION = 0.55  # memory-bound but HBM traffic keeps power draw well above idle
COMMUNICATION_POWER_FRACTION = 0.25  # SMs mostly stalled waiting on NCCL collectives


def power_watts(gpu: GpuSpec, busy_fraction: float) -> float:
    """busy_fraction in [0, 1]: 0 = idle, 1 = full TDP."""
    busy_fraction = max(0.0, min(1.0, busy_fraction))
    return gpu.idle_watts + busy_fraction * (gpu.tdp_watts - gpu.idle_watts)


def training_step_power_watts(gpu: GpuSpec, compute_s: float, communication_s: float, utilization: float) -> float:
    """Time-weighted average power across a step's compute phase (draws at
    `utilization` fraction of TDP, same efficiency factor used for step time)
    and communication phase (network-bound, lower draw).
    """
    total_s = compute_s + communication_s
    if total_s <= 0:
        return gpu.idle_watts
    compute_power = power_watts(gpu, utilization)
    comm_power = power_watts(gpu, COMMUNICATION_POWER_FRACTION)
    return (compute_s * compute_power + communication_s * comm_power) / total_s


def prefill_power_watts(gpu: GpuSpec, utilization: float) -> float:
    return power_watts(gpu, utilization)


def decode_power_watts(gpu: GpuSpec) -> float:
    return power_watts(gpu, DECODE_POWER_FRACTION)
