"""Step-time estimation: how long one training step takes, split into compute time
(FLOPs / achievable TFLOPS) and communication time (gradient all-reduce over the
cluster's bottleneck interconnect).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.memory import ModelShape, bytes_per_param
from engine.topology import ClusterTopology

GBPS_TO_BYTES_PER_SEC = 1e9 / 8  # Gb/s -> bytes/s


@dataclass(frozen=True)
class StepTimeBreakdown:
    compute_s: float
    communication_s: float

    @property
    def total_s(self) -> float:
        return self.compute_s + self.communication_s


def flops_per_step(model: ModelShape, tokens_per_step: int) -> float:
    """Standard approximation for forward+backward FLOPs (Kaplan et al. 2020):
    ~6 FLOPs per parameter per token (2x fwd, 4x bwd). For MoE models, only
    the active (routed) params do work for a given token, so this uses
    effective_active_params rather than the total resident param count.
    """
    return 6.0 * model.effective_active_params * tokens_per_step


def achievable_tflops(peak_tflops: float, utilization: float = 0.35) -> float:
    """Real training runs rarely hit peak nameplate TFLOPS due to kernel
    efficiency, memory stalls, and pipeline bubbles. 30-40% of peak is a
    commonly cited realistic range for well-optimized large model training.
    """
    return peak_tflops * utilization


def ring_all_reduce_seconds(payload_bytes: float, num_gpus: int, bandwidth_gbps: float) -> float:
    """Ring all-reduce communication time: each GPU sends/receives ~2*(N-1)/N of
    the payload, bounded by the link bandwidth. Returns 0 for a single GPU
    (no communication needed).
    """
    if num_gpus <= 1 or bandwidth_gbps <= 0:
        return 0.0
    bandwidth_bytes_per_sec = bandwidth_gbps * GBPS_TO_BYTES_PER_SEC
    data_moved = payload_bytes * 2 * (num_gpus - 1) / num_gpus
    return data_moved / bandwidth_bytes_per_sec


def estimate_step_time(
    model: ModelShape,
    topology: ClusterTopology,
    tokens_per_step: int,
    precision: str = "bf16",
    utilization: float = 0.35,
) -> StepTimeBreakdown:
    """Total wall-clock time for one data-parallel training step across the
    given topology. Compute happens in parallel per-GPU (tokens_per_step is
    split across GPUs); communication is the gradient all-reduce across the
    cluster's bottleneck link (NVLink within a node, the inter-node fabric
    once you scale beyond one node).
    """
    tokens_per_gpu = max(1, tokens_per_step // topology.total_gpus)
    step_flops = flops_per_step(model, tokens_per_gpu)
    tflops = achievable_tflops(topology.gpu.bf16_tflops, utilization)
    compute_s = step_flops / (tflops * 1e12)

    grad_bytes = model.params * bytes_per_param(precision)
    bottleneck_gbps = topology.bottleneck_bandwidth_gbps()
    comm_s = ring_all_reduce_seconds(grad_bytes, topology.total_gpus, bottleneck_gbps)

    return StepTimeBreakdown(compute_s=compute_s, communication_s=comm_s)
