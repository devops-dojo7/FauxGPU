"""Tensor and pipeline parallelism on top of the existing data-parallel step
time (engine/compute.py). Together with the DP gradient all-reduce, this
mirrors the three ways NCCL collectives show up in real distributed training
(see e.g. https://lambda.ai/blog/introduction-multi-gpu-multi-node-distributed-training-nccl-2-0
for the DP/ring-all-reduce half of this picture):

- Data parallelism (DP): each replica holds the whole model; only gradients
  are synced, once per step, via ring all-reduce (already modeled in
  compute.py). Communication volume is fixed regardless of batch size.
- Tensor parallelism (TP): a single layer's matmuls are split across GPUs,
  cutting per-GPU compute ~linearly, but requiring an all-reduce of
  activations *inside every layer* (forward and backward) — far more
  frequent, smaller messages, which is why TP is normally kept within a
  node on NVLink rather than spread across the slower inter-node fabric.
- Pipeline parallelism (PP): different layers live on different GPUs, which
  hand activations to each other in sequence. The first/last microbatches
  can't overlap with anything, leaving GPUs idle — the "pipeline bubble" —
  shrinking as more microbatches keep the pipeline full.

Total GPUs used = topology.total_gpus (the DP replica group) * tp_degree * pp_degree.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.compute import GBPS_TO_BYTES_PER_SEC, StepTimeBreakdown, estimate_step_time
from engine.memory import ModelShape, bytes_per_param
from engine.topology import ClusterTopology


def tensor_parallel_communication_seconds(
    model: ModelShape,
    batch_size: int,
    seq_len: int,
    precision: str,
    tp_degree: int,
    nvlink_gbps: float | None,
    forward_and_backward: bool = True,
) -> float:
    """Two activation all-reduces per layer (attention output + MLP output),
    over NVLink (TP requires low-latency intra-node links — modeled here as
    unavailable, i.e. infinite time, if the GPU has no NVLink and
    tp_degree > 1). forward_and_backward=True (training's default) doubles
    that for the backward pass; inference call sites pass False, since
    serving only ever runs a forward pass.
    """
    if tp_degree <= 1:
        return 0.0
    if not nvlink_gbps:
        return float("inf")

    activation_bytes = batch_size * seq_len * model.hidden_dim * bytes_per_param(precision)
    bytes_per_allreduce = activation_bytes * 2 * (tp_degree - 1) / tp_degree
    passes = 2 if forward_and_backward else 1
    allreduces_per_step = model.num_layers * 2 * passes  # attn+mlp, x2 for fwd+bwd when training
    bandwidth_bytes_per_sec = nvlink_gbps * GBPS_TO_BYTES_PER_SEC
    return (bytes_per_allreduce * allreduces_per_step) / bandwidth_bytes_per_sec


def pipeline_bubble_fraction(pp_degree: int, num_microbatches: int) -> float:
    """Fraction of step time GPUs sit idle waiting for the pipeline to fill
    and drain. Shrinks toward 0 as num_microbatches grows relative to
    pp_degree — the standard reason to use many small microbatches with PP.
    """
    if pp_degree <= 1 or num_microbatches <= 0:
        return 0.0
    return (pp_degree - 1) / num_microbatches


@dataclass(frozen=True)
class ParallelStepTimeBreakdown:
    compute_s: float
    dp_communication_s: float
    tp_communication_s: float
    pipeline_bubble_s: float
    total_gpus: int

    @property
    def total_s(self) -> float:
        return self.compute_s + self.dp_communication_s + self.tp_communication_s + self.pipeline_bubble_s


def estimate_parallel_step_time(
    model: ModelShape,
    topology: ClusterTopology,
    tokens_per_step: int,
    precision: str,
    utilization: float,
    tp_degree: int = 1,
    pp_degree: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    num_microbatches: int = 1,
) -> ParallelStepTimeBreakdown:
    """DP step time (existing formula) with per-GPU compute divided by
    tp_degree, plus TP's per-layer activation all-reduce and PP's pipeline
    bubble overhead layered on top.
    """
    dp_step: StepTimeBreakdown = estimate_step_time(model, topology, tokens_per_step, precision, utilization)

    compute_s = dp_step.compute_s / max(tp_degree, 1)
    bubble_s = compute_s * pipeline_bubble_fraction(pp_degree, num_microbatches)
    tp_comm_s = tensor_parallel_communication_seconds(
        model, batch_size, seq_len, precision, tp_degree, topology.gpu.nvlink_gbps
    )

    return ParallelStepTimeBreakdown(
        compute_s=compute_s,
        dp_communication_s=dp_step.communication_s,
        tp_communication_s=tp_comm_s,
        pipeline_bubble_s=bubble_s,
        total_gpus=topology.total_gpus * tp_degree * pp_degree,
    )
