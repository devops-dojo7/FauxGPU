"""Step-time estimation: how long one training step takes, split into compute time
(FLOPs / achievable TFLOPS) and communication time (gradient all-reduce over the
cluster's bottleneck interconnect).
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.memory import ModelShape, bytes_per_param, lora_trainable_params
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

    The "4x bwd" splits evenly into 2x recomputing the backward pass's
    *input* gradient (dL/dx, needed to keep propagating error to earlier
    layers regardless of which weights are trainable) and 2x the *weight*
    gradient (dL/dW, needed only for parameters that actually get updated)
    — see lora_flops_multiplier, which zeroes out the weight-gradient half
    for LoRA/QLoRA's frozen base weights.
    """
    return 6.0 * model.effective_active_params * tokens_per_step


# Of the "6x" approximation above, the split is 2x forward + 2x backward
# input-gradient (dL/dx) + 2x backward weight-gradient (dL/dW) — see e.g.
# https://erenovic.github.io/posts/2025-10-07-recap-on-lora-without-regret/
# summarizing Thinking Machines' "LoRA without Regret" (2025): "LoRA saves
# compute because we only calculate gradients for the tiny A and B
# matrices... roughly a 30% reduction in FLOPs per training step". LoRA/
# QLoRA must still backprop the *input* gradient through the frozen base
# (2x) to keep the error signal flowing to earlier layers, and still needs
# the full 2x forward pass — only the 2x weight-gradient term shrinks, and
# only for the frozen majority (the tiny adapter itself still needs its
# own, negligible, weight-gradient compute). In the N -> infinity limit
# (adapter params vanishingly small next to the frozen base), this is
# 4/6 = 1/3 fewer FLOPs, matching the cited "~30%" almost exactly; this
# model computes the adapter's own small share exactly rather than
# assuming it away.
LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION = 2.0 / 6.0


def lora_flops_multiplier(model: ModelShape, peft_rank: int, peft_target_modules: int) -> float:
    """Fraction of flops_per_step's FLOPs actually needed under LoRA/QLoRA:
    1.0 minus the weight-gradient FLOPs skipped for the frozen (non-
    adapter) parameter share. See LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION.

    Uses model.effective_active_params (not model.params) as the frozen-
    share denominator, matching flops_per_step's own convention — for MoE
    models this is correct precisely because LoRA only ever adapts the
    attention projections (Wq/Wk/Wv/Wo), which are dense (not routed) in
    essentially every MoE architecture this project models (Mixtral,
    DeepSeek, etc.); the adapter's tiny share is being compared against
    the same per-token active-compute base flops_per_step already uses,
    not the full (mostly-inactive) resident parameter count.
    """
    adapter_params = lora_trainable_params(model, peft_rank, peft_target_modules)
    frozen_fraction = max(0.0, 1.0 - adapter_params / model.effective_active_params)
    return 1.0 - LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION * frozen_fraction


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


# ZeRO stage 3 / FSDP "full sharding" needs an extra all-gather of each
# layer's parameters before it can be used (forward and, again, backward),
# on top of the reduce-scatter every ZeRO stage already does for
# gradients — the ZeRO paper's own communication analysis (Rajbhandari et
# al. 2020, Sec. 7) reports this lands at "a modest 50% increase" over
# plain DP's baseline all-reduce volume. Stages 1/2 (optimizer
# states/gradients only) do not change communication volume at all — only
# stage 3's added parameter sharding does.
ZERO_STAGE_3_COMM_MULTIPLIER = 1.5


def zero_communication_multiplier(zero_stage: int) -> float:
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2, or 3; got {zero_stage}")
    return ZERO_STAGE_3_COMM_MULTIPLIER if zero_stage >= 3 else 1.0


def estimate_step_time(
    model: ModelShape,
    topology: ClusterTopology,
    tokens_per_step: int,
    precision: str = "bf16",
    utilization: float = 0.35,
    zero_stage: int = 0,
    peft_method: str = "full",
    peft_rank: int = 8,
    peft_target_modules: int = 2,
) -> StepTimeBreakdown:
    """Total wall-clock time for one data-parallel training step across the
    given topology. Compute happens in parallel per-GPU (tokens_per_step is
    split across GPUs); communication is the gradient all-reduce across the
    cluster's bottleneck link (NVLink within a node, the inter-node fabric
    once you scale beyond one node).

    zero_stage (see engine.memory.compute_vram_breakdown for the memory
    side of this) scales up communication time only at stage 3 — see
    zero_communication_multiplier.

    peft_method="lora"/"qlora" (see engine.memory.compute_vram_breakdown for
    the memory side) affects both terms of this function, each for a
    distinct, independently-derivable reason:

    - Communication: the gradient all-reduce payload shrinks to just the
      tiny trainable adapter's gradients instead of the full model's —
      with the vast majority of parameters frozen, there's nothing to
      synchronize for them.
    - Compute: see lora_flops_multiplier — backpropagating the *weight*
      gradient (dL/dW) is skipped for frozen parameters (only the input
      gradient dL/dx, needed to keep propagating error to earlier layers,
      plus the tiny adapter's own weight gradient, are computed), which is
      a real, separately-documented ~30% FLOPs reduction (see
      lora_flops_multiplier's docstring for the citation) — distinct from,
      and in addition to, the communication saving above.

    Neither of the above reproduces the LoRA paper's own measured "25%"
    throughput speedup for GPT-3 175B (Sec 4.2, footnote — 32.5 -> 43.1
    tokens/s/GPU) as a black-box multiplier, since that number folds in
    other differences (a different model-parallel sharding count between
    the compared runs) beyond just compute+communication savings — this
    model only applies the two effects it can derive directly. peft_method
    is mutually exclusive with zero_stage != 0, matching engine.memory.
    """
    if peft_method != "full" and zero_stage != 0:
        raise ValueError(
            f"zero_stage={zero_stage} combined with peft_method={peft_method!r} is not modeled "
            "(mutually exclusive here) — see engine.memory.compute_vram_breakdown's docstring."
        )

    tokens_per_gpu = max(1, tokens_per_step // topology.total_gpus)
    step_flops = flops_per_step(model, tokens_per_gpu)
    if peft_method in ("lora", "qlora"):
        step_flops *= lora_flops_multiplier(model, peft_rank, peft_target_modules)
    tflops = achievable_tflops(topology.gpu.bf16_tflops, utilization)
    compute_s = step_flops / (tflops * 1e12)

    if peft_method in ("lora", "qlora"):
        grad_bytes = lora_trainable_params(model, peft_rank, peft_target_modules) * bytes_per_param(precision)
    else:
        grad_bytes = model.params * bytes_per_param(precision)
    bottleneck_gbps = topology.bottleneck_bandwidth_gbps()
    comm_s = ring_all_reduce_seconds(grad_bytes, topology.total_gpus, bottleneck_gbps)
    comm_s *= zero_communication_multiplier(zero_stage)

    return StepTimeBreakdown(compute_s=compute_s, communication_s=comm_s)
