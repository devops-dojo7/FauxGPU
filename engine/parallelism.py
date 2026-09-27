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
- Expert parallelism (EP): MoE models shard their expert pool across GPUs
  rather than replicating every expert everywhere. Each token's chosen
  experts may live on a different rank, so every MoE layer needs two
  all-to-all collectives — "dispatch" (send each token to its expert-
  owning rank) and "combine" (send the expert's output back) — the
  defining, distinctively communication-bound cost of distributed MoE
  training (see engine.parallelism.expert_all_to_all_seconds).

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


def expert_all_to_all_seconds(
    model: ModelShape,
    tokens_per_step: int,
    precision: str,
    ep_degree: int,
    bandwidth_gbps: float | None,
    forward_and_backward: bool = True,
) -> float:
    """Communication time for MoE expert-parallel token dispatch + combine
    (Lepikhin et al. 2020, GShard, https://arxiv.org/abs/2006.16668; Fedus
    et al. 2022, Switch Transformer, https://arxiv.org/abs/2101.03961):
    every MoE layer routes each token to its top_k chosen experts, which
    may live on a different EP rank, requiring an all-to-all to send tokens
    there ("dispatch") and a second all-to-all to send expert outputs back
    ("combine") — this double all-to-all is described in the literature as
    "the defining cost of distributed MoE" (see e.g. the GShard-principles
    summary this project's own docs cite).

    Payload per all-to-all, per MoE layer: tokens_per_step * top_k *
    hidden_dim * bytes_per_param(precision) — each of a token's top_k
    routed copies carries one full hidden-dim activation vector. All-to-all
    (unlike ring all-reduce) sends the *full* payload once per collective,
    not a (N-1)/N fraction of it, since every rank both sends and receives
    a real payload rather than passing a partial-reduction result around a
    ring — see e.g. https://duoan.github.io/posts/moe-expert-parallelism-principles/
    (dispatch plus combine, "bytes ~= 2 * T * k * H * b").

    ep_degree <= 1 (experts not actually sharded across ranks, e.g. plain
    DP-replicated experts) returns 0.0 — nothing to route remotely. A model
    with no num_experts/top_k set (model.uses_expert_parallelism is False)
    is likewise unaffected regardless of ep_degree, since this project has
    no way to know its real routing shape. forward_and_backward=True
    (training's default) accounts for the backward pass needing its own
    dispatch+combine of activation *gradients* through the same routing
    pattern; inference-serving call sites (a future addition, not wired up
    yet) would pass False.
    """
    if ep_degree <= 1 or not model.uses_expert_parallelism:
        return 0.0
    if not bandwidth_gbps:
        return float("inf")

    payload_bytes_per_collective = tokens_per_step * model.top_k * model.hidden_dim * bytes_per_param(precision)
    collectives_per_layer = 2  # dispatch + combine
    passes = 2 if forward_and_backward else 1
    total_collectives = model.num_layers * collectives_per_layer * passes
    bandwidth_bytes_per_sec = bandwidth_gbps * GBPS_TO_BYTES_PER_SEC
    return (payload_bytes_per_collective * total_collectives) / bandwidth_bytes_per_sec


@dataclass(frozen=True)
class ParallelStepTimeBreakdown:
    compute_s: float
    dp_communication_s: float
    tp_communication_s: float
    pipeline_bubble_s: float
    expert_communication_s: float
    total_gpus: int

    @property
    def total_s(self) -> float:
        return (
            self.compute_s
            + self.dp_communication_s
            + self.tp_communication_s
            + self.pipeline_bubble_s
            + self.expert_communication_s
        )


def estimate_parallel_step_time(
    model: ModelShape,
    topology: ClusterTopology,
    tokens_per_step: int,
    precision: str,
    utilization: float,
    tp_degree: int = 1,
    pp_degree: int = 1,
    ep_degree: int = 1,
    batch_size: int = 1,
    seq_len: int = 2048,
    num_microbatches: int = 1,
    zero_stage: int = 0,
    peft_method: str = "full",
    peft_rank: int = 8,
    peft_target_modules: int = 2,
) -> ParallelStepTimeBreakdown:
    """DP step time (existing formula) with per-GPU compute divided by
    tp_degree, plus TP's per-layer activation all-reduce and PP's pipeline
    bubble overhead layered on top.

    zero_stage shards model states across topology's DP replica group (see
    engine.compute.estimate_step_time / engine.memory.compute_vram_breakdown)
    — orthogonal to tp_degree/pp_degree, which shard the model itself via a
    completely different mechanism.

    peft_method/peft_rank/peft_target_modules (see engine.compute.
    estimate_step_time) shrink both the DP gradient all-reduce (to the
    tiny LoRA adapter's gradients only) and the per-GPU compute (skipping
    frozen-weight backward-gradient FLOPs) — both orthogonal to
    tp_degree/pp_degree.

    ep_degree > 1 shards an MoE model's expert pool across that many ranks
    (see expert_all_to_all_seconds) — orthogonal to tp_degree/pp_degree/
    zero_stage/peft_method the same way TP/PP are, and a no-op (0 added
    communication) for a model with no num_experts/top_k set, regardless
    of ep_degree. Uses the NVLink bandwidth like TP's own collective,
    since EP's all-to-all is likewise normally kept within a fast-
    interconnect domain rather than spread across the slower inter-node
    fabric.
    """
    dp_step: StepTimeBreakdown = estimate_step_time(
        model,
        topology,
        tokens_per_step,
        precision,
        utilization,
        zero_stage=zero_stage,
        peft_method=peft_method,
        peft_rank=peft_rank,
        peft_target_modules=peft_target_modules,
    )

    compute_s = dp_step.compute_s / max(tp_degree, 1)
    bubble_s = compute_s * pipeline_bubble_fraction(pp_degree, num_microbatches)
    tp_comm_s = tensor_parallel_communication_seconds(
        model, batch_size, seq_len, precision, tp_degree, topology.gpu.nvlink_gbps
    )
    tokens_per_gpu = max(1, tokens_per_step // topology.total_gpus)
    ep_comm_s = expert_all_to_all_seconds(model, tokens_per_gpu, precision, ep_degree, topology.gpu.nvlink_gbps)
    # Only a model that actually uses expert parallelism should have its GPU
    # count multiplied by ep_degree — for a dense model (or an MoE model
    # with no num_experts/top_k set), ep_degree > 1 is a no-op exactly like
    # expert_all_to_all_seconds treats it (0 added communication), so
    # inflating total_gpus (and therefore total_cost_usd) for zero benefit
    # would be a real, silently-wrong answer rather than a harmless default.
    effective_ep_degree = max(ep_degree, 1) if model.uses_expert_parallelism else 1

    return ParallelStepTimeBreakdown(
        compute_s=compute_s,
        dp_communication_s=dp_step.communication_s,
        tp_communication_s=tp_comm_s,
        pipeline_bubble_s=bubble_s,
        expert_communication_s=ep_comm_s,
        total_gpus=topology.total_gpus * tp_degree * pp_degree * effective_ep_degree,
    )
