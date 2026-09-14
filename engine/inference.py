"""Inference/serving formulas, modeling the same ideas as llm-d
(https://llm-d.ai/): disaggregated prefill/decode serving and prefix-cache-aware
routing on top of vLLM-style engines on Kubernetes.

Two phases of autoregressive serving behave very differently:
- Prefill (time-to-first-token): a single forward pass over the whole prompt.
  Compute-bound — dominated by FLOPs, same shape as one training forward pass.
- Decode (time-per-output-token): one token at a time. Memory-bandwidth-bound —
  each step has to stream the active model weights (and the growing KV cache)
  out of HBM, not run much math, so it's bounded by memory bandwidth rather
  than TFLOPS.

Serving these two phases on the same GPU pool ("colocated") means a burst of
incoming prompts (prefill work) steals GPU cycles from in-flight decode
requests, spiking tail latency. llm-d's disaggregated serving pattern runs
prefill and decode on separate GPU pools so decode throughput stays steady
regardless of prefill load — that's the property `simulate_serving` models.

Also models vLLM's PagedAttention: KV cache is allocated in fixed-size
blocks rather than exact per-token, and `gpu_memory_utilization` caps how
much VRAM the engine will use in total (weights + KV cache), the same knob
vLLM exposes — used here for a "how many concurrent sequences fit" estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.compute import achievable_tflops
from engine.gpu_specs import GpuSpec
from engine.memory import ModelShape, active_weights_bytes, kv_cache_bytes_per_token, weights_bytes
from engine.parallelism import tensor_parallel_communication_seconds


def prefill_seconds(
    model: ModelShape,
    prompt_tokens: int,
    gpu: GpuSpec,
    precision: str,
    utilization: float = 0.35,
    cache_hit_fraction: float = 0.0,
    tp_degree: int = 1,
) -> float:
    """Time-to-first-token: one compute-bound forward pass over the prompt.
    cache_hit_fraction models a prefix-cache routing hit (llm-d's "Precise
    Prefix-Cache Aware Routing") — the matching prefix doesn't need recomputing.
    Uses effective_active_params so MoE models only pay for routed experts.

    tp_degree > 1 splits that compute across GPUs but adds a per-layer NVLink
    all-reduce (tensor_parallel_communication_seconds, forward-only — prefill
    never runs a backward pass). If the GPU has no NVLink, that returns inf,
    which (finite + inf == inf) propagates straight through as "infeasible"
    with no extra branching needed here.
    """
    effective_tokens = prompt_tokens * (1 - cache_hit_fraction)
    flops = 2.0 * model.effective_active_params * effective_tokens  # forward pass only, no backward
    tflops = achievable_tflops(gpu.bf16_tflops, utilization)
    compute_s = flops / (tflops * 1e12) / max(tp_degree, 1)
    comm_s = tensor_parallel_communication_seconds(
        model, batch_size=1, seq_len=prompt_tokens, precision=precision,
        tp_degree=tp_degree, nvlink_gbps=gpu.nvlink_gbps, forward_and_backward=False,
    )
    return compute_s + comm_s


def _paged_kv_tokens(kv_tokens: float, paged_attention: bool, block_size: int) -> float:
    """PagedAttention allocates KV cache in fixed-size blocks; the last block
    per sequence is usually partially filled, rounding usage up to the next
    block boundary. block_size=16 is vLLM's default.
    """
    if not paged_attention or block_size <= 0:
        return kv_tokens
    return math.ceil(kv_tokens / block_size) * block_size


def decode_step_seconds(
    model: ModelShape,
    gpu: GpuSpec,
    precision: str,
    batch_size: int,
    avg_kv_tokens: float,
    paged_attention: bool = False,
    block_size: int = 16,
    tp_degree: int = 1,
) -> float:
    """Time for one decode step (one new token for every sequence in the
    batch): memory-bandwidth-bound on reading active weights + each
    sequence's KV cache out of HBM.

    tp_degree > 1 shards both the active weights and each sequence's KV
    cache across the group (real Megatron-style TP splits KV heads too), so
    the byte-streaming time divides by tp_degree; a per-layer NVLink
    all-reduce (one new token per sequence this step, so seq_len=1) is added
    on top. Same inf-propagation behavior as prefill_seconds above when the
    GPU has no NVLink.
    """
    weights = active_weights_bytes(model, precision)
    kv_tokens = _paged_kv_tokens(avg_kv_tokens, paged_attention, block_size)
    kv_bytes = kv_cache_bytes_per_token(model, precision) * kv_tokens * batch_size
    total_bytes = weights + kv_bytes
    bandwidth_bytes_per_sec = gpu.mem_bandwidth_gbps * 1e9
    compute_s = (total_bytes / bandwidth_bytes_per_sec) / max(tp_degree, 1)
    comm_s = tensor_parallel_communication_seconds(
        model, batch_size=batch_size, seq_len=1, precision=precision,
        tp_degree=tp_degree, nvlink_gbps=gpu.nvlink_gbps, forward_and_backward=False,
    )
    return compute_s + comm_s


@dataclass(frozen=True)
class ServingResult:
    ttft_s: float
    decode_step_s: float
    tokens_per_sec_per_gpu: float
    prefill_interference_fraction: float
    colocated_tokens_per_sec_per_gpu: float
    disaggregated_tokens_per_sec_per_gpu: float
    tp_communication_overhead_fraction: float
    tokens_per_sec_per_gpu_amortized: float


def simulate_serving(
    model: ModelShape,
    gpu: GpuSpec,
    precision: str,
    prompt_tokens: int,
    output_tokens: int,
    decode_batch_size: int,
    requests_per_sec: float,
    cache_hit_fraction: float = 0.0,
    utilization: float = 0.35,
    paged_attention: bool = False,
    block_size: int = 16,
    tp_degree: int = 1,
) -> ServingResult:
    """Compare colocated vs. disaggregated prefill/decode serving on one GPU
    (or one GPU pool). Disaggregated decode throughput is unaffected by
    incoming request rate because prefill runs on separate GPUs; colocated
    decode throughput degrades as prefill bursts eat into the same GPU's
    cycles — the core lesson behind llm-d's disaggregated serving pattern.

    tp_degree > 1 shards this "one GPU" across a tensor-parallel group (see
    prefill_seconds/decode_step_seconds); tokens_per_sec_per_gpu then means
    the whole group's aggregate per-step throughput (a "replica", however
    many chips it spans), same as it always has at tp_degree=1.
    tokens_per_sec_per_gpu_amortized divides that back down to a genuine
    per-chip number, and tp_communication_overhead_fraction reports how much
    of one decode step went to the NVLink all-reduce rather than compute —
    both 0/unaffected at tp_degree=1.
    """
    ttft = prefill_seconds(model, prompt_tokens, gpu, precision, utilization, cache_hit_fraction, tp_degree)
    avg_kv_tokens = prompt_tokens + output_tokens / 2  # rough mean sequence length across decode
    decode_step = decode_step_seconds(
        model, gpu, precision, decode_batch_size, avg_kv_tokens, paged_attention, block_size, tp_degree
    )
    base_tps = decode_batch_size / decode_step

    prefill_load_fraction = min(1.0, max(0.0, requests_per_sec * ttft))
    colocated_tps = base_tps * (1 - prefill_load_fraction)

    tp_comm_s = tensor_parallel_communication_seconds(
        model, batch_size=decode_batch_size, seq_len=1, precision=precision,
        tp_degree=tp_degree, nvlink_gbps=gpu.nvlink_gbps, forward_and_backward=False,
    )
    if decode_step == float("inf"):
        tp_overhead_fraction = 1.0  # infeasible (no NVLink) — comm cost is the entire (infinite) step
    else:
        tp_overhead_fraction = tp_comm_s / decode_step if decode_step > 0 else 0.0

    return ServingResult(
        ttft_s=ttft,
        decode_step_s=decode_step,
        tokens_per_sec_per_gpu=base_tps,
        prefill_interference_fraction=prefill_load_fraction,
        colocated_tokens_per_sec_per_gpu=colocated_tps,
        disaggregated_tokens_per_sec_per_gpu=base_tps,
        tp_communication_overhead_fraction=tp_overhead_fraction,
        tokens_per_sec_per_gpu_amortized=base_tps / max(tp_degree, 1),
    )


@dataclass(frozen=True)
class CapacityEstimate:
    usable_vram_gb: float
    weights_gb: float
    kv_budget_gb: float
    max_concurrent_sequences: int


def estimate_serving_capacity(
    model: ModelShape,
    gpu: GpuSpec,
    precision: str,
    avg_seq_len: int,
    gpu_memory_utilization: float = 0.9,
    tp_degree: int = 1,
) -> CapacityEstimate:
    """vLLM-style capacity planning: gpu_memory_utilization caps the total
    fraction of VRAM the engine will use for weights + KV cache combined
    (vLLM's real --gpu-memory-utilization flag); whatever's left after
    weights is the KV cache budget, which bounds how many sequences of a
    given average length can be served concurrently.

    tp_degree > 1 shards weights evenly across the group (real Megatron-style
    TP), so usable_vram_gb/weights_gb below are per-GPU — this is what lets a
    model whose weights don't fit on one GPU (max_concurrent_sequences == 0
    at tp_degree=1) fit once sharded. KV cache is sharded too, so
    kv_budget_gb/max_concurrent_sequences are reported as the *aggregate*
    across the whole tp_degree-GPU group, not per-GPU — that's the number
    that actually answers "how many concurrent sequences can this deployment
    serve." At tp_degree=1 every field is identical to the un-sharded case.
    """
    tp_degree = max(tp_degree, 1)
    usable_bytes_per_gpu = gpu.vram_gb * 1e9 * gpu_memory_utilization
    w_bytes_per_gpu = weights_bytes(model, precision) / tp_degree
    kv_budget_bytes_per_gpu = max(0.0, usable_bytes_per_gpu - w_bytes_per_gpu)
    kv_budget_bytes_total = kv_budget_bytes_per_gpu * tp_degree
    per_seq_bytes = kv_cache_bytes_per_token(model, precision) * avg_seq_len
    max_sequences = int(kv_budget_bytes_total // per_seq_bytes) if per_seq_bytes > 0 else 0

    return CapacityEstimate(
        usable_vram_gb=usable_bytes_per_gpu / 1e9,
        weights_gb=w_bytes_per_gpu / 1e9,
        kv_budget_gb=kv_budget_bytes_total / 1e9,
        max_concurrent_sequences=max_sequences,
    )
