"""What-if configuration search: given a model shape and an objective
(minimize cost or minimize time), search GPU type x GPU count x
(tensor-parallel, pipeline-parallel) degree for feasible, ranked candidates.

Built entirely on top of the existing per-scenario formulas in
engine/memory.py, engine/topology.py, engine/parallelism.py and
engine/cost.py — this module only adds the search/ranking loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.cost import estimate_training_cost
from engine.gpu_specs import GpuSpec, get_gpu, list_gpus
from engine.memory import ModelShape, compute_vram_breakdown
from engine.parallelism import estimate_parallel_step_time
from engine.topology import ClusterTopology, build_topology

# Candidate search space. Kept small and power-of-two-shaped, consistent with
# how real clusters are usually sized, so the brute-force search stays fast.
_DP_GPU_COUNTS = (1, 2, 4, 8, 16, 32, 64)
_TP_DEGREES = (1, 2, 4, 8)
_PP_DEGREES = (1, 2, 4)


@dataclass(frozen=True)
class RecommendationCandidate:
    gpu_id: str
    gpu_name: str
    num_gpus: int
    tp_degree: int
    pp_degree: int
    total_cost_usd: float
    total_time_hours: float
    cost_per_1k_tokens_usd: float
    vram_per_gpu_gb: float
    vram_headroom_gb: float


def _sharded_model(model: ModelShape, tp_degree: int, pp_degree: int) -> ModelShape:
    """Approximate per-GPU shape after tensor + pipeline sharding: pipeline
    parallelism splits layers across stages, tensor parallelism further
    splits each layer's matmul weights — modeled here as dividing params
    (and active_params) by the combined shard factor while also reducing
    num_layers by pp_degree, so both the weight/optimizer/gradient VRAM and
    the activation VRAM (which scales with num_layers) shrink accordingly.
    A simplification, same tier as the rest of this teaching-tool's formulas.
    """
    shard = tp_degree * pp_degree
    layers_per_stage = max(1, model.num_layers // pp_degree)
    return ModelShape(
        params=model.params / shard,
        num_layers=layers_per_stage,
        hidden_dim=model.hidden_dim,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        num_kv_heads=model.num_kv_heads,
        active_params=(model.active_params / shard) if model.active_params is not None else None,
        kv_latent_dim=model.kv_latent_dim,
    )


def _topology_for_dp_gpus(gpu_id: str, dp_gpus: int) -> ClusterTopology:
    if dp_gpus <= 1:
        return build_topology("single_gpu", gpu_id)
    if dp_gpus <= 8:
        return build_topology("nvlink_node", gpu_id, gpus_per_node=dp_gpus)
    num_nodes = -(-dp_gpus // 8)  # ceil div, 8 GPUs/node
    return build_topology("multi_node", gpu_id, gpus_per_node=8, num_nodes=num_nodes)


def recommend_configurations(
    model: ModelShape,
    precision: str,
    tokens_per_step: int,
    total_training_tokens: float,
    objective: str,
    batch_size: int = 1,
    seq_len: int = 2048,
    utilization: float = 0.35,
    num_microbatches: int = 1,
    max_gpus: int = 64,
    budget_usd: float | None = None,
    max_time_hours: float | None = None,
    candidate_gpu_ids: list[str] | None = None,
    max_results: int = 10,
) -> list[RecommendationCandidate]:
    if objective not in ("cost", "time"):
        raise ValueError(f"Unknown objective: {objective!r}. Known: cost, time")

    if candidate_gpu_ids:
        gpus: list[GpuSpec] = [get_gpu(gid) for gid in candidate_gpu_ids]
    else:
        # Exclude unreleased/roadmap entries (price_per_hr_usd == 0, e.g.
        # ascend-970-roadmap) from the default pool — they aren't actually
        # obtainable, so recommending them isn't a real "what-if" answer.
        # Explicitly requesting one via candidate_gpu_ids still works.
        gpus = [g for g in list_gpus() if g.price_per_hr_usd > 0]

    candidates: list[RecommendationCandidate] = []
    for gpu in gpus:
        for tp_degree in _TP_DEGREES:
            if tp_degree > 1 and gpu.nvlink_gbps is None:
                continue  # TP needs NVLink — engine.parallelism already models this as infinite time
            for pp_degree in _PP_DEGREES:
                per_gpu_model = _sharded_model(model, tp_degree, pp_degree)
                vram = compute_vram_breakdown(per_gpu_model, precision, batch_size, seq_len, training=True)
                if vram.total_gb > gpu.vram_gb:
                    continue  # doesn't fit even before adding DP replicas

                for dp_gpus in _DP_GPU_COUNTS:
                    total_gpus = dp_gpus * tp_degree * pp_degree
                    if total_gpus > max_gpus:
                        continue
                    try:
                        topo = _topology_for_dp_gpus(gpu.id, dp_gpus)
                    except ValueError:
                        continue  # e.g. nvlink_node requested for a GPU with no NVLink

                    step = estimate_parallel_step_time(
                        model,
                        topo,
                        tokens_per_step=tokens_per_step,
                        precision=precision,
                        utilization=utilization,
                        tp_degree=tp_degree,
                        pp_degree=pp_degree,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        num_microbatches=num_microbatches,
                    )
                    if step.total_s == float("inf"):
                        continue

                    cost = estimate_training_cost(
                        topo, step, tokens_per_step, total_training_tokens, total_gpus_override=step.total_gpus
                    )
                    if budget_usd is not None and cost.total_cost_usd > budget_usd:
                        continue
                    if max_time_hours is not None and cost.total_time_hours > max_time_hours:
                        continue

                    candidates.append(
                        RecommendationCandidate(
                            gpu_id=gpu.id,
                            gpu_name=gpu.name,
                            num_gpus=step.total_gpus,
                            tp_degree=tp_degree,
                            pp_degree=pp_degree,
                            total_cost_usd=cost.total_cost_usd,
                            total_time_hours=cost.total_time_hours,
                            cost_per_1k_tokens_usd=cost.cost_per_1k_tokens_usd,
                            vram_per_gpu_gb=vram.total_gb,
                            vram_headroom_gb=gpu.vram_gb - vram.total_gb,
                        )
                    )

    candidates.sort(key=lambda c: c.total_cost_usd if objective == "cost" else c.total_time_hours)
    return candidates[:max_results]
