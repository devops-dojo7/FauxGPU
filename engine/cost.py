"""Cost-to-train estimation: step time * cluster $/hr, accumulated to a target
token budget. This is where GPU/vendor choice and topology translate into a
dollar figure people can actually compare.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.compute import StepTimeBreakdown
from engine.topology import ClusterTopology


@dataclass(frozen=True)
class CostEstimate:
    total_steps: int
    total_time_hours: float
    total_cost_usd: float
    cost_per_1k_tokens_usd: float


def estimate_training_cost(
    topology: ClusterTopology,
    step_time: StepTimeBreakdown,
    tokens_per_step: int,
    total_training_tokens: float,
    total_gpus_override: int | None = None,
) -> CostEstimate:
    """step_time only needs a `.total_s` property, so this also accepts
    engine.parallelism.ParallelStepTimeBreakdown — pass total_gpus_override
    in that case since TP/PP multiply the GPU count beyond topology.total_gpus.
    """
    total_steps = max(1, int(-(-total_training_tokens // tokens_per_step)))  # ceil div
    total_time_s = total_steps * step_time.total_s
    total_time_hours = total_time_s / 3600.0

    total_gpus = total_gpus_override if total_gpus_override is not None else topology.total_gpus
    cluster_price_per_hr = topology.gpu.price_per_hr_usd * total_gpus
    total_cost_usd = total_time_hours * cluster_price_per_hr

    cost_per_1k_tokens_usd = (total_cost_usd / total_training_tokens) * 1000

    return CostEstimate(
        total_steps=total_steps,
        total_time_hours=total_time_hours,
        total_cost_usd=total_cost_usd,
        cost_per_1k_tokens_usd=cost_per_1k_tokens_usd,
    )
