"""Cost and utilization history for a training run, derived from its
recorded steps: walks the run's compute/communication split and $/GPU-hour
to show cost accumulating and utilization over the run's simulated
timeline. Pure function of already-known primitives (price, GPU count, step
timings, step history) — no store/framework dependency, so it works the
same whether the caller pulled those from RunsStore or elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

IDLE_UTILIZATION_THRESHOLD_PCT = 15.0


@dataclass(frozen=True)
class EconomicsPoint:
    step: int
    elapsed_hours: float
    cost_usd: float
    cumulative_cost_usd: float
    utilization_pct: float


@dataclass(frozen=True)
class EconomicsSummary:
    total_cost_usd: float
    total_gpu_hours: float
    avg_utilization_pct: float
    idle_cost_usd: float
    idle_pct: float


@dataclass(frozen=True)
class RunEconomics:
    points: list[EconomicsPoint]
    summary: EconomicsSummary


def gpu_hours_cost(price_per_hr_usd: float, total_gpus: int, hours: float) -> float:
    return price_per_hr_usd * total_gpus * hours


def compute_run_economics(
    price_per_hr_usd: float,
    total_gpus: int,
    compute_s_per_step: float,
    communication_s_per_step: float,
    steps: list[dict],
) -> RunEconomics:
    """steps: the run's recorded {"step", "tokens_seen", "elapsed_s"} dicts
    (RunState.steps) — only "step" is used here, since cost/utilization are
    derived from the run's simulated step time (compute_s_per_step +
    communication_s_per_step), not real wall-clock elapsed_s, which is
    compressed by the run's speedup factor and not a dollar-comparable hour.
    """
    total_s_per_step = compute_s_per_step + communication_s_per_step
    utilization_pct = (compute_s_per_step / total_s_per_step * 100) if total_s_per_step > 0 else 0.0
    is_idle = utilization_pct < IDLE_UTILIZATION_THRESHOLD_PCT

    points: list[EconomicsPoint] = []
    cumulative_cost = 0.0
    idle_cost = 0.0
    prev_hours = 0.0
    for s in steps:
        elapsed_hours = (s["step"] * total_s_per_step) / 3600.0
        interval_hours = max(0.0, elapsed_hours - prev_hours)
        interval_cost = gpu_hours_cost(price_per_hr_usd, total_gpus, interval_hours)
        cumulative_cost += interval_cost
        if is_idle:
            idle_cost += interval_cost
        points.append(
            EconomicsPoint(
                step=s["step"],
                elapsed_hours=elapsed_hours,
                cost_usd=interval_cost,
                cumulative_cost_usd=cumulative_cost,
                utilization_pct=utilization_pct,
            )
        )
        prev_hours = elapsed_hours

    summary = EconomicsSummary(
        total_cost_usd=cumulative_cost,
        total_gpu_hours=prev_hours * total_gpus,
        avg_utilization_pct=utilization_pct,
        idle_cost_usd=idle_cost,
        idle_pct=100.0 if is_idle else 0.0,
    )
    return RunEconomics(points=points, summary=summary)
