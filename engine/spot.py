"""Spot/preemptible instance economics: a steep hourly discount in exchange
for preemption risk. Deliberately does *not* invent a per-GPU spot price in
engine/data/gpus.yaml (real spot pricing varies by cloud, region, and time
of day far more than on-demand pricing does, and fabricating 41 more numbers
would be pure guesswork) — instead it's a formula, matching this project's
"every number is computed, not looked up" approach: spot price = on-demand
price * (1 - discount), with `discount` a single, clearly-illustrative,
user-adjustable parameter.

A spot preemption is economically identical to a node_drain-style crash
(engine.chaos) requiring a checkpoint restore + redoing lost steps, so
expected recovery overhead reuses engine.checkpointing's
recovery_overhead_seconds formula rather than modeling that twice.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.checkpointing import recovery_overhead_seconds

# Illustrative: spot/preemptible instances commonly run 60-90% below
# on-demand across major clouds. One configurable midpoint, not a per-cloud
# lookup table (see gpus.yaml's own "illustrative snapshot, not live data"
# convention for on-demand pricing — spot pricing is if anything more
# volatile).
DEFAULT_SPOT_DISCOUNT = 0.65

# Illustrative baseline: ~1 preemption per 1000 GPU-hours run on spot
# capacity (i.e. an 8-GPU job sees roughly one preemption every ~125 wall
# hours). Real interruption rates vary widely by cloud, instance type,
# region, and time — expose as a parameter, don't hardcode it as fact.
DEFAULT_PREEMPTIONS_PER_1000_GPU_HOURS = 1.0


def spot_price_per_hr(on_demand_price_per_hr_usd: float, discount: float = DEFAULT_SPOT_DISCOUNT) -> float:
    if not (0.0 <= discount < 1.0):
        raise ValueError(f"discount must be in [0, 1); got {discount}")
    return on_demand_price_per_hr_usd * (1.0 - discount)


@dataclass(frozen=True)
class SpotRunEstimate:
    on_demand_cost_usd: float
    spot_price_per_hr_usd: float
    expected_preemptions: float
    expected_lost_steps: float
    expected_recovery_overhead_hours: float
    expected_wall_clock_hours: float
    expected_total_cost_usd: float
    savings_usd: float
    savings_pct: float


def estimate_spot_run(
    on_demand_price_per_hr_usd: float,
    total_gpus: int,
    base_time_hours: float,
    step_time_s: float,
    checkpoint_interval_steps: int,
    checkpoint_size_gb: float,
    preemptions_per_1000_gpu_hours: float = DEFAULT_PREEMPTIONS_PER_1000_GPU_HOURS,
    discount: float = DEFAULT_SPOT_DISCOUNT,
) -> SpotRunEstimate:
    """Expected cost/time running the same job on spot capacity instead of
    on-demand: the discounted hourly rate, offset by the expected extra wall
    time preemptions cost (checkpoint restore + redoing steps lost since the
    last checkpoint).

    A preemption is modeled as landing uniformly at random within a
    checkpoint interval, so on average it loses half an interval's worth of
    steps (`checkpoint_interval_steps / 2`) — the same reasoning that makes
    smaller checkpoint intervals reduce expected loss on real preemptible
    fleets, at the cost of more frequent checkpoint-write overhead (not
    charged here; see engine.checkpointing.checkpoint_write_seconds for
    that side of the tradeoff, already accounted for in a live run's own
    step time via api/routers/runs.py's checkpoint_interval_steps handling).
    """
    if total_gpus <= 0:
        raise ValueError(f"total_gpus must be positive; got {total_gpus}")
    if base_time_hours < 0:
        raise ValueError(f"base_time_hours must be non-negative; got {base_time_hours}")
    if checkpoint_interval_steps <= 0:
        raise ValueError(f"checkpoint_interval_steps must be positive; got {checkpoint_interval_steps}")

    spot_price = spot_price_per_hr(on_demand_price_per_hr_usd, discount)
    on_demand_cost = on_demand_price_per_hr_usd * total_gpus * base_time_hours

    gpu_hours = total_gpus * base_time_hours
    expected_preemptions = gpu_hours * preemptions_per_1000_gpu_hours / 1000.0

    avg_lost_steps_per_preemption = checkpoint_interval_steps / 2.0
    expected_lost_steps = expected_preemptions * avg_lost_steps_per_preemption

    recovery_s_per_event = recovery_overhead_seconds(
        checkpoint_size_gb, int(round(avg_lost_steps_per_preemption)), step_time_s
    )
    expected_recovery_overhead_hours = (expected_preemptions * recovery_s_per_event) / 3600.0

    expected_wall_clock_hours = base_time_hours + expected_recovery_overhead_hours
    expected_total_cost_usd = spot_price * total_gpus * expected_wall_clock_hours

    savings_usd = on_demand_cost - expected_total_cost_usd
    savings_pct = (savings_usd / on_demand_cost * 100.0) if on_demand_cost > 0 else 0.0

    return SpotRunEstimate(
        on_demand_cost_usd=on_demand_cost,
        spot_price_per_hr_usd=spot_price,
        expected_preemptions=expected_preemptions,
        expected_lost_steps=expected_lost_steps,
        expected_recovery_overhead_hours=expected_recovery_overhead_hours,
        expected_wall_clock_hours=expected_wall_clock_hours,
        expected_total_cost_usd=expected_total_cost_usd,
        savings_usd=savings_usd,
        savings_pct=savings_pct,
    )
