"""HPA/KEDA-style autoscaling simulation for inference-serving replicas.

A fixed per-replica throughput (from engine.inference.simulate_serving's
disaggregated serving mode — the scalable-replica baseline) is checked
against a synthetic traffic curve shaped like a k6 ramping-vus load test
(scripts/load-test.js's own api_concurrency scenario), and a discrete-time
control loop mirrors Kubernetes HorizontalPodAutoscaler v2's real algorithm:
immediate scale-up, scale-down held to the max recently-desired replica
count until a stabilization window elapses (this is HPA's actual
anti-flapping mechanism, not a simplification of it).

Scope, deliberately bounded:
- Capacity is a fixed per-replica requests/sec figure derived from
  disaggregated decode throughput / output_tokens. It does not additionally
  model engine.inference.estimate_serving_capacity's VRAM-derived
  max_concurrent_sequences as a second, independent concurrency ceiling.
- backlog/est_queue_delay_s is a simplified Little's-law-style
  approximation (queue_delay = backlog / current_capacity), not a full
  queueing-theory (M/M/c) simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.gpu_specs import GpuSpec
from engine.inference import simulate_serving
from engine.memory import ModelShape


@dataclass(frozen=True)
class TrafficStage:
    duration_s: float
    target_rps: float  # linear ramp from the previous stage's target_rps to this one, over duration_s


@dataclass(frozen=True)
class AutoscalingConfig:
    min_replicas: int
    max_replicas: int
    target_utilization_pct: float = 70.0
    eval_interval_s: float = 15.0  # HPA default --horizontal-pod-autoscaler-sync-period
    scale_down_stabilization_s: float = 300.0  # HPA v2 default scaleDown stabilizationWindowSeconds
    # Scale-up is immediate — HPA v2's default scaleUp policy has no stabilization window.


@dataclass(frozen=True)
class AutoscalingPoint:
    t: float
    demand_rps: float
    replicas: int
    capacity_rps: float
    backlog_requests: float
    est_queue_delay_s: float


@dataclass(frozen=True)
class AutoscalingResult:
    points: tuple[AutoscalingPoint, ...]
    per_replica_capacity_rps: float
    peak_replicas: int
    peak_backlog_requests: float
    peak_queue_delay_s: float


def per_replica_capacity_rps(
    model: ModelShape,
    gpu: GpuSpec,
    precision: str,
    prompt_tokens: int,
    output_tokens: int,
    decode_batch_size: int,
    cache_hit_fraction: float = 0.0,
    utilization: float = 0.35,
    paged_attention: bool = False,
    block_size: int = 16,
) -> float:
    result = simulate_serving(
        model, gpu, precision, prompt_tokens, output_tokens, decode_batch_size,
        requests_per_sec=0.0,  # disaggregated throughput is load-independent; value here is irrelevant
        cache_hit_fraction=cache_hit_fraction, utilization=utilization,
        paged_attention=paged_attention, block_size=block_size,
    )
    if output_tokens <= 0:
        raise ValueError("output_tokens must be > 0")
    return result.disaggregated_tokens_per_sec_per_gpu / output_tokens


def _demand_at(stages: list[TrafficStage], t: float) -> float:
    """Piecewise-linear interpolation across stages, k6 ramping-vus style:
    demand starts at the first stage's target and ramps toward each
    subsequent stage's target over that stage's duration.
    """
    if not stages:
        return 0.0
    elapsed = 0.0
    prev_target = stages[0].target_rps
    for stage in stages:
        if t <= elapsed + stage.duration_s:
            if stage.duration_s <= 0:
                return stage.target_rps
            frac = (t - elapsed) / stage.duration_s
            return prev_target + (stage.target_rps - prev_target) * frac
        elapsed += stage.duration_s
        prev_target = stage.target_rps
    return stages[-1].target_rps


def simulate_autoscaling(
    stages: list[TrafficStage],
    config: AutoscalingConfig,
    per_replica_rps: float,
    tick_s: float = 5.0,
) -> AutoscalingResult:
    if config.min_replicas < 1:
        raise ValueError("min_replicas must be at least 1")
    if config.max_replicas < config.min_replicas:
        raise ValueError("max_replicas must be >= min_replicas")
    if per_replica_rps <= 0:
        raise ValueError("per_replica_rps must be > 0")
    if tick_s <= 0:
        raise ValueError("tick_s must be > 0")

    total_duration = sum(s.duration_s for s in stages)
    target_capacity_per_replica = per_replica_rps * (config.target_utilization_pct / 100)

    replicas = config.min_replicas
    backlog = 0.0
    recent_desired: list[tuple[float, int]] = []  # (t, desired) within the trailing stabilization window
    last_eval_t = -math.inf
    points: list[AutoscalingPoint] = []

    t = 0.0
    while t <= total_duration + 1e-9:
        demand = _demand_at(stages, t)

        if t - last_eval_t >= config.eval_interval_s or last_eval_t == -math.inf:
            desired = max(config.min_replicas, min(config.max_replicas, math.ceil(demand / target_capacity_per_replica)))
            recent_desired.append((t, desired))
            recent_desired = [(rt, rd) for rt, rd in recent_desired if t - rt <= config.scale_down_stabilization_s]

            if desired > replicas:
                replicas = desired  # immediate scale-up
            else:
                held_floor = max(rd for _, rd in recent_desired)
                if held_floor < replicas:
                    replicas = held_floor  # scale down only once nothing in the window still wants more
            last_eval_t = t

        capacity = replicas * per_replica_rps
        if demand > capacity:
            backlog += (demand - capacity) * tick_s
        else:
            backlog = max(0.0, backlog - (capacity - demand) * tick_s)
        queue_delay = backlog / capacity if capacity > 0 else 0.0

        points.append(AutoscalingPoint(t=t, demand_rps=demand, replicas=replicas, capacity_rps=capacity, backlog_requests=backlog, est_queue_delay_s=queue_delay))
        t += tick_s

    return AutoscalingResult(
        points=tuple(points),
        per_replica_capacity_rps=per_replica_rps,
        peak_replicas=max((p.replicas for p in points), default=config.min_replicas),
        peak_backlog_requests=max((p.backlog_requests for p in points), default=0.0),
        peak_queue_delay_s=max((p.est_queue_delay_s for p in points), default=0.0),
    )
