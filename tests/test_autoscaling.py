import pytest

from engine.autoscaling import (
    AutoscalingConfig,
    TrafficStage,
    per_replica_capacity_rps,
    simulate_autoscaling,
)
from engine.gpu_specs import get_gpu
from engine.inference import simulate_serving
from engine.memory import ModelShape

LLAMA3_8B = ModelShape(params=8.03e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128, num_kv_heads=8)
H100 = get_gpu("h100-sxm")


def test_per_replica_capacity_matches_disaggregated_throughput_over_output_tokens():
    expected = simulate_serving(
        LLAMA3_8B, H100, "bf16", prompt_tokens=2048, output_tokens=256, decode_batch_size=8, requests_per_sec=0.0
    ).disaggregated_tokens_per_sec_per_gpu / 256
    actual = per_replica_capacity_rps(LLAMA3_8B, H100, "bf16", prompt_tokens=2048, output_tokens=256, decode_batch_size=8)
    assert actual == expected


def test_demand_interpolates_linearly_within_and_across_stages():
    # Stage 1: flat at 0 for 10s. Stage 2: ramps 0->20 over the next 10s. Stage 3: flat at 20.
    stages = [TrafficStage(10, 0), TrafficStage(10, 20), TrafficStage(10, 20)]
    config = AutoscalingConfig(min_replicas=1, max_replicas=100, eval_interval_s=1000, scale_down_stabilization_s=0)
    result = simulate_autoscaling(stages, config, per_replica_rps=1000, tick_s=5)
    demand_by_t = {round(p.t, 1): p.demand_rps for p in result.points}
    assert demand_by_t[0.0] == 0
    assert demand_by_t[5.0] == 0  # still flat within stage 1
    assert demand_by_t[15.0] == 10  # halfway through stage 2's ramp (0->20 over 10s)
    assert demand_by_t[20.0] == 20
    assert demand_by_t[25.0] == 20  # flat third stage


def test_desired_replicas_matches_ceil_formula_clamped_to_bounds():
    config = AutoscalingConfig(min_replicas=1, max_replicas=5, target_utilization_pct=100.0, eval_interval_s=1000, scale_down_stabilization_s=0)
    # demand=45, per_replica_rps=10 at 100% target -> ceil(45/10)=5, exactly at max
    stages = [TrafficStage(1, 45)]
    result = simulate_autoscaling(stages, config, per_replica_rps=10, tick_s=1)
    assert result.points[0].replicas == 5

    # demand tiny -> clamped to min_replicas, never 0
    stages = [TrafficStage(1, 0.001)]
    result = simulate_autoscaling(stages, config, per_replica_rps=10, tick_s=1)
    assert result.points[0].replicas == 1


def test_scale_up_is_immediate():
    # Flat low demand for 20s, a fast 1s ramp up to 100, then flat high demand.
    stages = [TrafficStage(20, 5), TrafficStage(1, 100), TrafficStage(79, 100)]
    config = AutoscalingConfig(min_replicas=1, max_replicas=20, target_utilization_pct=100.0, eval_interval_s=1, scale_down_stabilization_s=0)
    result = simulate_autoscaling(stages, config, per_replica_rps=10, tick_s=1)
    # the very next tick after demand finishes ramping to 100, replicas are already at ceil(100/10)=10 —
    # no extra delay beyond tracking the (fast) ramp itself.
    at_21 = next(p for p in result.points if p.t == 21.0)
    assert at_21.replicas == 10


def test_scale_down_is_held_until_stabilization_window_elapses():
    # Demand flat at 100 for 10s, then an instant drop to 0 held flat for 200s.
    stages = [TrafficStage(10, 100), TrafficStage(0, 0), TrafficStage(200, 0)]
    config = AutoscalingConfig(min_replicas=1, max_replicas=20, target_utilization_pct=100.0, eval_interval_s=10, scale_down_stabilization_s=60)
    result = simulate_autoscaling(stages, config, per_replica_rps=10, tick_s=10)

    just_after_drop = next(p for p in result.points if p.t == 20.0)
    assert just_after_drop.replicas == 10  # still held near peak, well within the 60s window

    well_after_window = next(p for p in result.points if p.t == 80.0)
    assert well_after_window.replicas == 1  # stabilization window (60s after t=10's peak eval) has elapsed


def test_backlog_accumulates_when_demand_exceeds_capacity_and_drains_after():
    # min_replicas=max_replicas=1 caps capacity at per_replica_rps regardless of demand.
    # 20s of overload (demand 50 vs capacity 10), then an instant drop to 0 held for 100s
    # — plenty of time for the backlog built up during overload to fully drain.
    stages = [TrafficStage(20, 50), TrafficStage(0, 0), TrafficStage(100, 0)]
    config = AutoscalingConfig(min_replicas=1, max_replicas=1, eval_interval_s=1000, scale_down_stabilization_s=0)
    result = simulate_autoscaling(stages, config, per_replica_rps=10, tick_s=5)

    mid_overload = next(p for p in result.points if p.t == 15.0)
    assert mid_overload.backlog_requests > 0

    end = result.points[-1]
    assert end.backlog_requests == 0  # fully drained once demand dropped to 0 and capacity had time to catch up


def test_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        simulate_autoscaling([TrafficStage(1, 1)], AutoscalingConfig(min_replicas=0, max_replicas=1), per_replica_rps=1)
    with pytest.raises(ValueError):
        simulate_autoscaling([TrafficStage(1, 1)], AutoscalingConfig(min_replicas=5, max_replicas=1), per_replica_rps=1)
    with pytest.raises(ValueError):
        simulate_autoscaling([TrafficStage(1, 1)], AutoscalingConfig(min_replicas=1, max_replicas=1), per_replica_rps=0)
