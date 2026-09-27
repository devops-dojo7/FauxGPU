import pytest

from engine.spot import (
    DEFAULT_PREEMPTIONS_PER_1000_GPU_HOURS,
    estimate_spot_run,
    spot_price_per_hr,
)


def test_spot_price_is_discounted():
    assert spot_price_per_hr(10.0, discount=0.65) == pytest.approx(3.5)


def test_spot_price_zero_discount_matches_on_demand():
    assert spot_price_per_hr(10.0, discount=0.0) == 10.0


def test_spot_price_rejects_invalid_discount():
    with pytest.raises(ValueError, match="discount"):
        spot_price_per_hr(10.0, discount=1.0)
    with pytest.raises(ValueError, match="discount"):
        spot_price_per_hr(10.0, discount=-0.1)


def test_no_preemptions_means_pure_discount_savings():
    result = estimate_spot_run(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=100.0,
        step_time_s=10.0,
        checkpoint_interval_steps=50,
        checkpoint_size_gb=20.0,
        preemptions_per_1000_gpu_hours=0.0,
        discount=0.65,
    )
    assert result.expected_preemptions == 0.0
    assert result.expected_wall_clock_hours == 100.0
    assert result.expected_total_cost_usd == pytest.approx(result.on_demand_cost_usd * 0.35)
    assert result.savings_pct == pytest.approx(65.0)


def test_more_preemptions_increase_expected_wall_clock_and_reduce_savings():
    low = estimate_spot_run(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=1000.0,
        step_time_s=10.0,
        checkpoint_interval_steps=50,
        checkpoint_size_gb=20.0,
        preemptions_per_1000_gpu_hours=1.0,
    )
    high = estimate_spot_run(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=1000.0,
        step_time_s=10.0,
        checkpoint_interval_steps=50,
        checkpoint_size_gb=20.0,
        preemptions_per_1000_gpu_hours=10.0,
    )
    assert high.expected_preemptions > low.expected_preemptions
    assert high.expected_wall_clock_hours > low.expected_wall_clock_hours
    assert high.savings_pct < low.savings_pct


def test_larger_checkpoint_interval_loses_more_steps_per_preemption():
    frequent = estimate_spot_run(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=1000.0,
        step_time_s=10.0,
        checkpoint_interval_steps=10,
        checkpoint_size_gb=20.0,
        preemptions_per_1000_gpu_hours=5.0,
    )
    rare = estimate_spot_run(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=1000.0,
        step_time_s=10.0,
        checkpoint_interval_steps=200,
        checkpoint_size_gb=20.0,
        preemptions_per_1000_gpu_hours=5.0,
    )
    assert frequent.expected_lost_steps < rare.expected_lost_steps
    assert frequent.expected_recovery_overhead_hours < rare.expected_recovery_overhead_hours


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="total_gpus"):
        estimate_spot_run(4.5, 0, 100.0, 10.0, 50, 20.0)
    with pytest.raises(ValueError, match="base_time_hours"):
        estimate_spot_run(4.5, 8, -1.0, 10.0, 50, 20.0)
    with pytest.raises(ValueError, match="checkpoint_interval_steps"):
        estimate_spot_run(4.5, 8, 100.0, 10.0, 0, 20.0)


def test_default_preemption_rate_is_a_named_constant():
    # Regression guard: the module's own documented default, not a magic
    # number duplicated at call sites.
    assert DEFAULT_PREEMPTIONS_PER_1000_GPU_HOURS > 0
