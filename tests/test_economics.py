from engine.economics import compute_run_economics


def test_cost_accumulates_with_price_and_gpu_count():
    econ = compute_run_economics(
        price_per_hr_usd=2.0,
        total_gpus=8,
        compute_s_per_step=3.0,
        communication_s_per_step=1.0,
        steps=[{"step": 1}, {"step": 2}, {"step": 3}],
    )
    assert len(econ.points) == 3
    assert econ.points[-1].cumulative_cost_usd == econ.summary.total_cost_usd
    assert econ.summary.total_cost_usd > 0
    # cost only grows step over step
    costs = [p.cumulative_cost_usd for p in econ.points]
    assert costs == sorted(costs)


def test_utilization_reflects_compute_share():
    econ = compute_run_economics(
        price_per_hr_usd=1.0, total_gpus=1, compute_s_per_step=9.0, communication_s_per_step=1.0, steps=[{"step": 1}]
    )
    assert econ.points[0].utilization_pct == 90.0
    assert econ.summary.idle_pct == 0.0
    assert econ.summary.idle_cost_usd == 0.0


def test_mostly_communication_bound_step_counts_as_idle():
    econ = compute_run_economics(
        price_per_hr_usd=1.0, total_gpus=1, compute_s_per_step=1.0, communication_s_per_step=9.0, steps=[{"step": 1}]
    )
    assert econ.points[0].utilization_pct == 10.0
    assert econ.summary.idle_pct == 100.0
    assert econ.summary.idle_cost_usd == econ.summary.total_cost_usd


def test_no_steps_yields_zero_cost():
    econ = compute_run_economics(
        price_per_hr_usd=2.0, total_gpus=8, compute_s_per_step=3.0, communication_s_per_step=1.0, steps=[]
    )
    assert econ.points == []
    assert econ.summary.total_cost_usd == 0.0
    assert econ.summary.total_gpu_hours == 0.0
