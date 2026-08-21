import time

import pytest

from api.live_phase import instantaneous_power_watts, phase_fraction
from api.runs_store import RunsStore
from engine.gpu_specs import get_gpu

H100 = get_gpu("h100-sxm")


def test_phase_fraction_none_without_any_steps():
    store = RunsStore()
    run = store.start("run-1", {"compute_s_per_step": 0.5, "communication_s_per_step": 0.2})
    assert phase_fraction(run) is None


def test_phase_fraction_defined_once_a_step_is_reported():
    store = RunsStore()
    run = store.start("run-1", {"compute_s_per_step": 0.5, "communication_s_per_step": 0.2})
    store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 1.0})
    run = store.get("run-1")
    frac = phase_fraction(run)
    assert frac is not None
    assert 0.0 <= frac < 1.0


def test_instantaneous_power_falls_back_to_average_without_steps():
    store = RunsStore()
    run = store.start("run-1", {"compute_s_per_step": 1.0, "communication_s_per_step": 0.0})
    power, in_compute = instantaneous_power_watts(run, H100, utilization=0.35)
    assert in_compute is True
    assert power > H100.idle_watts


def test_instantaneous_power_reflects_current_phase_just_after_a_step():
    # Immediately after add_step, elapsed time since start is ~0, so we're
    # right at the beginning of the "current" step — should read as the
    # compute phase for a step that starts with compute time > 0.
    store = RunsStore()
    run = store.start("run-1", {"compute_s_per_step": 0.5, "communication_s_per_step": 0.2})
    store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 5.0})
    run = store.get("run-1")
    power, in_compute = instantaneous_power_watts(run, H100, utilization=0.35)
    assert in_compute is True
    assert power == pytest.approx(H100.idle_watts + 0.35 * (H100.tdp_watts - H100.idle_watts))


def test_communication_phase_draws_less_power_than_compute_phase():
    from engine.power import power_watts, COMMUNICATION_POWER_FRACTION

    compute_power = power_watts(H100, 0.35)
    comm_power = power_watts(H100, COMMUNICATION_POWER_FRACTION)
    assert comm_power < compute_power
