import pytest

from engine.gpu_specs import get_gpu
from engine.memory import ModelShape
from engine.parallelism import (
    estimate_parallel_step_time,
    pipeline_bubble_fraction,
    tensor_parallel_communication_seconds,
)
from engine.power import decode_power_watts, power_watts, prefill_power_watts, training_step_power_watts
from engine.topology import build_topology

H100 = get_gpu("h100-sxm")
L40S = get_gpu("l40s")  # no NVLink
LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)


def test_tensor_parallel_reduces_compute_time():
    topo = build_topology("nvlink_node", "h100-sxm", gpus_per_node=8)
    tp1 = estimate_parallel_step_time(LLAMA2_7B, topo, 32768, "bf16", 0.35, tp_degree=1)
    tp4 = estimate_parallel_step_time(LLAMA2_7B, topo, 32768, "bf16", 0.35, tp_degree=4)
    assert tp4.compute_s == pytest.approx(tp1.compute_s / 4)
    assert tp4.tp_communication_s > 0
    assert tp4.total_gpus == topo.total_gpus * 4


def test_tensor_parallel_without_nvlink_is_infeasible():
    topo = build_topology("single_gpu", "l40s")
    result = estimate_parallel_step_time(LLAMA2_7B, topo, 32768, "bf16", 0.35, tp_degree=4)
    assert result.tp_communication_s == float("inf")


def test_pipeline_bubble_shrinks_with_more_microbatches():
    few = pipeline_bubble_fraction(pp_degree=4, num_microbatches=4)
    many = pipeline_bubble_fraction(pp_degree=4, num_microbatches=64)
    assert few > many > 0


def test_pipeline_bubble_zero_for_pp1():
    assert pipeline_bubble_fraction(pp_degree=1, num_microbatches=8) == 0.0


def test_power_watts_interpolates_between_idle_and_tdp():
    assert power_watts(H100, 0.0) == H100.idle_watts
    assert power_watts(H100, 1.0) == H100.tdp_watts
    mid = power_watts(H100, 0.5)
    assert H100.idle_watts < mid < H100.tdp_watts


def test_decode_power_less_than_full_compute_power():
    assert decode_power_watts(H100) < prefill_power_watts(H100, utilization=1.0)


def test_training_step_power_weighted_by_phase_duration():
    # All-compute step should draw close to compute-phase power.
    compute_only = training_step_power_watts(H100, compute_s=10, communication_s=0, utilization=0.35)
    assert compute_only == pytest.approx(power_watts(H100, 0.35))
    # All-communication step should draw close to the lower comm-phase power.
    comm_only = training_step_power_watts(H100, compute_s=0, communication_s=10, utilization=0.35)
    assert comm_only < compute_only
