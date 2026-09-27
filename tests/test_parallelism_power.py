import pytest

from engine.gpu_specs import get_gpu
from engine.memory import ModelShape
from engine.parallelism import (
    estimate_parallel_step_time,
    expert_all_to_all_seconds,
    pipeline_bubble_fraction,
    tensor_parallel_communication_seconds,
)
from engine.power import decode_power_watts, power_watts, prefill_power_watts, training_step_power_watts
from engine.topology import build_topology

H100 = get_gpu("h100-sxm")
L40S = get_gpu("l40s")  # no NVLink
LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)

# Mixtral 8x7B's real published config (mistralai/Mixtral-8x7B-v0.1,
# huggingface.co config.json): 8 local experts, top-2 routing.
MIXTRAL_8X7B = ModelShape(
    params=46.7e9, active_params=12.9e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128,
    num_kv_heads=8, num_experts=8, top_k=2,
)


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


# --- Expert parallelism (GShard, Lepikhin et al. 2020; Switch Transformer,
# Fedus et al. 2022): MoE dispatch+combine all-to-all ---


def test_expert_all_to_all_is_zero_for_dense_models():
    # LLAMA2_7B has no num_experts/top_k set (dense) -- unaffected regardless of ep_degree.
    assert expert_all_to_all_seconds(LLAMA2_7B, tokens_per_step=4096, precision="bf16", ep_degree=8, bandwidth_gbps=900) == 0.0


def test_expert_all_to_all_is_zero_when_not_actually_sharded():
    # A real MoE model, but ep_degree<=1 means experts aren't split across ranks.
    assert expert_all_to_all_seconds(MIXTRAL_8X7B, tokens_per_step=4096, precision="bf16", ep_degree=1, bandwidth_gbps=900) == 0.0


def test_expert_all_to_all_matches_dispatch_plus_combine_formula():
    # Hand-derived: 2 collectives (dispatch+combine) x 2 passes (fwd+bwd) x
    # num_layers, each moving tokens_per_step * top_k * hidden_dim * bytes/param.
    tokens, precision, ep_degree, bw = 4096, "bf16", 8, 900.0
    actual = expert_all_to_all_seconds(MIXTRAL_8X7B, tokens, precision, ep_degree, bw)
    payload = tokens * MIXTRAL_8X7B.top_k * MIXTRAL_8X7B.hidden_dim * 2.0  # bf16 = 2 bytes/param
    total_collectives = MIXTRAL_8X7B.num_layers * 2 * 2
    bandwidth_bytes_per_sec = bw * 1e9 / 8
    expected = (payload * total_collectives) / bandwidth_bytes_per_sec
    assert actual == pytest.approx(expected)


def test_expert_all_to_all_scales_linearly_with_top_k():
    # Doubling top_k (more experts routed to per token) should double the
    # dispatch+combine payload exactly.
    top1_model = ModelShape(
        params=MIXTRAL_8X7B.params, active_params=MIXTRAL_8X7B.active_params, num_layers=MIXTRAL_8X7B.num_layers,
        hidden_dim=MIXTRAL_8X7B.hidden_dim, num_heads=MIXTRAL_8X7B.num_heads, head_dim=MIXTRAL_8X7B.head_dim,
        num_kv_heads=MIXTRAL_8X7B.num_kv_heads, num_experts=8, top_k=1,
    )
    top1 = expert_all_to_all_seconds(top1_model, 4096, "bf16", ep_degree=8, bandwidth_gbps=900)
    top2 = expert_all_to_all_seconds(MIXTRAL_8X7B, 4096, "bf16", ep_degree=8, bandwidth_gbps=900)  # top_k=2
    assert top2 == pytest.approx(top1 * 2)


def test_expert_all_to_all_without_nvlink_is_infeasible():
    assert (
        expert_all_to_all_seconds(MIXTRAL_8X7B, tokens_per_step=4096, precision="bf16", ep_degree=4, bandwidth_gbps=None)
        == float("inf")
    )


def test_estimate_parallel_step_time_wires_expert_communication():
    topo = build_topology("nvlink_node", "h100-sxm", gpus_per_node=8)
    ep1 = estimate_parallel_step_time(MIXTRAL_8X7B, topo, 32768, "bf16", 0.35, ep_degree=1)
    ep4 = estimate_parallel_step_time(MIXTRAL_8X7B, topo, 32768, "bf16", 0.35, ep_degree=4)
    assert ep1.expert_communication_s == 0.0
    assert ep4.expert_communication_s > 0.0
    assert ep4.total_gpus == topo.total_gpus * 4
    # Compute time is unaffected by EP -- it's a communication-only mechanism.
    assert ep4.compute_s == ep1.compute_s


def test_estimate_parallel_step_time_ep_is_noop_for_dense_models():
    topo = build_topology("nvlink_node", "h100-sxm", gpus_per_node=8)
    dense_ep1 = estimate_parallel_step_time(LLAMA2_7B, topo, 32768, "bf16", 0.35, ep_degree=1)
    dense_ep8 = estimate_parallel_step_time(LLAMA2_7B, topo, 32768, "bf16", 0.35, ep_degree=8)
    assert dense_ep1.total_s == dense_ep8.total_s
    # Regression test for a real bug found during expert-parallelism
    # validation: total_gpus was multiplied by ep_degree unconditionally,
    # so a dense model requesting ep_degree=8 silently got billed for 8x
    # the GPUs (and 8x the cost) for a mechanism that does nothing for it —
    # a fully "no-op" ep_degree must mean the GPU count doesn't move either,
    # not just that communication time stays at 0.
    assert dense_ep8.total_gpus == dense_ep1.total_gpus == topo.total_gpus
