from engine.compute import estimate_step_time
from engine.cost import estimate_training_cost
from engine.memory import ModelShape
from engine.topology import build_topology

LLAMA2_7B = ModelShape(
    params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128
)


def test_single_gpu_has_no_communication_overhead():
    topo = build_topology("single_gpu", gpu_id="h100-sxm")
    step = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=4096)
    assert step.communication_s == 0.0
    assert step.compute_s > 0.0


def test_multi_node_is_slower_per_step_than_single_node_nvlink_at_same_gpu_count():
    nvlink_node = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    multi_node = build_topology(
        "multi_node",
        gpu_id="h100-sxm",
        gpus_per_node=1,
        num_nodes=8,
        fabric_id="infiniband-hdr",
    )
    assert nvlink_node.total_gpus == multi_node.total_gpus == 8

    nvlink_step = estimate_step_time(LLAMA2_7B, nvlink_node, tokens_per_step=32768)
    multi_step = estimate_step_time(LLAMA2_7B, multi_node, tokens_per_step=32768)

    # InfiniBand is far slower than NVLink, so the same collective costs more time.
    assert multi_step.communication_s > nvlink_step.communication_s


def test_gpu_without_nvlink_rejects_nvlink_node_topology():
    import pytest

    with pytest.raises(ValueError):
        build_topology("nvlink_node", gpu_id="l40s", gpus_per_node=8)


def test_cost_scales_with_gpu_count_and_price():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    step = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768)
    cost = estimate_training_cost(
        topo, step, tokens_per_step=32768, total_training_tokens=1e9
    )
    assert cost.total_cost_usd > 0
    assert cost.total_steps == -(-10**9 // 32768)  # ceil(1e9 / 32768)
    assert cost.cost_per_1k_tokens_usd > 0
