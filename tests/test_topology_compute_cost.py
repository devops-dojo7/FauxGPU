import pytest

from engine.compute import estimate_step_time
from engine.cost import estimate_training_cost
from engine.memory import ModelShape, lora_trainable_params
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


def test_zero_stage3_increases_communication_time_by_1_5x():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768)
    zero3 = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, zero_stage=3)
    assert zero3.communication_s == pytest.approx(baseline.communication_s * 1.5)
    # Compute time is unaffected by ZeRO's communication-volume change.
    assert zero3.compute_s == baseline.compute_s


def test_zero_stage1_and_2_do_not_change_communication_time():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768)
    for stage in (0, 1, 2):
        step = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, zero_stage=stage)
        assert step.communication_s == baseline.communication_s


def test_zero_communication_multiplier_rejects_invalid_stage():
    from engine.compute import zero_communication_multiplier

    with pytest.raises(ValueError, match="zero_stage"):
        zero_communication_multiplier(5)


# --- LoRA / QLoRA step-time (gradient all-reduce shrinks to the adapter) ---


def test_lora_shrinks_communication_time_to_adapter_gradients_only():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768)
    lora = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="lora", peft_rank=8)
    assert lora.communication_s < baseline.communication_s
    # Compute time (forward+backward FLOPs) is unaffected by LoRA — only
    # fewer gradients need synchronizing after the backward pass.
    assert lora.compute_s == baseline.compute_s


def test_lora_communication_time_matches_adapter_param_ratio():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, precision="bf16")
    lora = estimate_step_time(
        LLAMA2_7B, topo, tokens_per_step=32768, precision="bf16", peft_method="lora", peft_rank=8, peft_target_modules=2
    )
    adapter_params = lora_trainable_params(LLAMA2_7B, rank=8, target_modules=2)
    expected_ratio = adapter_params / LLAMA2_7B.params
    assert lora.communication_s == pytest.approx(baseline.communication_s * expected_ratio)


def test_qlora_communication_time_matches_lora_exactly():
    # QLoRA's communication savings come from the same tiny trainable
    # adapter as LoRA (only the base weights' storage format differs,
    # which doesn't affect gradient communication volume).
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    lora = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="lora", peft_rank=8)
    qlora = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="qlora", peft_rank=8)
    assert lora.communication_s == qlora.communication_s


def test_peft_and_zero_stage_are_mutually_exclusive_in_step_time():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    with pytest.raises(ValueError, match="zero_stage"):
        estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="lora", zero_stage=1)
