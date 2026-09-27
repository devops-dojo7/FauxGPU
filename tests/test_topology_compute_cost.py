import pytest

from engine.compute import LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION, estimate_step_time, lora_flops_multiplier
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
    # Compute time also shrinks — LoRA skips the backward weight-gradient
    # FLOPs for frozen parameters (see engine.compute.lora_flops_multiplier),
    # a real, separately-documented effect distinct from the communication
    # saving above (not zero, and not equal to the communication ratio).
    assert lora.compute_s < baseline.compute_s


def test_lora_compute_time_matches_flops_multiplier_exactly():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768)
    lora = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="lora", peft_rank=8, peft_target_modules=2)
    multiplier = lora_flops_multiplier(LLAMA2_7B, peft_rank=8, peft_target_modules=2)
    assert lora.compute_s == pytest.approx(baseline.compute_s * multiplier)


def test_lora_flops_multiplier_approaches_two_thirds_for_large_models():
    # In the limit where the adapter is vanishingly small next to the base
    # model, LoRA skips almost the entire weight-gradient FLOPs term (2/6
    # of the "6N" approximation), landing near the ~30% FLOPs reduction
    # widely cited in the literature (e.g. Thinking Machines' "LoRA
    # without Regret", 2025) — i.e. a multiplier approaching 4/6 = 2/3.
    huge = ModelShape(params=175e9, num_layers=96, hidden_dim=12288, num_heads=96, head_dim=128)
    multiplier = lora_flops_multiplier(huge, peft_rank=8, peft_target_modules=2)
    assert multiplier == pytest.approx(2.0 / 3.0, rel=0.01)


def test_lora_flops_multiplier_accounts_for_the_adapters_own_small_share():
    # Unlike a limit approximation, this model computes the (small but
    # nonzero) FLOPs the adapter itself still needs for its own weight
    # gradient — so the multiplier is always >= 2/3, never falling below
    # it even for a deliberately tiny model where the adapter isn't
    # actually negligible relative to the base.
    tiny = ModelShape(params=1000, num_layers=1, hidden_dim=8, num_heads=1, head_dim=8)
    multiplier = lora_flops_multiplier(tiny, peft_rank=8, peft_target_modules=2)
    assert multiplier >= 2.0 / 3.0


def test_lora_flops_multiplier_uses_active_params_for_moe_models():
    # LoRA only ever adapts attention projections (Wq/Wk/Wv/Wo), which are
    # dense (not routed) in essentially every MoE architecture this project
    # models — so the frozen-share denominator should be
    # effective_active_params (matching flops_per_step's own per-token
    # compute base), not the full (mostly-inactive) resident param count.
    # A real catalog shape (Mixtral 8x7B: 46.7B total, 12.9B active).
    mixtral = ModelShape(
        params=46.7e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128, active_params=12.9e9
    )
    multiplier = lora_flops_multiplier(mixtral, peft_rank=8, peft_target_modules=2)
    adapter_params = lora_trainable_params(mixtral, rank=8, target_modules=2)
    expected = 1.0 - LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION * (1.0 - adapter_params / mixtral.active_params)
    assert multiplier == pytest.approx(expected)
    # Sanity: using the (much larger) total param count instead would give
    # a materially different, wrong answer -- this pins the denominator choice.
    wrong_denominator = 1.0 - LORA_FROZEN_WEIGHT_GRAD_FLOPS_FRACTION * (1.0 - adapter_params / mixtral.params)
    assert multiplier != pytest.approx(wrong_denominator)


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
    # QLoRA's frozen-base quantization doesn't change which FLOPs are
    # skipped either (same frozen/trainable split as LoRA).
    assert lora.compute_s == qlora.compute_s


def test_peft_and_zero_stage_are_mutually_exclusive_in_step_time():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    with pytest.raises(ValueError, match="zero_stage"):
        estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768, peft_method="lora", zero_stage=1)
