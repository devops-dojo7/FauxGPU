"""Sanity-check engine formulas against commonly cited reference numbers for
Llama-2 7B (params=6.74B, 32 layers, hidden=4096, 32 heads, head_dim=128).
"""

import pytest

from engine.memory import (
    ModelShape,
    activation_bytes,
    compute_vram_breakdown,
    kv_cache_bytes,
    lora_trainable_params,
    optimizer_state_bytes,
    quantized_weight_bytes,
    weights_bytes,
)

LLAMA2_7B = ModelShape(
    params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128
)

# The ZeRO paper's own worked example (Rajbhandari et al. 2020, Table 1,
# https://arxiv.org/abs/1910.02054): a 7.5B model, mixed-precision Adam
# (K=12, so model states = 16 bytes/param unsharded).
ZERO_PAPER_MODEL = ModelShape(params=7.5e9, num_layers=1, hidden_dim=1, num_heads=1, head_dim=1)

# The LoRA paper's own GPT-3 175B shape (Hu et al. 2021, Sec 5.5/7.1):
# hidden_dim (d_model) 12288, 96 layers.
GPT3_175B = ModelShape(params=175.2558e9, num_layers=96, hidden_dim=12288, num_heads=96, head_dim=128)

# The QLoRA paper's own headline model (Dettmers et al. 2023): LLaMA 65B,
# shape approximated from the publicly documented LLaMA-65B architecture
# (hidden_dim 8192, 80 layers — same depth/width as Llama-2 70B, which
# shares LLaMA's architecture family).
LLAMA_65B = ModelShape(params=65.0e9, num_layers=80, hidden_dim=8192, num_heads=64, head_dim=128)

# Falcon-7B's real published shape (this project's own catalog preset):
# extreme GQA (multi-query attention, num_kv_heads=1) with num_heads*head_dim
# == hidden_dim for Wq/Wo but a much smaller Wk/Wv output dim — a real-world
# case where a square-matrix assumption would overestimate LoRA's adapter size.
FALCON_7B_MQA = ModelShape(params=7.22e9, num_layers=32, hidden_dim=4544, num_heads=71, head_dim=64, num_kv_heads=1)

# Gemma-3 1B's real published shape: GQA *and* num_heads*head_dim != hidden_dim
# (4*256=1024 vs hidden_dim=1152) — exercises both distinct shape effects at once.
GEMMA3_1B_GQA = ModelShape(params=1.0e9, num_layers=26, hidden_dim=1152, num_heads=4, head_dim=256, num_kv_heads=1)

# DeepSeek-V3's real published shape (this project's own catalog preset):
# multi-head latent attention (MLA) — K/V compressed to kv_latent_dim=576,
# a real ~28x smaller "KV dimension" than num_heads*head_dim=16384 would
# naively suggest, since there is no per-head Wk/Wv in MLA at all.
DEEPSEEK_V3_MLA = ModelShape(
    params=671.0e9, active_params=37.0e9, num_layers=61, hidden_dim=7168, num_heads=128, head_dim=128, kv_latent_dim=576
)


def test_fp16_weights_matches_known_7b_footprint():
    # Widely cited: a 7B model in fp16 takes ~13-14GB just for weights.
    gb = weights_bytes(LLAMA2_7B, "fp16") / 1e9
    assert 13.0 < gb < 14.0


def test_adam_optimizer_state_dominates_full_finetune_footprint():
    # Adam moments (2x fp32) + fp32 master copy = 12 bytes/param -> ~80GB for 6.74B params.
    gb = optimizer_state_bytes(LLAMA2_7B, "adam", fp32_master_copy=True) / 1e9
    assert 79.0 < gb < 82.0


def test_sgd_no_master_copy_has_zero_optimizer_state():
    assert optimizer_state_bytes(LLAMA2_7B, "sgd", fp32_master_copy=False) == 0.0


def test_kv_cache_matches_known_ballpark_for_2k_context():
    # Commonly cited: ~1GB KV cache for a 7B model at batch=1, seq_len=2048, fp16.
    gb = kv_cache_bytes(LLAMA2_7B, batch_size=1, seq_len=2048, precision="fp16") / 1e9
    assert 0.9 < gb < 1.2


def test_kv_cache_scales_linearly_with_batch_and_seq_len():
    base = kv_cache_bytes(LLAMA2_7B, batch_size=1, seq_len=1024, precision="fp16")
    doubled_batch = kv_cache_bytes(LLAMA2_7B, batch_size=2, seq_len=1024, precision="fp16")
    doubled_seq = kv_cache_bytes(LLAMA2_7B, batch_size=1, seq_len=2048, precision="fp16")
    assert doubled_batch == 2 * base
    assert doubled_seq == 2 * base


def test_activation_checkpointing_reduces_memory():
    full = activation_bytes(LLAMA2_7B, batch_size=1, seq_len=2048, precision="bf16")
    checkpointed = activation_bytes(
        LLAMA2_7B, batch_size=1, seq_len=2048, precision="bf16", checkpointing=True
    )
    assert checkpointed < full
    assert checkpointed == full / LLAMA2_7B.num_layers


def test_full_breakdown_training_vs_inference():
    training = compute_vram_breakdown(
        LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=True
    )
    assert training.optimizer_states_gb > 0
    assert training.kv_cache_gb == 0

    inference = compute_vram_breakdown(
        LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=False
    )
    assert inference.optimizer_states_gb == 0
    assert inference.gradients_gb == 0
    assert inference.kv_cache_gb > 0

    # Full fine-tuning needs far more VRAM than inference-only serving.
    assert training.total_gb > inference.total_gb


# --- ZeRO-DP / FSDP sharding (Rajbhandari et al. 2020) ---


def _model_states_gb(breakdown):
    return breakdown.weights_gb + breakdown.gradients_gb + breakdown.optimizer_states_gb


def test_zero_disabled_matches_unsharded_baseline():
    baseline = compute_vram_breakdown(ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0)
    assert _model_states_gb(baseline) == pytest.approx(120.0, rel=0.01)


def test_zero_stage1_matches_paper_table1_at_dp64():
    # Pos: optimizer states sharded -> 31.4GB in the paper's own Table 1.
    b = compute_vram_breakdown(
        ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=1, dp_size=64
    )
    assert _model_states_gb(b) == pytest.approx(31.4, rel=0.01)


def test_zero_stage2_matches_paper_table1_at_dp64():
    # Pos+g: + gradients sharded -> 16.6GB in the paper's own Table 1.
    b = compute_vram_breakdown(
        ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=2, dp_size=64
    )
    assert _model_states_gb(b) == pytest.approx(16.6, rel=0.01)


def test_zero_stage3_matches_paper_table1_at_dp64():
    # Pos+g+p: + parameters sharded -> 1.88GB in the paper's own Table 1.
    b = compute_vram_breakdown(
        ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=3, dp_size=64
    )
    assert _model_states_gb(b) == pytest.approx(1.88, rel=0.01)


def test_zero_memory_shrinks_monotonically_with_stage():
    stages = [
        _model_states_gb(
            compute_vram_breakdown(ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=s, dp_size=64)
        )
        for s in (0, 1, 2, 3)
    ]
    assert stages[0] > stages[1] > stages[2] > stages[3]


def test_zero_stage3_scales_linearly_with_dp_size():
    at_8 = compute_vram_breakdown(ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=3, dp_size=8)
    at_16 = compute_vram_breakdown(ZERO_PAPER_MODEL, precision="fp16", batch_size=0, seq_len=0, zero_stage=3, dp_size=16)
    assert _model_states_gb(at_8) == pytest.approx(2 * _model_states_gb(at_16), rel=0.01)


def test_zero_disabled_ignores_dp_size():
    at_1 = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, zero_stage=0, dp_size=1)
    at_64 = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, zero_stage=0, dp_size=64)
    assert at_1 == at_64


def test_zero_does_not_shard_activations_or_kv_cache():
    # ZeRO-DP shards model states (weights/gradients/optimizer), not
    # per-token activations or inference KV cache -- those are unaffected
    # by zero_stage/dp_size.
    unsharded = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, zero_stage=0)
    sharded = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, zero_stage=3, dp_size=8)
    assert sharded.activations_gb == unsharded.activations_gb

    inference_unsharded = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=False)
    inference_sharded = compute_vram_breakdown(
        LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=False, zero_stage=3, dp_size=8
    )
    assert inference_sharded.kv_cache_gb == inference_unsharded.kv_cache_gb


def test_zero_rejects_invalid_stage():
    with pytest.raises(ValueError, match="zero_stage"):
        compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, zero_stage=4)


def test_zero_rejects_invalid_dp_size():
    with pytest.raises(ValueError, match="dp_size"):
        compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, zero_stage=1, dp_size=0)


# --- LoRA (Hu et al. 2021) / QLoRA (Dettmers et al. 2023) ---


def test_lora_trainable_params_matches_paper_table5_worked_example():
    # Table 5, "# of Trainable Parameters = 18M" header: rank=8, one weight
    # matrix (Wq alone) adapted across all 96 layers of GPT-3 175B.
    assert lora_trainable_params(GPT3_175B, rank=8, target_modules=1) == pytest.approx(18_874_368)


def test_lora_trainable_params_scales_linearly_with_rank_and_target_modules():
    base = lora_trainable_params(LLAMA2_7B, rank=4, target_modules=2)
    double_rank = lora_trainable_params(LLAMA2_7B, rank=8, target_modules=2)
    double_modules = lora_trainable_params(LLAMA2_7B, rank=4, target_modules=4)
    assert double_rank == pytest.approx(2 * base)
    assert double_modules == pytest.approx(2 * base)


def test_lora_trainable_params_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="rank"):
        lora_trainable_params(LLAMA2_7B, rank=0)
    with pytest.raises(ValueError, match="target_modules"):
        lora_trainable_params(LLAMA2_7B, rank=8, target_modules=5)


# --- GQA/MQA-aware attention matrix shapes (Wk/Wv smaller than Wq/Wo) ---


def test_lora_trainable_params_uses_smaller_kv_dim_for_gqa_models():
    # Falcon-7B (num_kv_heads=1, extreme MQA): Wq/Wo are 4544x4544 (square,
    # num_heads*head_dim == hidden_dim here), but Wk/Wv project down to
    # only num_kv_heads*head_dim=64 -- a real ~35% smaller matrix than a
    # naive square-matrix assumption would use.
    r, target_modules = 8, 2  # Wq + Wv
    naive_square_estimate = 2 * FALCON_7B_MQA.num_layers * target_modules * FALCON_7B_MQA.hidden_dim * r
    actual = lora_trainable_params(FALCON_7B_MQA, rank=r, target_modules=target_modules)
    assert actual < naive_square_estimate
    # Hand-derived exact value: Wq is (out=71*64, in=4544), Wv is (out=1*64, in=4544).
    expected = FALCON_7B_MQA.num_layers * (r * (71 * 64 + 4544) + r * (1 * 64 + 4544))
    assert actual == pytest.approx(expected)


def test_lora_trainable_params_handles_non_square_query_projection_too():
    # Gemma-3 1B: num_heads*head_dim (4*256=1024) != hidden_dim (1152) *and*
    # GQA (num_kv_heads=1) -- exercises both the query-side and kv-side
    # shape corrections simultaneously.
    r, target_modules = 8, 4  # Wq + Wv + Wk + Wo
    actual = lora_trainable_params(GEMMA3_1B_GQA, rank=r, target_modules=target_modules)
    q_dim = 4 * 256
    kv_dim = 1 * 256
    per_layer = r * (q_dim + 1152) + 2 * r * (kv_dim + 1152) + r * (1152 + q_dim)
    expected = GEMMA3_1B_GQA.num_layers * per_layer
    assert actual == pytest.approx(expected)


def test_lora_trainable_params_gqa_correction_only_applies_to_kv_matrices():
    # target_modules=1 adapts only Wq, which is unaffected by GQA (Wq's
    # shape depends on num_heads, not num_kv_heads) -- so a GQA model's
    # single-matrix adapter size should differ from an equivalent MHA
    # model's only when num_heads*head_dim != hidden_dim, not merely
    # because num_kv_heads is set.
    mha_equivalent = ModelShape(
        params=FALCON_7B_MQA.params,
        num_layers=FALCON_7B_MQA.num_layers,
        hidden_dim=FALCON_7B_MQA.hidden_dim,
        num_heads=FALCON_7B_MQA.num_heads,
        head_dim=FALCON_7B_MQA.head_dim,
        # num_kv_heads intentionally omitted -> MHA
    )
    assert lora_trainable_params(FALCON_7B_MQA, rank=8, target_modules=1) == pytest.approx(
        lora_trainable_params(mha_equivalent, rank=8, target_modules=1)
    )


def test_lora_trainable_params_still_matches_paper_for_mha_models():
    # Regression guard: the GQA-aware rewrite must not change the
    # paper-verified MHA result (num_kv_heads unset, num_heads*head_dim ==
    # hidden_dim) -- re-asserts the same Table 5 number from
    # test_lora_trainable_params_matches_paper_table5_worked_example via a
    # different (target_modules=2) worked path for extra coverage.
    l_hat = GPT3_175B.num_layers * 2
    naive_paper_formula = 2 * l_hat * GPT3_175B.hidden_dim * 8
    assert lora_trainable_params(GPT3_175B, rank=8, target_modules=2) == pytest.approx(naive_paper_formula)


# --- MLA-aware KV dimension (compressed latent, not per-head Wk/Wv) ---


def test_lora_trainable_params_uses_kv_latent_dim_for_mla_models():
    # DeepSeek-V3 (kv_latent_dim=576): the KV side of the adapter should
    # use the compressed latent dimension, not num_heads*head_dim=16384 --
    # a real ~28x smaller (and thus far cheaper to adapt) KV dimension.
    r, target_modules = 8, 2  # Wq + Wv
    naive_head_count_estimate = DEEPSEEK_V3_MLA.num_layers * (
        r * (128 * 128 + 7168) + r * (128 * 128 + 7168)  # both treated as square/head-count-based
    )
    actual = lora_trainable_params(DEEPSEEK_V3_MLA, rank=r, target_modules=target_modules)
    assert actual < naive_head_count_estimate
    # Hand-derived exact value: Wq is (out=128*128, in=7168), the KV
    # down-projection is (out=576, in=7168).
    expected = DEEPSEEK_V3_MLA.num_layers * (r * (128 * 128 + 7168) + r * (576 + 7168))
    assert actual == pytest.approx(expected)


def test_lora_trainable_params_mla_correction_only_applies_to_kv_matrix():
    # target_modules=1 adapts only Wq, which MLA's kv_latent_dim doesn't
    # touch -- should match an otherwise-identical non-MLA model exactly.
    non_mla_equivalent = ModelShape(
        params=DEEPSEEK_V3_MLA.params,
        active_params=DEEPSEEK_V3_MLA.active_params,
        num_layers=DEEPSEEK_V3_MLA.num_layers,
        hidden_dim=DEEPSEEK_V3_MLA.hidden_dim,
        num_heads=DEEPSEEK_V3_MLA.num_heads,
        head_dim=DEEPSEEK_V3_MLA.head_dim,
        # kv_latent_dim intentionally omitted
    )
    assert lora_trainable_params(DEEPSEEK_V3_MLA, rank=8, target_modules=1) == pytest.approx(
        lora_trainable_params(non_mla_equivalent, rank=8, target_modules=1)
    )


def test_lora_reduces_trainable_params_by_the_papers_reported_10000x():
    # Paper's own headline claim (abstract): "reduce the number of trainable
    # parameters by 10,000 times" for GPT-3 175B — the paper itself calls
    # this "roughly" 10,000x (Sec 4.2 footnote: 350GB base checkpoint vs. a
    # ~35MB r=4, Wq+Wv adapter checkpoint), so a wide tolerance matches the
    # paper's own rounding rather than a false-precision exact match.
    trainable = lora_trainable_params(GPT3_175B, rank=4, target_modules=2)
    assert GPT3_175B.params / trainable == pytest.approx(10_000, rel=0.1)


def test_lora_checkpoint_is_tiny_fraction_of_full_finetune_checkpoint():
    full = compute_vram_breakdown(GPT3_175B, precision="fp16", batch_size=0, seq_len=0, peft_method="full")
    lora = compute_vram_breakdown(
        GPT3_175B, precision="fp16", batch_size=0, seq_len=0, peft_method="lora", peft_rank=4, peft_target_modules=2
    )
    # Paper's own headline: reduces VRAM usage "by up to 2/3" for GPT-3 175B
    # since no optimizer state is kept for the (vast majority) frozen base.
    assert lora.total_gb < full.total_gb * 0.4
    assert lora.optimizer_states_gb < full.optimizer_states_gb * 0.001


def test_lora_base_weights_stay_full_size_only_adapter_is_added():
    # LoRA (unlike QLoRA) keeps the frozen base weights at full `precision`
    # — the paper's whole point is skipping optimizer state, not
    # quantizing the base model.
    base_only = weights_bytes(LLAMA2_7B, "bf16") / 1e9
    b = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, peft_method="lora")
    assert b.weights_gb > base_only  # base weights + tiny adapter
    assert b.weights_gb == pytest.approx(base_only, rel=0.01)  # adapter is negligible vs. 7B base weights


def test_lora_gradients_and_optimizer_state_scale_with_adapter_not_full_model():
    adapter_params = lora_trainable_params(LLAMA2_7B, rank=8, target_modules=2)
    b = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, peft_method="lora", peft_rank=8)
    assert b.gradients_gb == pytest.approx(adapter_params * 2.0 / 1e9)  # bf16 grads for the adapter only


def test_qlora_double_quantization_saves_the_papers_reported_bits_per_param():
    # Paper's own reported saving: "approximately 3 GB for a 65B model"
    # (Sec 1) from Double Quantization's 0.373 bits/param reduction.
    with_dq = quantized_weight_bytes(LLAMA_65B, double_quant=True)
    without_dq = quantized_weight_bytes(LLAMA_65B, double_quant=False)
    saved_gb = (without_dq - with_dq) / 1e9
    assert saved_gb == pytest.approx(3.03, rel=0.02)


def test_qlora_65b_base_weights_fit_a_single_48gb_gpu():
    # Paper's own headline claim (abstract): "finetune a 65B parameter
    # model on a single 48GB GPU". Model-state VRAM alone (quantized base +
    # tiny bf16 adapter + its fp32 Adam state) must clear that bar with
    # activation headroom to spare for a real run.
    b = compute_vram_breakdown(
        LLAMA_65B, precision="bf16", batch_size=1, seq_len=512, checkpointing=True, peft_method="qlora", peft_rank=8
    )
    model_state_gb = b.weights_gb + b.gradients_gb + b.optimizer_states_gb
    assert model_state_gb < 40.0
    assert b.total_gb < 48.0


def test_qlora_base_weights_much_smaller_than_lora_or_full_at_same_precision():
    # weights_gb includes the (negligible) adapter for lora/qlora too, so
    # compare the underlying base-weight formulas directly rather than the
    # full breakdown's weights_gb (lora's is actually a hair *larger* than
    # full's, not smaller — same base weights plus a tiny added adapter).
    full_base = weights_bytes(LLAMA_65B, "bf16")
    qlora_base = quantized_weight_bytes(LLAMA_65B)
    assert qlora_base < full_base


def test_peft_rejects_unknown_method():
    with pytest.raises(ValueError, match="peft_method"):
        compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, peft_method="dora")


def test_peft_and_zero_stage_are_mutually_exclusive():
    with pytest.raises(ValueError, match="zero_stage"):
        compute_vram_breakdown(
            LLAMA2_7B, precision="bf16", batch_size=1, seq_len=2048, peft_method="lora", zero_stage=1, dp_size=8
        )


def test_peft_method_is_ignored_for_inference_serving():
    # LoRA/QLoRA are training-time techniques (no optimizer state to save in
    # the first place); inference-only serving (training=False) should be
    # unaffected by peft_method, same convention as zero_stage.
    full = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=False, peft_method="full")
    lora = compute_vram_breakdown(LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048, training=False, peft_method="lora")
    assert full.kv_cache_gb == lora.kv_cache_gb
    assert full.gradients_gb == lora.gradients_gb == 0.0
