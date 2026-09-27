"""Sanity-check engine formulas against commonly cited reference numbers for
Llama-2 7B (params=6.74B, 32 layers, hidden=4096, 32 heads, head_dim=128).
"""

import pytest

from engine.memory import (
    ModelShape,
    activation_bytes,
    compute_vram_breakdown,
    kv_cache_bytes,
    optimizer_state_bytes,
    weights_bytes,
)

LLAMA2_7B = ModelShape(
    params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128
)

# The ZeRO paper's own worked example (Rajbhandari et al. 2020, Table 1,
# https://arxiv.org/abs/1910.02054): a 7.5B model, mixed-precision Adam
# (K=12, so model states = 16 bytes/param unsharded).
ZERO_PAPER_MODEL = ModelShape(params=7.5e9, num_layers=1, hidden_dim=1, num_heads=1, head_dim=1)


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
