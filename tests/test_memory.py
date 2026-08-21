"""Sanity-check engine formulas against commonly cited reference numbers for
Llama-2 7B (params=6.74B, 32 layers, hidden=4096, 32 heads, head_dim=128).
"""

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
