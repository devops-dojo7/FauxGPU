import pytest

from engine.compute import flops_per_step
from engine.gpu_specs import get_gpu
from engine.inference import decode_step_seconds, estimate_serving_capacity, prefill_seconds
from engine.memory import ModelShape, kv_cache_bytes_per_token, weights_bytes

H100 = get_gpu("h100-sxm")

DENSE = ModelShape(params=7e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)
GQA = ModelShape(params=7e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128, num_kv_heads=8)
MOE = ModelShape(params=671e9, num_layers=61, hidden_dim=7168, num_heads=128, head_dim=128, active_params=37e9)
MLA = ModelShape(params=671e9, num_layers=61, hidden_dim=7168, num_heads=128, head_dim=128, kv_latent_dim=576)


def test_gqa_shrinks_kv_cache_proportionally_to_kv_heads():
    dense_kv = kv_cache_bytes_per_token(DENSE, "bf16")
    gqa_kv = kv_cache_bytes_per_token(GQA, "bf16")
    assert gqa_kv == pytest.approx(dense_kv * (8 / 32))


def test_mha_default_matches_num_heads():
    assert DENSE.effective_kv_heads == DENSE.num_heads


def test_moe_flops_use_active_params_not_total():
    dense_flops = flops_per_step(DENSE, tokens_per_step=1000)
    moe_flops = flops_per_step(MOE, tokens_per_step=1000)
    # MOE has ~96x the total params of DENSE but only ~5x the active params.
    assert moe_flops < dense_flops * 10
    assert MOE.effective_active_params == 37e9


def test_moe_weights_bytes_use_total_params_not_active():
    # VRAM must hold every expert even though only some are active per token.
    w = weights_bytes(MOE, "bf16")
    assert w == pytest.approx(671e9 * 2)


def test_moe_decode_is_cheaper_per_step_than_dense_dense_flops_would_suggest():
    dense_step = decode_step_seconds(DENSE, H100, "bf16", batch_size=1, avg_kv_tokens=1024)
    moe_step = decode_step_seconds(MOE, H100, "bf16", batch_size=1, avg_kv_tokens=1024)
    # MOE's active-weight read (37B) is bigger than dense's full 7B read, so
    # its decode step is slower here, but nowhere near "671B-params slower".
    assert moe_step > dense_step
    assert moe_step < dense_step * 10


def test_mla_kv_cache_much_smaller_than_standard_formula_would_give():
    mla_kv = kv_cache_bytes_per_token(MLA, "bf16")
    # A same-shape non-MLA model (128 heads, head_dim 128) would have a huge
    # per-token KV cache; MLA's compressed latent should be far smaller.
    non_mla_equivalent = ModelShape(
        params=MLA.params, num_layers=MLA.num_layers, hidden_dim=MLA.hidden_dim,
        num_heads=MLA.num_heads, head_dim=MLA.head_dim,
    )
    assert mla_kv < kv_cache_bytes_per_token(non_mla_equivalent, "bf16") / 10


def test_prefill_seconds_uses_active_params_for_moe():
    dense_ttft = prefill_seconds(DENSE, 2048, H100, "bf16")
    moe_ttft = prefill_seconds(MOE, 2048, H100, "bf16")
    assert moe_ttft > dense_ttft  # 37B active > 7B active
    assert moe_ttft < dense_ttft * 10  # but nowhere near the 671B/7B total-param ratio


def test_paged_attention_rounds_kv_up_to_block_boundary():
    unpaged = decode_step_seconds(DENSE, H100, "bf16", batch_size=1, avg_kv_tokens=17, paged_attention=False)
    paged = decode_step_seconds(DENSE, H100, "bf16", batch_size=1, avg_kv_tokens=17, paged_attention=True, block_size=16)
    # 17 tokens rounds up to 32 (two 16-token blocks), so paged should read more.
    assert paged > unpaged


def test_serving_capacity_bounded_by_gpu_memory_utilization():
    full_util = estimate_serving_capacity(DENSE, H100, "bf16", avg_seq_len=2048, gpu_memory_utilization=0.9)
    low_util = estimate_serving_capacity(DENSE, H100, "bf16", avg_seq_len=2048, gpu_memory_utilization=0.3)
    assert full_util.max_concurrent_sequences > low_util.max_concurrent_sequences
    assert full_util.usable_vram_gb == pytest.approx(H100.vram_gb * 0.9)
