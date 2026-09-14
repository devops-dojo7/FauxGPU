import pytest

from engine.gpu_specs import get_gpu
from engine.inference import decode_step_seconds, estimate_serving_capacity, prefill_seconds, simulate_serving
from engine.memory import ModelShape

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)
H100 = get_gpu("h100-sxm")
L40S = get_gpu("l40s")  # no NVLink
B200 = get_gpu("b200-sxm")

# DeepSeek-V4.1-Flash's real published shape (web/src/lib/types.ts) — its
# 552B total params don't fit on a single B200 (max_concurrent_sequences==0
# at tp_degree=1), which is exactly the case tp_degree sharding should fix.
DEEPSEEK_V41_FLASH = ModelShape(
    params=552.0e9, active_params=16.0e9, num_layers=40, hidden_dim=5120,
    num_heads=64, head_dim=128, kv_latent_dim=576,
)


def test_cache_hit_reduces_prefill_time():
    cold = prefill_seconds(LLAMA2_7B, 2048, H100, "bf16", cache_hit_fraction=0.0)
    warm = prefill_seconds(LLAMA2_7B, 2048, H100, "bf16", cache_hit_fraction=0.8)
    assert warm < cold
    assert warm == pytest.approx(cold * 0.2)


def test_decode_step_scales_with_batch_and_kv_length():
    small_batch = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=1, avg_kv_tokens=1024)
    big_batch = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=8, avg_kv_tokens=1024)
    assert big_batch > small_batch


def test_disaggregated_throughput_unaffected_by_request_rate():
    low_load = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=0.01)
    high_load = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=5.0)
    assert low_load.disaggregated_tokens_per_sec_per_gpu == high_load.disaggregated_tokens_per_sec_per_gpu


def test_colocated_throughput_degrades_with_request_rate():
    low_load = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=0.01)
    high_load = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=5.0)
    assert high_load.colocated_tokens_per_sec_per_gpu < low_load.colocated_tokens_per_sec_per_gpu
    assert high_load.colocated_tokens_per_sec_per_gpu >= 0


def test_colocated_never_exceeds_disaggregated():
    result = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=1.0)
    assert result.colocated_tokens_per_sec_per_gpu <= result.disaggregated_tokens_per_sec_per_gpu


def test_tp_degree_default_matches_single_gpu_behavior():
    explicit = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=8, avg_kv_tokens=1024, tp_degree=1)
    default = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=8, avg_kv_tokens=1024)
    assert explicit == default


def test_tensor_parallel_decode_roughly_halves_compute_at_tp2():
    tp1 = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=8, avg_kv_tokens=1024, tp_degree=1)
    tp2 = decode_step_seconds(LLAMA2_7B, H100, "bf16", batch_size=8, avg_kv_tokens=1024, tp_degree=2)
    # tp2 < tp1 (compute halves) but tp2 > tp1/2 (communication overhead eats into the win)
    assert tp1 / 2 < tp2 < tp1


def test_tensor_parallel_prefill_communication_overhead_reported():
    result = simulate_serving(LLAMA2_7B, H100, "bf16", 2048, 256, 8, requests_per_sec=1.0, tp_degree=4)
    assert result.tp_communication_overhead_fraction > 0
    assert result.tokens_per_sec_per_gpu_amortized == pytest.approx(result.tokens_per_sec_per_gpu / 4)


def test_tensor_parallel_without_nvlink_is_infeasible():
    assert decode_step_seconds(LLAMA2_7B, L40S, "bf16", batch_size=8, avg_kv_tokens=1024, tp_degree=4) == float("inf")
    assert prefill_seconds(LLAMA2_7B, 2048, L40S, "bf16", tp_degree=4) == float("inf")


def test_sharding_lets_an_oversized_model_fit():
    unsharded = estimate_serving_capacity(DEEPSEEK_V41_FLASH, B200, "bf16", avg_seq_len=2048)
    assert unsharded.max_concurrent_sequences == 0  # 552B params, bf16, don't fit on one 192GB B200

    sharded = estimate_serving_capacity(DEEPSEEK_V41_FLASH, B200, "bf16", avg_seq_len=2048, tp_degree=8)
    assert sharded.max_concurrent_sequences > 0
    assert sharded.weights_gb == pytest.approx(unsharded.weights_gb / 8)
