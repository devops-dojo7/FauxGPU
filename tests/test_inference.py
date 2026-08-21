import pytest

from engine.gpu_specs import get_gpu
from engine.inference import decode_step_seconds, prefill_seconds, simulate_serving
from engine.memory import ModelShape

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)
H100 = get_gpu("h100-sxm")


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
