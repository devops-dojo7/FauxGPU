import pytest

from engine.gpu_specs import get_gpu
from engine.memory import ModelShape
from engine.speculative import expected_tokens_per_round, simulate_speculative_decoding

H100 = get_gpu("h100-sxm")
DRAFT = ModelShape(params=1.1e9, num_layers=22, hidden_dim=2048, num_heads=16, head_dim=128)
TARGET = ModelShape(params=70.0e9, num_layers=80, hidden_dim=8192, num_heads=64, head_dim=128)


def test_expected_tokens_at_full_acceptance_is_ceiling():
    assert expected_tokens_per_round(gamma=4, acceptance_rate=1.0) == 5


def test_expected_tokens_grows_with_acceptance_rate():
    low = expected_tokens_per_round(gamma=4, acceptance_rate=0.3)
    high = expected_tokens_per_round(gamma=4, acceptance_rate=0.9)
    assert 1 < low < high < 5


def test_speedup_over_one_with_good_acceptance():
    result = simulate_speculative_decoding(
        DRAFT, TARGET, H100, "bf16", gamma=4, acceptance_rate=0.8, avg_kv_tokens=1024
    )
    assert result.speedup > 1.0
    assert result.speculative_tokens_per_sec > result.baseline_tokens_per_sec


def test_low_acceptance_can_be_slower_than_baseline():
    # A near-useless draft model still costs gamma draft-steps per round for
    # almost no accepted tokens — speculative decoding isn't free.
    result = simulate_speculative_decoding(
        DRAFT, TARGET, H100, "bf16", gamma=8, acceptance_rate=0.05, avg_kv_tokens=1024
    )
    assert result.speedup < 1.0


def test_draft_step_much_faster_than_verify_for_much_smaller_model():
    result = simulate_speculative_decoding(
        DRAFT, TARGET, H100, "bf16", gamma=4, acceptance_rate=0.8, avg_kv_tokens=1024
    )
    assert result.draft_step_s < result.verify_step_s
