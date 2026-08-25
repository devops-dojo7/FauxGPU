import pytest

from engine.memory import ModelShape
from engine.recommend import recommend_configurations

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)


def test_recommend_returns_feasible_ranked_candidates():
    candidates = recommend_configurations(
        LLAMA2_7B,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e9,
        objective="cost",
        max_gpus=16,
    )
    assert len(candidates) > 0
    costs = [c.total_cost_usd for c in candidates]
    assert costs == sorted(costs)
    for c in candidates:
        assert c.vram_headroom_gb >= 0  # only feasible (VRAM-fitting) candidates should be returned
        assert c.num_gpus >= 1
        # total_cost_usd can be 0 for unreleased/roadmap GPUs priced at $0/hr
        assert c.total_cost_usd >= 0
        assert c.total_time_hours > 0


def test_recommend_time_objective_ranks_by_time_not_cost():
    candidates = recommend_configurations(
        LLAMA2_7B,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e9,
        objective="time",
        max_gpus=16,
    )
    times = [c.total_time_hours for c in candidates]
    assert times == sorted(times)


def test_recommend_respects_budget_constraint():
    unconstrained = recommend_configurations(
        LLAMA2_7B, precision="bf16", tokens_per_step=32768, total_training_tokens=1e9, objective="cost", max_gpus=16
    )
    tight_budget = min(c.total_cost_usd for c in unconstrained) - 0.01
    constrained = recommend_configurations(
        LLAMA2_7B,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e9,
        objective="cost",
        max_gpus=16,
        budget_usd=tight_budget,
    )
    assert constrained == []


def test_recommend_restricts_to_candidate_gpu_ids():
    candidates = recommend_configurations(
        LLAMA2_7B,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e9,
        objective="cost",
        max_gpus=16,
        candidate_gpu_ids=["h100-sxm"],
    )
    assert all(c.gpu_id == "h100-sxm" for c in candidates)


def test_recommend_rejects_unknown_objective():
    with pytest.raises(ValueError):
        recommend_configurations(
            LLAMA2_7B, precision="bf16", tokens_per_step=32768, total_training_tokens=1e9, objective="bogus"
        )
