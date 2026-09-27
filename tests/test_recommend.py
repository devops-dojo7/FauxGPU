import pytest

from engine.memory import ModelShape
from engine.recommend import recommend_configurations

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)

# Mixtral 8x7B (real config: mistralai/Mixtral-8x7B-v0.1) — a modestly sized
# MoE model whose total params already fit on a single large GPU even at
# ep_degree=1, unlike DEEPSEEK_V3 below.
MIXTRAL_8X7B = ModelShape(
    params=46.7e9,
    active_params=12.9e9,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    head_dim=128,
    num_kv_heads=8,
    num_experts=8,
    top_k=2,
)

# DeepSeek-V3 (real config: deepseek-ai/DeepSeek-V3) — 671B total params
# exceeds every GPU's VRAM at ep_degree=1 even after TP8xPP4 sharding, so
# this model can only appear in recommend_configurations' results if EP
# sharding is actually applied to the per-GPU VRAM feasibility check, not
# just to the communication-time estimate (see test_recommend_moe_model_needs_ep_to_fit_vram).
DEEPSEEK_V3 = ModelShape(
    params=671e9,
    active_params=37e9,
    num_layers=61,
    hidden_dim=7168,
    num_heads=128,
    head_dim=128,
    kv_latent_dim=576,
    num_experts=256,
    top_k=8,
)


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


def test_recommend_dense_model_never_searches_ep_degree():
    """A dense model (no num_experts/top_k) should only ever appear with
    ep_degree=1 — searching ep_degree > 1 would be pure wasted GPU count for
    a model expert_all_to_all_seconds treats as a no-op regardless."""
    candidates = recommend_configurations(
        LLAMA2_7B, precision="bf16", tokens_per_step=32768, total_training_tokens=1e9, objective="cost", max_gpus=64
    )
    assert len(candidates) > 0
    assert all(c.ep_degree == 1 for c in candidates)


def test_recommend_moe_model_can_use_ep_degree():
    """A real MoE model (Mixtral 8x7B) should be able to appear with
    ep_degree > 1 among its candidates — proof the EP search dimension
    (added alongside expert parallelism) is actually exercised, not just
    accepted as an unused parameter."""
    candidates = recommend_configurations(
        MIXTRAL_8X7B,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e9,
        objective="cost",
        max_gpus=64,
        max_results=50,
    )
    assert len(candidates) > 0
    assert any(c.ep_degree > 1 for c in candidates)


def test_recommend_moe_model_needs_ep_to_fit_vram():
    """Regression test for a real bug found during expert-parallelism
    validation: DeepSeek-V3's 671B total params need far more than any
    single GPU's VRAM even after TP8xPP4 sharding alone (~42GB/GPU for
    weights, ~383GB/GPU total once gradients + Adam optimizer state for a
    full-parameter training run are included — every real, obtainable GPU
    in this project's catalog tops out at 256GB), so recommend_configurations
    returned zero candidates for it until the per-GPU VRAM feasibility
    check was fixed to actually divide resident params by ep_degree too
    (previously only tp_degree*pp_degree), same as any real EP deployment
    relies on to fit such a large expert pool at all."""
    candidates = recommend_configurations(
        DEEPSEEK_V3,
        precision="bf16",
        tokens_per_step=32768,
        total_training_tokens=1e12,
        objective="cost",
        max_gpus=256,
    )
    assert len(candidates) > 0
    assert all(c.ep_degree > 1 for c in candidates)  # ep_degree=1 alone can never fit this model
    assert all(c.vram_headroom_gb >= 0 for c in candidates)
