import pytest
from fastapi import HTTPException

from api.routers.calculate import calculate_cost, calculate_spot_pricing, calculate_vram
from api.schemas import CostRequest, ModelShapeIn, SpotPricingRequest, TopologyRequest, VramRequest

LLAMA2_7B = ModelShapeIn(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)


def _cost_request(**overrides) -> CostRequest:
    defaults = dict(
        model=LLAMA2_7B,
        topology=TopologyRequest(shape="single_gpu", gpu_id="h100-sxm"),
        tokens_per_step=32768,
        total_training_tokens=1e11,
    )
    defaults.update(overrides)
    return CostRequest(**defaults)


def test_cost_endpoint_includes_carbon_fields_by_default():
    res = calculate_cost(_cost_request())
    assert res.carbon_region == "global-avg"
    assert res.grid_intensity_g_per_kwh > 0
    assert res.co2e_kg > 0
    assert res.equivalent_car_km > 0
    assert res.equivalent_flights_ny_london > 0
    # co2e should be internally consistent with the reported energy figure.
    assert res.co2e_kg == pytest.approx(res.total_energy_kwh * res.grid_intensity_g_per_kwh / 1000.0)


def test_cost_endpoint_respects_carbon_region_override():
    dirty = calculate_cost(_cost_request(carbon_region="india"))
    clean = calculate_cost(_cost_request(carbon_region="eu-france-nuclear"))
    assert dirty.total_energy_kwh == pytest.approx(clean.total_energy_kwh)  # same run, different grid only
    assert dirty.co2e_kg > clean.co2e_kg


def test_cost_endpoint_rejects_unknown_carbon_region():
    with pytest.raises(HTTPException) as exc_info:
        calculate_cost(_cost_request(carbon_region="mars"))
    assert exc_info.value.status_code == 400


def test_spot_pricing_endpoint_returns_expected_savings():
    req = SpotPricingRequest(
        on_demand_price_per_hr_usd=4.5,
        total_gpus=8,
        base_time_hours=500.0,
        step_time_s=10.0,
        checkpoint_interval_steps=50,
        checkpoint_size_gb=20.0,
    )
    res = calculate_spot_pricing(req)
    assert res.on_demand_cost_usd > 0
    assert res.expected_total_cost_usd < res.on_demand_cost_usd
    assert res.savings_pct > 0


def test_spot_pricing_request_schema_rejects_invalid_fields():
    # Pydantic's own Field constraints (gt/ge/lt) already cover every input
    # engine.spot.estimate_spot_run would otherwise raise ValueError on —
    # confirms that validation boundary at the schema level, at the API's
    # actual entry point (request construction), rather than duplicating
    # engine.spot's own ValueError tests (see tests/test_spot.py) here.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SpotPricingRequest(
            on_demand_price_per_hr_usd=4.5,
            total_gpus=0,  # gt=0
            base_time_hours=500.0,
            step_time_s=10.0,
            checkpoint_interval_steps=50,
            checkpoint_size_gb=20.0,
        )
    with pytest.raises(ValidationError):
        SpotPricingRequest(
            on_demand_price_per_hr_usd=4.5,
            total_gpus=8,
            base_time_hours=500.0,
            step_time_s=10.0,
            checkpoint_interval_steps=50,
            checkpoint_size_gb=20.0,
            discount=1.0,  # lt=1
        )


def _vram_request(**overrides) -> VramRequest:
    defaults = dict(model=LLAMA2_7B, precision="bf16", batch_size=4, seq_len=2048)
    defaults.update(overrides)
    return VramRequest(**defaults)


def test_vram_endpoint_zero_stage_shrinks_model_state_memory():
    baseline = calculate_vram(_vram_request())
    sharded = calculate_vram(_vram_request(zero_stage=3, dp_size=8))
    assert sharded.optimizer_states_gb < baseline.optimizer_states_gb
    assert sharded.gradients_gb < baseline.gradients_gb
    assert sharded.weights_gb < baseline.weights_gb
    # Activations are untouched by ZeRO-DP model-state sharding.
    assert sharded.activations_gb == baseline.activations_gb


def test_vram_endpoint_rejects_invalid_zero_stage():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _vram_request(zero_stage=4)


def test_cost_endpoint_zero_stage3_increases_communication_time():
    baseline = calculate_cost(_cost_request())
    sharded = calculate_cost(_cost_request(zero_stage=3))
    assert sharded.communication_s_per_step == pytest.approx(baseline.communication_s_per_step * 1.5)
    assert sharded.compute_s_per_step == baseline.compute_s_per_step


def test_cost_endpoint_zero_stage_1_and_2_leave_communication_time_unchanged():
    baseline = calculate_cost(_cost_request())
    for stage in (1, 2):
        result = calculate_cost(_cost_request(zero_stage=stage))
        assert result.communication_s_per_step == pytest.approx(baseline.communication_s_per_step)


# --- LoRA / QLoRA (Hu et al. 2021 / Dettmers et al. 2023) ---


def test_vram_endpoint_lora_shrinks_optimizer_state_and_gradients():
    baseline = calculate_vram(_vram_request())
    lora = calculate_vram(_vram_request(peft_method="lora", peft_rank=8, peft_target_modules=2))
    assert lora.optimizer_states_gb < baseline.optimizer_states_gb
    assert lora.gradients_gb < baseline.gradients_gb
    # Base weights stay resident (frozen, not removed) — LoRA adds a tiny
    # adapter on top rather than shrinking weights_gb.
    assert lora.weights_gb >= baseline.weights_gb


def test_vram_endpoint_qlora_shrinks_weights_too():
    lora = calculate_vram(_vram_request(peft_method="lora"))
    qlora = calculate_vram(_vram_request(peft_method="qlora"))
    assert qlora.weights_gb < lora.weights_gb


def test_vram_endpoint_rejects_unknown_peft_method():
    with pytest.raises(HTTPException) as exc_info:
        calculate_vram(_vram_request(peft_method="dora"))
    assert exc_info.value.status_code == 400


def test_vram_endpoint_rejects_peft_combined_with_zero_stage():
    with pytest.raises(HTTPException) as exc_info:
        calculate_vram(_vram_request(peft_method="lora", zero_stage=1, dp_size=8))
    assert exc_info.value.status_code == 400
