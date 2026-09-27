import pytest
from fastapi import HTTPException

from api.routers.calculate import calculate_cost, calculate_spot_pricing
from api.schemas import CostRequest, ModelShapeIn, SpotPricingRequest, TopologyRequest

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
