import pytest
from fastapi import HTTPException

from api.routers.autoscaling import simulate
from api.schemas import AutoscalingConfigIn, AutoscalingRequest, ModelShapeIn, TrafficStageIn

LLAMA3_8B = ModelShapeIn(params=8.03e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128, num_kv_heads=8)


def _req(**overrides) -> AutoscalingRequest:
    defaults = dict(
        model=LLAMA3_8B,
        gpu_id="h100-sxm",
        stages=[TrafficStageIn(duration_s=20, target_rps=10), TrafficStageIn(duration_s=30, target_rps=40)],
        config=AutoscalingConfigIn(min_replicas=1, max_replicas=8),
    )
    defaults.update(overrides)
    return AutoscalingRequest(**defaults)


def test_simulate_returns_points_and_summary():
    res = simulate(_req())
    assert len(res.points) > 0
    assert res.per_replica_capacity_rps > 0
    assert res.peak_replicas >= 1


def test_simulate_400_on_unknown_gpu():
    with pytest.raises(HTTPException) as exc_info:
        simulate(_req(gpu_id="not-a-real-gpu"))
    assert exc_info.value.status_code == 400


def test_simulate_400_on_max_less_than_min():
    with pytest.raises(HTTPException) as exc_info:
        simulate(_req(config=AutoscalingConfigIn(min_replicas=5, max_replicas=1)))
    assert exc_info.value.status_code == 400
