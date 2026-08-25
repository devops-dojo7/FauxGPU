import pytest
from fastapi import HTTPException

from api.routers.network import contention
from api.schemas import NetworkContentionRequest, NetworkJobIn


def test_contention_happy_path():
    req = NetworkContentionRequest(
        fabric_id="infiniband-hdr",
        jobs=[
            NetworkJobIn(job_id="j1", team="team-a", num_gpus=8, payload_gb=1.0),
            NetworkJobIn(job_id="j2", team="team-b", num_gpus=8, payload_gb=1.0),
        ],
    )
    res = contention(req)
    assert res.fabric_id == "infiniband-hdr"
    assert len(res.jobs) == 2
    assert res.total_gpus_sharing_fabric == 16


def test_contention_400_on_unknown_fabric():
    req = NetworkContentionRequest(fabric_id="not-a-real-fabric", jobs=[NetworkJobIn(job_id="j1", team="a", num_gpus=8, payload_gb=1.0)])
    with pytest.raises(HTTPException) as exc_info:
        contention(req)
    assert exc_info.value.status_code == 400


def test_contention_400_on_empty_jobs():
    req = NetworkContentionRequest(fabric_id="infiniband-hdr", jobs=[])
    with pytest.raises(HTTPException) as exc_info:
        contention(req)
    assert exc_info.value.status_code == 400
