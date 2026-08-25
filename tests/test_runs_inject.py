import pytest
from fastapi import HTTPException

from api.routers.runs import inject_failure
from api.runs_store import store
from api.schemas import ChaosInjectRequest


def test_inject_unknown_run_returns_404():
    with pytest.raises(HTTPException) as exc_info:
        inject_failure("nope", ChaosInjectRequest(kind="xid_error", severity=5.0, duration_steps=3))
    assert exc_info.value.status_code == 404


def test_inject_into_non_running_run_returns_409():
    run_id = "inject-test-done"
    store.start(run_id, {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1})
    store.finish(run_id)
    with pytest.raises(HTTPException) as exc_info:
        inject_failure(run_id, ChaosInjectRequest(kind="xid_error", severity=5.0, duration_steps=3))
    assert exc_info.value.status_code == 409


@pytest.mark.parametrize(
    "req",
    [
        ChaosInjectRequest(kind="node_drain", severity=0.5, duration_steps=None),  # not an integer
        ChaosInjectRequest(kind="nvlink_degradation", severity=1.5, duration_steps=5),  # out of (0,1) range
        ChaosInjectRequest(kind="nvlink_degradation", severity=0.5, duration_steps=None),  # missing duration
        ChaosInjectRequest(kind="xid_error", severity=1.0, duration_steps=5),  # not > 1
        ChaosInjectRequest(kind="xid_error", severity=5.0, duration_steps=None),  # missing duration
        ChaosInjectRequest(kind="bogus_kind", severity=1.0, duration_steps=None),
    ],
)
def test_inject_rejects_invalid_requests_with_400(req):
    run_id = "inject-test-invalid"
    store.start(run_id, {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1})
    with pytest.raises(HTTPException) as exc_info:
        inject_failure(run_id, req)
    assert exc_info.value.status_code == 400


def test_successful_injection_appears_in_run_chaos_events():
    run_id = "inject-test-ok"
    store.start(
        run_id,
        {
            "model": "llama2-7b",
            "gpu": "h100-sxm",
            "topology": "multi_node",
            "total_gpus": 32,
            "compute_s_per_step": 0.5,
            "communication_s_per_step": 0.2,
            "total_s_per_step": 0.7,
            "total_steps": 50,
        },
    )
    inject_failure(run_id, ChaosInjectRequest(kind="node_drain", severity=1, duration_steps=None))

    run = store.get(run_id)
    assert len(run.chaos_events) == 1
    event = run.chaos_events[0]
    assert event["kind"] == "node_drain"
    assert event["severity"] == 1
    assert event["duration_steps"] is None  # forced permanent regardless of request
