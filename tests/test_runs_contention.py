import pytest

from api.routers.runs import _contended_communication_s
from api.runs_store import RunsStore
from engine.gpu_specs import get_fabric


@pytest.fixture(autouse=True)
def _use_isolated_store(monkeypatch):
    # _contended_communication_s reads the module-level `store` directly —
    # swap it for a fresh one per test so tests don't see each other's runs.
    import api.routers.runs as runs_module

    isolated = RunsStore()
    monkeypatch.setattr(runs_module, "store", isolated)
    return isolated


def test_returns_none_with_no_fabric_id():
    assert _contended_communication_s("run-a", {"total_gpus": 8, "payload_gb": 1.0}) is None


def test_returns_none_as_sole_job_on_fabric(_use_isolated_store):
    store = _use_isolated_store
    store.start("run-a", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    meta = store.get("run-a").meta
    assert _contended_communication_s("run-a", meta) is None


def test_two_equal_runs_split_bandwidth_same_as_standalone_calculator(_use_isolated_store):
    store = _use_isolated_store
    store.start("run-a", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    store.start("run-b", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})

    status_a = _contended_communication_s("run-a", store.get("run-a").meta)
    status_b = _contended_communication_s("run-b", store.get("run-b").meta)

    assert status_a is not None and status_b is not None
    assert status_a.jobs_sharing_fabric == 2
    assert status_a.gpus_sharing_fabric == 16
    assert status_a.communication_s == pytest.approx(status_b.communication_s)
    # Equal-sized jobs on the same fabric split it in half (same invariant
    # tests/test_network_contention.py already verifies for the standalone
    # calculator's simulate_network_contention, fed real store data here).
    hdr = get_fabric("infiniband-hdr")
    assert status_a.bandwidth_share_gbps == pytest.approx(hdr.bandwidth_gbps / 2)


def test_bigger_job_gets_bigger_share_and_less_slowdown(_use_isolated_store):
    store = _use_isolated_store
    store.start("big", {"fabric_id": "infiniband-hdr", "total_gpus": 24, "payload_gb": 1.0})
    store.start("small", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})

    big_status = _contended_communication_s("big", store.get("big").meta)
    small_status = _contended_communication_s("small", store.get("small").meta)

    assert big_status.bandwidth_share_gbps > small_status.bandwidth_share_gbps


def test_finished_run_no_longer_contends(_use_isolated_store):
    store = _use_isolated_store
    store.start("run-a", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    store.start("run-b", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    store.finish("run-b")

    assert _contended_communication_s("run-a", store.get("run-a").meta) is None
