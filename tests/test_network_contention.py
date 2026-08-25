import pytest

from engine.gpu_specs import get_fabric
from engine.network_contention import NetworkJob, simulate_network_contention

HDR = get_fabric("infiniband-hdr")  # 200 Gbps


def test_single_job_alone_gets_full_bandwidth_no_slowdown():
    result = simulate_network_contention(HDR, [NetworkJob("j1", "team-a", num_gpus=8, payload_gb=1.0)])
    job = result.jobs[0]
    assert job.bandwidth_share_gbps == HDR.bandwidth_gbps
    assert job.contended_comm_s == pytest.approx(job.isolated_comm_s)
    assert job.slowdown_factor == pytest.approx(1.0)


def test_two_equal_sized_jobs_split_bandwidth_in_half():
    jobs = [
        NetworkJob("j1", "team-a", num_gpus=8, payload_gb=1.0),
        NetworkJob("j2", "team-b", num_gpus=8, payload_gb=1.0),
    ]
    result = simulate_network_contention(HDR, jobs)
    for job in result.jobs:
        assert job.bandwidth_share_gbps == pytest.approx(HDR.bandwidth_gbps / 2)
        assert job.slowdown_factor == pytest.approx(2.0)


def test_larger_job_gets_proportionally_bigger_share():
    jobs = [
        NetworkJob("big", "team-a", num_gpus=24, payload_gb=1.0),
        NetworkJob("small", "team-b", num_gpus=8, payload_gb=1.0),
    ]
    result = simulate_network_contention(HDR, jobs)
    by_id = {j.job_id: j for j in result.jobs}
    assert by_id["big"].bandwidth_share_gbps == pytest.approx(HDR.bandwidth_gbps * 0.75)
    assert by_id["small"].bandwidth_share_gbps == pytest.approx(HDR.bandwidth_gbps * 0.25)
    assert by_id["big"].bandwidth_share_gbps > by_id["small"].bandwidth_share_gbps


def test_contended_is_never_faster_than_isolated_with_multiple_jobs():
    jobs = [
        NetworkJob("j1", "team-a", num_gpus=4, payload_gb=2.0),
        NetworkJob("j2", "team-b", num_gpus=12, payload_gb=0.5),
        NetworkJob("j3", "team-c", num_gpus=8, payload_gb=1.5),
    ]
    result = simulate_network_contention(HDR, jobs)
    for job in result.jobs:
        assert job.contended_comm_s >= job.isolated_comm_s


def test_bandwidth_shares_sum_to_fabric_bandwidth():
    jobs = [
        NetworkJob("j1", "team-a", num_gpus=4, payload_gb=1.0),
        NetworkJob("j2", "team-b", num_gpus=12, payload_gb=1.0),
        NetworkJob("j3", "team-c", num_gpus=8, payload_gb=1.0),
    ]
    result = simulate_network_contention(HDR, jobs)
    assert sum(j.bandwidth_share_gbps for j in result.jobs) == pytest.approx(HDR.bandwidth_gbps)
    assert result.total_gpus_sharing_fabric == 24


def test_empty_job_list_rejected():
    with pytest.raises(ValueError):
        simulate_network_contention(HDR, [])


def test_duplicate_job_ids_rejected():
    jobs = [
        NetworkJob("dupe", "team-a", num_gpus=4, payload_gb=1.0),
        NetworkJob("dupe", "team-b", num_gpus=4, payload_gb=1.0),
    ]
    with pytest.raises(ValueError):
        simulate_network_contention(HDR, jobs)


def test_unknown_fabric_raises():
    with pytest.raises(ValueError):
        get_fabric("not-a-real-fabric")
