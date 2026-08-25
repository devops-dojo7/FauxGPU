import pytest
from fastapi import HTTPException

from api.routers.scheduler import simulate as simulate_endpoint
from api.schemas import SchedJobIn, SchedulerRequest
from engine.scheduler import GpuPool, SchedJob, simulate_scheduler

POOL_8 = GpuPool(gpu_id="h100-sxm", total_count=8)


def test_no_contention_runs_immediately():
    jobs = [
        SchedJob(job_id="a", team="alpha", priority=1, gpu_count=4, submit_time=0, duration=10),
        SchedJob(job_id="b", team="beta", priority=1, gpu_count=4, submit_time=0, duration=5),
    ]
    result = simulate_scheduler(jobs, POOL_8)
    by_id = {j.job_id: j for j in result.jobs}
    assert by_id["a"].final_status == "completed"
    assert by_id["b"].final_status == "completed"
    assert by_id["a"].segments[0].start == 0
    assert by_id["a"].segments[0].end == 10
    assert by_id["b"].segments[0].end == 5
    assert by_id["a"].preempted_count == 0
    assert result.makespan == 10


def test_contention_without_preemption_queues_lower_priority_job():
    jobs = [
        SchedJob(job_id="big", team="alpha", priority=5, gpu_count=8, submit_time=0, duration=10),
        SchedJob(job_id="small", team="beta", priority=1, gpu_count=8, submit_time=1, duration=5),
    ]
    result = simulate_scheduler(jobs, POOL_8, preemption_enabled=False)
    by_id = {j.job_id: j for j in result.jobs}
    assert by_id["big"].final_status == "completed"
    assert by_id["big"].preempted_count == 0
    # "small" can't fit until "big" finishes at t=10, since preemption is disabled
    assert by_id["small"].segments[0].start == 10
    assert by_id["small"].wait_time_total == pytest.approx(9)  # queued from t=1 to t=10


def test_higher_priority_arrival_preempts_lower_priority_running_job():
    jobs = [
        SchedJob(job_id="low", team="beta", priority=1, gpu_count=8, submit_time=0, duration=10),
        SchedJob(job_id="high", team="alpha", priority=5, gpu_count=8, submit_time=2, duration=3),
    ]
    result = simulate_scheduler(jobs, POOL_8, preemption_enabled=True)
    by_id = {j.job_id: j for j in result.jobs}

    # "high" preempts "low" at t=2, runs t=2..5, then "low" resumes t=5 with
    # its remaining 8 units of duration (10 - 2 already run), finishing t=13.
    assert by_id["low"].preempted_count == 1
    assert len(by_id["low"].segments) == 2
    assert by_id["low"].segments[0].start == 0
    assert by_id["low"].segments[0].end == 2
    assert by_id["low"].segments[1].start == 5
    assert by_id["low"].segments[1].end == 13
    assert by_id["low"].final_status == "completed"

    assert by_id["high"].segments[0].start == 2
    assert by_id["high"].segments[0].end == 5
    assert by_id["high"].final_status == "completed"

    # Total time "low" actually ran, across both segments, equals its original duration.
    ran = sum(seg.end - seg.start for seg in by_id["low"].segments)
    assert ran == pytest.approx(10)


def test_job_that_never_fits_even_with_preemption_stays_never_started():
    jobs = [
        SchedJob(job_id="perpetual", team="alpha", priority=10, gpu_count=8, submit_time=0, duration=100),
        SchedJob(job_id="stuck", team="beta", priority=1, gpu_count=8, submit_time=1, duration=5),
    ]
    result = simulate_scheduler(jobs, POOL_8, preemption_enabled=True, horizon=50)
    by_id = {j.job_id: j for j in result.jobs}
    # "stuck" has lower priority than the perpetually-running job, so it can
    # never preempt it and never fits alongside it.
    assert by_id["stuck"].final_status == "never_started"
    assert by_id["stuck"].segments == ()


def test_simultaneous_arrivals_are_admitted_by_priority_deterministically():
    jobs = [
        SchedJob(job_id="lo", team="beta", priority=1, gpu_count=8, submit_time=0, duration=5),
        SchedJob(job_id="hi", team="alpha", priority=9, gpu_count=8, submit_time=0, duration=5),
    ]
    result = simulate_scheduler(jobs, POOL_8, preemption_enabled=True)
    by_id = {j.job_id: j for j in result.jobs}
    # Only one 8-GPU job can run at a time in an 8-GPU pool; the
    # higher-priority job should be admitted first.
    assert by_id["hi"].segments[0].start == 0
    assert by_id["lo"].segments[0].start == 5


def test_oversized_job_rejected_at_validation():
    jobs = [SchedJob(job_id="too-big", team="alpha", priority=1, gpu_count=16, submit_time=0, duration=5)]
    with pytest.raises(ValueError):
        simulate_scheduler(jobs, POOL_8)


def test_duplicate_job_ids_rejected():
    jobs = [
        SchedJob(job_id="dupe", team="alpha", priority=1, gpu_count=4, submit_time=0, duration=5),
        SchedJob(job_id="dupe", team="beta", priority=1, gpu_count=4, submit_time=0, duration=5),
    ]
    with pytest.raises(ValueError):
        simulate_scheduler(jobs, POOL_8)


def test_empty_job_list_rejected():
    with pytest.raises(ValueError):
        simulate_scheduler([], POOL_8)


def test_router_rejects_oversized_job_with_400():
    req = SchedulerRequest(
        jobs=[SchedJobIn(job_id="too-big", team="alpha", priority=1, gpu_count=16, submit_time=0, duration=5)],
        gpu_id="h100-sxm",
        total_gpus=8,
    )
    with pytest.raises(HTTPException) as exc_info:
        simulate_endpoint(req)
    assert exc_info.value.status_code == 400


def test_gpu_utilization_reflects_idle_time():
    # One 4-GPU job runs for 10 time units in an 8-GPU pool with nothing else
    # scheduled: half the pool's GPU-time over the run is idle.
    jobs = [SchedJob(job_id="a", team="alpha", priority=1, gpu_count=4, submit_time=0, duration=10)]
    result = simulate_scheduler(jobs, POOL_8)
    assert result.gpu_utilization_pct == pytest.approx(50.0)
