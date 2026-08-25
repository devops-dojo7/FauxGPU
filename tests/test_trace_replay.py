import pytest

from engine.scheduler import GpuPool, simulate_scheduler
from engine.trace_replay import load_sample_trace_csv, parse_trace_csv

WELL_FORMED_CSV = """job_id,team,priority,gpu_count,submit_time,duration
j1,team-a,5,8,0,10
j2,team-b,1,8,2,5
"""


def test_parse_trace_csv_produces_expected_jobs():
    jobs = parse_trace_csv(WELL_FORMED_CSV)
    assert len(jobs) == 2
    assert jobs[0].job_id == "j1"
    assert jobs[0].team == "team-a"
    assert jobs[0].priority == 5
    assert jobs[0].gpu_count == 8
    assert jobs[0].submit_time == 0
    assert jobs[0].duration == 10


def test_parse_trace_csv_rejects_missing_column():
    csv_text = "job_id,team,priority,gpu_count,submit_time\nj1,team-a,5,8,0\n"  # missing duration
    with pytest.raises(ValueError):
        parse_trace_csv(csv_text)


def test_parse_trace_csv_rejects_non_numeric_field():
    csv_text = "job_id,team,priority,gpu_count,submit_time,duration\nj1,team-a,five,8,0,10\n"
    with pytest.raises(ValueError):
        parse_trace_csv(csv_text)


def test_parse_trace_csv_rejects_empty_input():
    with pytest.raises(ValueError):
        parse_trace_csv("")


def test_parse_trace_csv_rejects_header_only_csv():
    with pytest.raises(ValueError):
        parse_trace_csv("job_id,team,priority,gpu_count,submit_time,duration\n")


def test_bundled_sample_trace_parses_cleanly():
    csv_text = load_sample_trace_csv()
    jobs = parse_trace_csv(csv_text)
    assert len(jobs) > 0
    ids = [j.job_id for j in jobs]
    assert len(ids) == len(set(ids))  # no duplicate job ids in the shipped sample


def test_bundled_sample_trace_replays_through_the_scheduler():
    jobs = parse_trace_csv(load_sample_trace_csv())
    max_gpus = max(j.gpu_count for j in jobs)
    pool = GpuPool(gpu_id="h100-sxm", total_count=max_gpus)
    result = simulate_scheduler(jobs, pool, preemption_enabled=True)
    assert result.makespan > 0
    assert len(result.jobs) == len(jobs)
