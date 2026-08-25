import pytest
from fastapi import HTTPException

from api.routers.scheduler import replay_trace, sample_trace
from api.schemas import TraceReplayRequest
from engine.trace_replay import parse_trace_csv

WELL_FORMED_CSV = """job_id,team,priority,gpu_count,submit_time,duration
j1,team-a,5,8,0,10
j2,team-b,1,8,2,5
"""


def test_replay_trace_happy_path_matches_row_count():
    req = TraceReplayRequest(trace_csv=WELL_FORMED_CSV, gpu_id="h100-sxm", total_gpus=16)
    res = replay_trace(req)
    assert len(res.jobs) == 2
    assert res.pool_total_gpus == 16


def test_replay_trace_400_on_malformed_csv():
    req = TraceReplayRequest(trace_csv="not,a,valid,trace", gpu_id="h100-sxm", total_gpus=16)
    with pytest.raises(HTTPException) as exc_info:
        replay_trace(req)
    assert exc_info.value.status_code == 400


def test_sample_trace_endpoint_returns_parseable_csv():
    csv_text = sample_trace()
    assert len(csv_text) > 0
    jobs = parse_trace_csv(csv_text)
    assert len(jobs) > 0
