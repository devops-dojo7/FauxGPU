from fastapi import APIRouter, HTTPException

from api.schemas import (
    JobOutcomeOut,
    SchedulerRequest,
    SchedulerResponse,
    TimelineSegmentOut,
)
from engine.scheduler import GpuPool, SchedJob, simulate_scheduler

router = APIRouter(prefix="/scheduler", tags=["scheduler"])


@router.post("/simulate", response_model=SchedulerResponse)
def simulate(req: SchedulerRequest):
    jobs = [SchedJob(**j.model_dump()) for j in req.jobs]
    pool = GpuPool(gpu_id=req.gpu_id, total_count=req.total_gpus)
    try:
        result = simulate_scheduler(
            jobs, pool, preemption_enabled=req.preemption_enabled, horizon=req.horizon
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return SchedulerResponse(
        jobs=[
            JobOutcomeOut(
                job_id=j.job_id,
                team=j.team,
                segments=[TimelineSegmentOut(start=s.start, end=s.end) for s in j.segments],
                final_status=j.final_status,
                wait_time_total=j.wait_time_total,
                preempted_count=j.preempted_count,
            )
            for j in result.jobs
        ],
        pool_total_gpus=result.pool_total_gpus,
        makespan=result.makespan,
        gpu_utilization_pct=result.gpu_utilization_pct,
    )
