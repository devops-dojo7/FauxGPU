from fastapi import APIRouter, HTTPException

from api.schemas import AutoscalingPointOut, AutoscalingRequest, AutoscalingResponse
from engine.autoscaling import AutoscalingConfig, TrafficStage, per_replica_capacity_rps, simulate_autoscaling
from engine.gpu_specs import get_gpu
from engine.memory import ModelShape

router = APIRouter(prefix="/autoscaling", tags=["autoscaling"])


@router.post("/simulate", response_model=AutoscalingResponse)
def simulate(req: AutoscalingRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        gpu = get_gpu(req.gpu_id)
        per_replica_rps = per_replica_capacity_rps(
            model,
            gpu,
            precision=req.precision,
            prompt_tokens=req.prompt_tokens,
            output_tokens=req.output_tokens,
            decode_batch_size=req.decode_batch_size,
            cache_hit_fraction=req.cache_hit_fraction,
            utilization=req.utilization,
            paged_attention=req.paged_attention,
            block_size=req.block_size,
        )
        stages = [TrafficStage(duration_s=s.duration_s, target_rps=s.target_rps) for s in req.stages]
        config = AutoscalingConfig(
            min_replicas=req.config.min_replicas,
            max_replicas=req.config.max_replicas,
            target_utilization_pct=req.config.target_utilization_pct,
            eval_interval_s=req.config.eval_interval_s,
            scale_down_stabilization_s=req.config.scale_down_stabilization_s,
        )
        result = simulate_autoscaling(stages, config, per_replica_rps, tick_s=req.tick_s)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return AutoscalingResponse(
        points=[
            AutoscalingPointOut(
                t=p.t, demand_rps=p.demand_rps, replicas=p.replicas, capacity_rps=p.capacity_rps,
                backlog_requests=p.backlog_requests, est_queue_delay_s=p.est_queue_delay_s,
            )
            for p in result.points
        ],
        per_replica_capacity_rps=result.per_replica_capacity_rps,
        peak_replicas=result.peak_replicas,
        peak_backlog_requests=result.peak_backlog_requests,
        peak_queue_delay_s=result.peak_queue_delay_s,
    )
