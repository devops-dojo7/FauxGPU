from fastapi import APIRouter, HTTPException

from api.schemas import NetworkContentionRequest, NetworkContentionResponse, NetworkJobResultOut
from engine.gpu_specs import get_fabric
from engine.network_contention import NetworkJob, simulate_network_contention

router = APIRouter(prefix="/network", tags=["network"])


@router.post("/contention", response_model=NetworkContentionResponse)
def contention(req: NetworkContentionRequest):
    jobs = [NetworkJob(job_id=j.job_id, team=j.team, num_gpus=j.num_gpus, payload_gb=j.payload_gb) for j in req.jobs]
    try:
        fabric = get_fabric(req.fabric_id)
        result = simulate_network_contention(fabric, jobs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return NetworkContentionResponse(
        fabric_id=result.fabric_id,
        fabric_bandwidth_gbps=result.fabric_bandwidth_gbps,
        total_gpus_sharing_fabric=result.total_gpus_sharing_fabric,
        jobs=[
            NetworkJobResultOut(
                job_id=j.job_id, team=j.team, num_gpus=j.num_gpus, bandwidth_share_gbps=j.bandwidth_share_gbps,
                isolated_comm_s=j.isolated_comm_s, contended_comm_s=j.contended_comm_s, slowdown_factor=j.slowdown_factor,
            )
            for j in result.jobs
        ],
    )
