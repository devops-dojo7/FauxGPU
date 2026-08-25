from fastapi import APIRouter, HTTPException

from api.schemas import (
    MigGpuOut,
    MigPackRequest,
    MigPackResponse,
    MigPlacementOut,
    MigProfileOut,
    MigRequestIn,
)
from engine.gpu_specs import get_gpu
from engine.mig import MIG_PROFILES, MigRequest, list_mig_profiles, pack_mig_requests, supports_mig

router = APIRouter(prefix="/mig", tags=["mig"])


@router.get("/gpus", response_model=list[MigGpuOut])
def list_mig_gpus():
    return [MigGpuOut(gpu_id=gpu_id, gpu_name=get_gpu(gpu_id).name) for gpu_id in sorted(MIG_PROFILES)]


@router.get("/profiles/{gpu_id}", response_model=list[MigProfileOut])
def get_mig_profiles(gpu_id: str):
    if not supports_mig(gpu_id):
        raise HTTPException(status_code=404, detail=f"{gpu_id!r} does not support MIG. Known: {sorted(MIG_PROFILES)}")
    return [MigProfileOut(id=p.id, compute_slots=p.compute_slots, memory_slots=p.memory_slots, memory_gb=p.memory_gb) for p in list_mig_profiles(gpu_id)]


@router.post("/pack", response_model=MigPackResponse)
def pack(req: MigPackRequest):
    requests = [MigRequest(request_id=r.request_id, tenant=r.tenant, profile_id=r.profile_id) for r in req.requests]
    try:
        result = pack_mig_requests(req.gpu_id, req.pool_size, requests)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return MigPackResponse(
        placements=[
            MigPlacementOut(
                request_id=p.request_id,
                tenant=p.tenant,
                profile_id=p.profile_id,
                gpu_index=p.gpu_index,
                compute_slots=p.compute_slots,
                memory_slots=p.memory_slots,
            )
            for p in result.placements
        ],
        unplaced=[MigRequestIn(request_id=r.request_id, tenant=r.tenant, profile_id=r.profile_id) for r in result.unplaced],
        pool_size=result.pool_size,
        gpus_used=result.gpus_used,
        compute_utilization_pct=result.compute_utilization_pct,
        memory_utilization_pct=result.memory_utilization_pct,
    )
