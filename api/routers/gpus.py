from fastapi import APIRouter

from engine.gpu_specs import FABRICS, GPUS

router = APIRouter(prefix="/gpus", tags=["gpus"])


@router.get("")
def list_gpus():
    return list(GPUS.values())


@router.get("/fabrics")
def list_fabrics():
    return list(FABRICS.values())
