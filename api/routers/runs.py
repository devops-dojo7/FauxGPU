import asyncio
import time
import uuid

from fastapi import APIRouter, HTTPException

from api import grafana_push, k8s_launcher
from api.runs_store import store
from api.schemas import (
    K8sAvailabilityResponse,
    LaunchK8sJobResponse,
    RunDetail,
    RunStartRequest,
    RunStepRequest,
    RunSummary,
    SimulateRunRequest,
)
from engine.compute import estimate_step_time
from engine.memory import ModelShape
from engine.topology import build_topology

router = APIRouter(prefix="/runs", tags=["runs"])


def _to_summary(run) -> RunSummary:
    return RunSummary(
        run_id=run.run_id,
        status=run.status,
        meta=run.meta,
        latest_step=run.steps[-1] if run.steps else None,
        started_at=run.started_at,
        updated_at=run.updated_at,
    )


@router.post("/{run_id}/start", response_model=RunSummary)
def start_run(run_id: str, req: RunStartRequest):
    run = store.start(run_id, req.model_dump())
    grafana_push.post_annotation(
        f"simgpu: training run {run_id} started — {req.model} on {req.gpu} × {req.total_gpus} ({req.topology})",
        tags=["simgpu", "training-start"],
    )
    return _to_summary(run)


@router.post("/{run_id}/step", response_model=RunSummary)
def add_step(run_id: str, req: RunStepRequest):
    run = store.add_step(run_id, req.model_dump())
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id!r}")
    return _to_summary(run)


@router.post("/{run_id}/done", response_model=RunSummary)
def finish_run(run_id: str):
    run = store.finish(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id!r}")
    latest = run.steps[-1] if run.steps else None
    tokens_seen = latest["tokens_seen"] if latest else 0
    grafana_push.post_annotation(f"simgpu: training run {run_id} finished — {tokens_seen} tokens", tags=["simgpu", "training-done"])
    return _to_summary(run)


@router.get("", response_model=list[RunSummary])
def list_runs():
    return [_to_summary(r) for r in store.list()]


@router.get("/k8s-available", response_model=K8sAvailabilityResponse)
def k8s_available():
    return K8sAvailabilityResponse(available=k8s_launcher.is_available())


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str):
    run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown run: {run_id!r}")
    return RunDetail(
        run_id=run.run_id,
        status=run.status,
        meta=run.meta,
        latest_step=run.steps[-1] if run.steps else None,
        started_at=run.started_at,
        updated_at=run.updated_at,
        steps=run.steps,
    )


async def _run_simulation(run_id: str, model: ModelShape, topo, req: SimulateRunRequest) -> None:
    """Runs entirely inside the API process — no k8s Job needed. Lets the
    website trigger extra training runs on demand (any model/topology combo)
    without requiring kubectl/helm access.
    """
    step = estimate_step_time(
        model, topo, tokens_per_step=req.tokens_per_step, precision=req.precision, utilization=req.utilization
    )
    store.start(
        run_id,
        {
            "model": req.model_label,
            "gpu": topo.gpu.id,
            "topology": topo.shape,
            "total_gpus": topo.total_gpus,
            "compute_s_per_step": round(step.compute_s, 4),
            "communication_s_per_step": round(step.communication_s, 4),
            "total_s_per_step": round(step.total_s, 4),
            "total_steps": req.total_steps,
        },
    )
    grafana_push.post_annotation(
        f"simgpu: simulated training run {run_id} started — {req.model_label} on {topo.gpu.id} × {topo.total_gpus} ({topo.shape})",
        tags=["simgpu", "training-start", "in-process"],
    )
    start = time.time()
    tokens_seen = 0
    sleep_s = max(0.0, step.total_s / max(req.speedup, 0.001))
    for i in range(1, req.total_steps + 1):
        await asyncio.sleep(sleep_s)
        tokens_seen += req.tokens_per_step
        store.add_step(run_id, {"step": i, "tokens_seen": tokens_seen, "elapsed_s": round(time.time() - start, 2)})
    store.finish(run_id)
    grafana_push.post_annotation(
        f"simgpu: simulated training run {run_id} finished — {tokens_seen} tokens", tags=["simgpu", "training-done", "in-process"]
    )


@router.post("/simulate", response_model=RunSummary)
async def simulate_run(req: SimulateRunRequest):
    model = ModelShape(**req.model.model_dump())
    try:
        topo = build_topology(
            shape=req.topology.shape,
            gpu_id=req.topology.gpu_id,
            gpus_per_node=req.topology.gpus_per_node,
            num_nodes=req.topology.num_nodes,
            fabric_id=req.topology.fabric_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    run_id = f"sim-{uuid.uuid4().hex[:8]}"
    asyncio.create_task(_run_simulation(run_id, model, topo, req))
    await asyncio.sleep(0.05)  # let the task register the run before we respond
    run = store.get(run_id)
    if run is None:
        return RunSummary(run_id=run_id, status="running", meta=None, latest_step=None, started_at=time.time(), updated_at=time.time())
    return _to_summary(run)


@router.post("/launch-k8s-job", response_model=LaunchK8sJobResponse)
def launch_k8s_job(req: SimulateRunRequest):
    try:
        job_name = k8s_launcher.launch_job(
            model_label=req.model_label,
            model_params=req.model.model_dump(),
            gpu_id=req.topology.gpu_id,
            topology_shape=req.topology.shape,
            gpus_per_node=req.topology.gpus_per_node,
            num_nodes=req.topology.num_nodes,
            fabric_id=req.topology.fabric_id,
            precision=req.precision,
            tokens_per_step=req.tokens_per_step,
            total_steps=req.total_steps,
            speedup=req.speedup,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:  # k8s client errors, RBAC denials, etc.
        raise HTTPException(status_code=502, detail=f"Failed to create Job: {e}") from e

    return LaunchK8sJobResponse(job_name=job_name, run_id=job_name)
