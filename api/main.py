import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api import grafana_push, langfuse_client
from api.metrics import render_prometheus_metrics
from api.routers import ai, autoscaling, calculate, gpus, inference_stream, mig, network, runs, scheduler, topology
from api.runs_store import store

PUSH_INTERVAL_S = 2  # short enough that live runs visibly pulse between compute/comm phases on a dashboard


async def _grafana_push_loop():
    """Background task: while Grafana remote-write is configured and at
    least one run is active, push current gauges every few seconds. A
    Grafana outage never affects the simulator itself — push failures are
    swallowed inside grafana_push.push_run_metrics().
    """
    if not grafana_push.push_available():
        return
    while True:
        await asyncio.sleep(PUSH_INTERVAL_S)
        if any(r.status == "running" for r in store.list()):
            try:
                grafana_push.push_run_metrics()
            except Exception:
                pass  # best-effort; never crash the app over a push failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_grafana_push_loop())
    yield
    task.cancel()


app = FastAPI(title="FauxGPU API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    # Local teaching tool, no auth/cookies — open CORS so the website works
    # whether it's reached via `next dev`, a k8s port-forward, or a NodePort.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(gpus.router)
app.include_router(calculate.router)
app.include_router(topology.router)
app.include_router(runs.router)
app.include_router(inference_stream.router)
app.include_router(scheduler.router)
app.include_router(mig.router)
app.include_router(autoscaling.router)
app.include_router(network.router)
app.include_router(ai.router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=render_prometheus_metrics(), media_type="text/plain; version=0.0.4")


@app.get("/grafana/status")
def grafana_status():
    return {"push_available": grafana_push.push_available(), "annotations_available": grafana_push.annotations_available()}


@app.post("/grafana/push-now")
def grafana_push_now():
    samples_sent = grafana_push.push_run_metrics()
    return {"samples_sent": samples_sent}


@app.get("/langfuse/status")
def langfuse_status():
    return {
        "tracing_available": langfuse_client.tracing_available(),
        # The browser needs the host-facing URL, not LANGFUSE_HOST (which
        # points at the internal docker-network hostname the API uses).
        "public_url": os.environ.get("LANGFUSE_PUBLIC_URL", "http://localhost:3002"),
    }
