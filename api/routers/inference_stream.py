"""Streams a single simulated inference request token-by-token over SSE,
using the same prefill/decode formulas engine/inference.py already exposes
through the one-shot POST /calculate/inference. This is what gives the
website's playgrounds (and Langfuse tracing, see api/langfuse_client.py) a
real backend round trip per request instead of a purely client-side
setTimeout simulation — prompt text and timing now actually reach the API.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api import k8s_launcher, langfuse_client
from api.schemas import (
    InferenceStreamRequest,
    K8sAvailabilityResponse,
    LaunchInferenceServerRequest,
    LaunchInferenceServerResponse,
)
from engine.gpu_specs import get_gpu
from engine.inference import decode_step_seconds, prefill_seconds
from engine.memory import ModelShape

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get("/k8s-available", response_model=K8sAvailabilityResponse)
def inference_k8s_available():
    return K8sAvailabilityResponse(available=k8s_launcher.inference_available())


@router.post("/launch-k8s-server", response_model=LaunchInferenceServerResponse)
def launch_inference_k8s_server(req: LaunchInferenceServerRequest):
    try:
        name = k8s_launcher.launch_inference_job(req.model_preset, req.gpu_id, req.precision)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:  # k8s client errors, RBAC denials, etc.
        raise HTTPException(status_code=502, detail=f"Failed to create inference server: {e}") from e
    return LaunchInferenceServerResponse(name=name)


async def _stream(req: InferenceStreamRequest):
    request_id = f"inf-{uuid.uuid4().hex[:8]}"
    model = ModelShape(**req.model.model_dump())

    try:
        gpu = get_gpu(req.gpu_id)
        ttft_s = prefill_seconds(
            model, req.prompt_tokens, gpu, req.precision, req.utilization, req.cache_hit_fraction, req.tp_degree
        )
        if ttft_s == float("inf"):
            raise ValueError(f"Tensor parallelism (TP={req.tp_degree}) needs NVLink, but {gpu.name} has none.")
        # decode_step_seconds below shares this exact (tp_degree, gpu.nvlink_gbps)
        # feasibility gate, so it can never go inf (and asyncio.sleep it forever)
        # once ttft_s has already come back finite.
    except ValueError as e:
        yield {"event": "error", "data": json.dumps({"detail": str(e)})}
        return

    yield {"event": "start", "data": json.dumps({"request_id": request_id})}
    await asyncio.sleep(ttft_s)
    yield {"event": "ttft", "data": json.dumps({"ttft_s": ttft_s})}

    total_decode_s = 0.0
    last_step_s = 0.0
    for i in range(req.max_output_tokens):
        kv_tokens = req.prompt_tokens + i
        last_step_s = decode_step_seconds(
            model, gpu, req.precision, batch_size=1, avg_kv_tokens=kv_tokens, tp_degree=req.tp_degree
        )
        await asyncio.sleep(last_step_s)
        total_decode_s += last_step_s
        yield {"event": "token", "data": json.dumps({"index": i + 1, "step_s": last_step_s})}

    tokens_per_sec = req.max_output_tokens / total_decode_s if total_decode_s > 0 else 0.0
    cost_usd = (ttft_s + total_decode_s) / 3600.0 * gpu.price_per_hr_usd

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "request_id": request_id,
                "ttft_s": ttft_s,
                "output_tokens": req.max_output_tokens,
                "tokens_per_sec": tokens_per_sec,
                "cost_usd": cost_usd,
            }
        ),
    }

    langfuse_client.trace_inference_request(
        request_id=request_id,
        prompt=req.prompt,
        model_label=req.model_label,
        gpu_id=req.gpu_id,
        precision=req.precision,
        ttft_s=ttft_s,
        decode_step_s=last_step_s,
        output_tokens=req.max_output_tokens,
        tokens_per_sec=tokens_per_sec,
        cost_usd=cost_usd,
    )


@router.post("/stream")
async def inference_stream(req: InferenceStreamRequest):
    return EventSourceResponse(_stream(req))
