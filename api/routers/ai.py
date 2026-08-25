"""Bring-your-own-key AI features: a chat assistant, a natural-language front
end on the GPU recommender, and a natural-language job-trace generator. The
AI layer is a thin translation shim over the existing deterministic engine —
it never fabricates a recommendation or a trace on its own; it turns English
into the structured input the already-tested engine.recommend and
engine.trace_replay code paths expect, then those produce the real answer.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api import ai_settings_db
from api.ai_providers import PROVIDERS, complete_once, list_models, stream_chat_completion
from api.schemas import (
    AiProviderKeyIn,
    AiProviderStatus,
    ChatRequest,
    RecommendationCandidateOut,
    RecommendNlRequest,
    RecommendResponse,
    TraceGenerateNlRequest,
    TraceGenerateNlResponse,
)
from engine.memory import ModelShape
from engine.recommend import recommend_configurations
from engine.trace_replay import parse_trace_csv

router = APIRouter(prefix="/ai", tags=["ai"])

_SYSTEM_PROMPT = (
    "You are the embedded assistant for ErsatzGPU, a browser-based GPU cluster training/inference "
    "simulator with no real GPU required. It has tabs for Training (VRAM/step-time/cost), Inference "
    "(llm-d serving, prefill/decode, PagedAttention, speculative decoding), Datacenter (scaled "
    "cluster topology), Compare GPUs/Models, a What-if Recommender, a Cost Dashboard, a "
    "multi-tenant Scheduler, MIG Planner, Autoscaling (HPA-style), Network Contention, and Trace "
    "Replay. Help the user understand GPU training/inference concepts and how to use the simulator. "
    "Be concise."
)

_RECOMMEND_NL_SYSTEM_PROMPT = (
    "Extract training-run parameters from the user's request and reply with ONLY a JSON object "
    "(no markdown fences, no prose) with these fields: "
    '{"model": {"params": <float, total param count e.g. 7e9>, "num_layers": <int>, '
    '"hidden_dim": <int>, "num_heads": <int>, "head_dim": <int>, "num_kv_heads": <int or null>, '
    '"active_params": <float or null, MoE only>, "kv_latent_dim": <int or null, MLA only>}, '
    '"precision": <"bf16"|"fp16"|"fp8"|"fp32">, "tokens_per_step": <int>, '
    '"total_training_tokens": <float>, "objective": <"cost"|"time">, "batch_size": <int>, '
    '"seq_len": <int>}. If the user names a well-known model (e.g. "Llama 3 8B", "Qwen2.5 7B"), use '
    "its publicly known architecture. Pick sensible defaults for anything unspecified."
)

_TRACE_NL_SYSTEM_PROMPT = (
    "Generate a job-queue trace as CSV with ONLY these columns, no markdown fences, no prose: "
    "job_id,team,priority,gpu_count,submit_time,duration\n"
    "priority is an integer (higher = more important), gpu_count is an integer, submit_time and "
    "duration are in seconds (floats ok). Example rows:\n"
    "job-1,team-a,5,8,0,3600\n"
    "job-2,team-b,3,4,120,1800\n"
    "Generate a trace matching the user's description, with a realistic mix of job sizes and "
    "arrival times."
)


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _require_key(provider: str) -> str:
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}. Known: {list(PROVIDERS)}")
    try:
        key = ai_settings_db.get_key(provider)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not key:
        raise HTTPException(status_code=400, detail=f"No API key configured for provider {provider!r}")
    return key


@router.get("/providers", response_model=list[AiProviderStatus])
def list_providers():
    configured = ai_settings_db.list_configured()
    return [AiProviderStatus(provider=p, configured=p in configured) for p in PROVIDERS]


@router.put("/providers/{provider}/key")
def set_provider_key(provider: str, req: AiProviderKeyIn):
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}. Known: {list(PROVIDERS)}")
    try:
        ai_settings_db.set_key(provider, req.api_key)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}


@router.delete("/providers/{provider}/key")
def delete_provider_key(provider: str):
    if provider not in PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider!r}. Known: {list(PROVIDERS)}")
    ai_settings_db.delete_key(provider)
    return {"ok": True}


@router.get("/providers/{provider}/models", response_model=list[str])
async def provider_models(provider: str):
    api_key = _require_key(provider)
    try:
        return await list_models(provider, api_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Couldn't list models: {e}") from e


async def _chat_stream(provider: str, api_key: str, model: str, messages: list[dict[str, str]]):
    try:
        async for delta in stream_chat_completion(provider, api_key, model, messages):
            yield {"event": "token", "data": json.dumps({"delta": delta})}
    except Exception as e:
        yield {"event": "error", "data": json.dumps({"detail": str(e)})}
        return
    yield {"event": "done", "data": "{}"}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    api_key = _require_key(req.provider)
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + [
        {"role": m.role, "content": m.content} for m in req.messages
    ]
    return EventSourceResponse(_chat_stream(req.provider, api_key, req.model, messages))


@router.post("/recommend/nl", response_model=RecommendResponse)
async def recommend_nl(req: RecommendNlRequest):
    api_key = _require_key(req.provider)
    messages = [
        {"role": "system", "content": _RECOMMEND_NL_SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt},
    ]
    try:
        raw = await complete_once(req.provider, api_key, req.model, messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Provider call failed: {e}") from e

    try:
        parsed = json.loads(_strip_code_fence(raw))
        model = ModelShape(**parsed["model"])
        candidates = recommend_configurations(
            model,
            precision=parsed.get("precision", "bf16"),
            tokens_per_step=int(parsed.get("tokens_per_step", 32768)),
            total_training_tokens=float(parsed.get("total_training_tokens", 1e12)),
            objective=parsed.get("objective", "cost"),
            batch_size=int(parsed.get("batch_size", 1)),
            seq_len=int(parsed.get("seq_len", 2048)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise HTTPException(
            status_code=422, detail="The model's response couldn't be parsed as a config — try rephrasing."
        ) from e

    return RecommendResponse(candidates=[RecommendationCandidateOut(**c.__dict__) for c in candidates])


@router.post("/trace/generate", response_model=TraceGenerateNlResponse)
async def trace_generate_nl(req: TraceGenerateNlRequest):
    api_key = _require_key(req.provider)
    messages = [
        {"role": "system", "content": _TRACE_NL_SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt},
    ]
    try:
        raw = await complete_once(req.provider, api_key, req.model, messages)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Provider call failed: {e}") from e

    trace_csv = _strip_code_fence(raw)
    try:
        parse_trace_csv(trace_csv)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Generated trace failed validation: {e}") from e

    return TraceGenerateNlResponse(trace_csv=trace_csv)
