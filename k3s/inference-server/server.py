"""A real, long-running inference server (unlike the one-shot trainer Job)
exposing an OpenAI-Completions-shaped endpoint backed by the same synthetic
engine/inference.py timing the website uses. The point: a *real* HTTP
client (the `openai` SDK, curl, anything) can point at this and get a real
network round trip against the mock-GPU-backed cluster — the response
content/timing is still simulated, same philosophy as k3s/trainer/train.py
being a real scheduled pod with synthetic step timing rather than real
compute.

Every request is traced via api.langfuse_client (a no-op unless
LANGFUSE_PUBLIC_KEY/SECRET_KEY are set) so a real client's requests
against this server produce real Langfuse traces flowing through
simulated infra.
"""

import os
import sys
import time
import uuid

sys.path.insert(0, "/app")

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from api import langfuse_client  # noqa: E402
from engine.gpu_specs import get_gpu  # noqa: E402
from engine.inference import decode_step_seconds, prefill_seconds  # noqa: E402
from engine.memory import ModelShape  # noqa: E402

MODEL_PRESETS = {
    "llama2-7b": ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128),
    "llama2-13b": ModelShape(params=13.0e9, num_layers=40, hidden_dim=5120, num_heads=40, head_dim=128),
    "llama2-70b": ModelShape(params=70.0e9, num_layers=80, hidden_dim=8192, num_heads=64, head_dim=128),
}

FILLER_WORDS = (
    "the model considers each token in context, attending to prior state before predicting the next most "
    "likely continuation of the sequence based on patterns learned during training across a broad corpus "
    "of text data and code"
).split(" ")


def env(key: str, default: str) -> str:
    return os.environ.get(key, default)


MODEL_LABEL = env("MODEL_PRESET", "llama2-7b")
MODEL = MODEL_PRESETS.get(MODEL_LABEL, MODEL_PRESETS["llama2-7b"])
GPU_ID = env("SIMGPU_MODEL", "h100-sxm")
PRECISION = env("PRECISION", "bf16")
GPU = get_gpu(GPU_ID)

app = FastAPI(title="simgpu inference-server")


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 80
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_LABEL, "gpu": GPU_ID}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    prompt_tokens = max(1, len(req.prompt) // 4)  # same ~4 chars/token heuristic as the website playground

    ttft_s = prefill_seconds(MODEL, prompt_tokens, GPU, PRECISION)
    time.sleep(ttft_s)

    words = []
    total_decode_s = 0.0
    last_step_s = 0.0
    for i in range(req.max_tokens):
        last_step_s = decode_step_seconds(MODEL, GPU, PRECISION, batch_size=1, avg_kv_tokens=prompt_tokens + i)
        time.sleep(last_step_s)
        total_decode_s += last_step_s
        words.append(FILLER_WORDS[(len(words) + prompt_tokens) % len(FILLER_WORDS)])

    text = " ".join(words)
    tokens_per_sec = req.max_tokens / total_decode_s if total_decode_s > 0 else 0.0
    cost_usd = (ttft_s + total_decode_s) / 3600.0 * GPU.price_per_hr_usd

    langfuse_client.trace_inference_request(
        request_id=request_id,
        prompt=req.prompt,
        model_label=MODEL_LABEL,
        gpu_id=GPU_ID,
        precision=PRECISION,
        ttft_s=ttft_s,
        decode_step_s=last_step_s,
        output_tokens=req.max_tokens,
        tokens_per_sec=tokens_per_sec,
        cost_usd=cost_usd,
    )

    return {
        "id": request_id,
        "object": "text_completion",
        "model": req.model or MODEL_LABEL,
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": req.max_tokens, "total_tokens": prompt_tokens + req.max_tokens},
        "simgpu": {"ttft_s": ttft_s, "tokens_per_sec": tokens_per_sec, "cost_usd": cost_usd},
    }
