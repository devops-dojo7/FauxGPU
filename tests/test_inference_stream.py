import json

import pytest

from api.routers.inference_stream import _stream
from api.schemas import InferenceStreamRequest, ModelShapeIn

MODEL = ModelShapeIn(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)


async def _collect(req: InferenceStreamRequest) -> list[tuple[str, dict]]:
    return [(evt["event"], json.loads(evt["data"])) for evt in [e async for e in _stream(req)]]


@pytest.mark.anyio
async def test_stream_emits_start_ttft_tokens_and_done():
    req = InferenceStreamRequest(model=MODEL, gpu_id="h100-sxm", prompt="hi", prompt_tokens=4, max_output_tokens=3)
    events = await _collect(req)

    names = [name for name, _ in events]
    assert names == ["start", "ttft", "token", "token", "token", "done"]

    _, start_data = events[0]
    assert start_data["request_id"].startswith("inf-")

    token_indices = [data["index"] for name, data in events if name == "token"]
    assert token_indices == [1, 2, 3]

    _, done_data = events[-1]
    assert done_data["output_tokens"] == 3
    assert done_data["tokens_per_sec"] > 0
    assert done_data["cost_usd"] > 0


@pytest.mark.anyio
async def test_stream_emits_error_for_unknown_gpu():
    req = InferenceStreamRequest(model=MODEL, gpu_id="not-a-real-gpu", prompt="hi", prompt_tokens=4, max_output_tokens=3)
    events = await _collect(req)

    assert len(events) == 1
    name, data = events[0]
    assert name == "error"
    assert "not-a-real-gpu" in data["detail"]


@pytest.mark.anyio
async def test_stream_emits_error_for_tensor_parallel_without_nvlink():
    req = InferenceStreamRequest(
        model=MODEL, gpu_id="l40s", prompt="hi", prompt_tokens=4, max_output_tokens=3, tp_degree=4
    )
    events = await _collect(req)

    assert len(events) == 1
    name, data = events[0]
    assert name == "error"
    assert "TP=4" in data["detail"]
    assert "NVLink" in data["detail"]


@pytest.mark.anyio
async def test_stream_zero_output_tokens_emits_no_token_events():
    req = InferenceStreamRequest(model=MODEL, gpu_id="h100-sxm", prompt="hi", prompt_tokens=4, max_output_tokens=0)
    events = await _collect(req)

    names = [name for name, _ in events]
    assert names == ["start", "ttft", "done"]
    _, done_data = events[-1]
    assert done_data["tokens_per_sec"] == 0.0
