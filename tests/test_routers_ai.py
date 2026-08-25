import importlib
import json

import pytest
from fastapi import HTTPException

MODEL_JSON = {
    "params": 6.74e9,
    "num_layers": 32,
    "hidden_dim": 4096,
    "num_heads": 32,
    "head_dim": 128,
}


@pytest.fixture
def ai_router(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_SETTINGS_DB_PATH", str(tmp_path / "ai_settings.db"))
    monkeypatch.setenv("AI_SETTINGS_SECRET", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
    from api import ai_settings_db

    importlib.reload(ai_settings_db)
    from api.routers import ai as ai_router_module

    importlib.reload(ai_router_module)
    return ai_router_module


def test_list_providers_reports_unconfigured_by_default(ai_router):
    statuses = ai_router.list_providers()
    assert {s.provider for s in statuses} == set(ai_router.PROVIDERS)
    assert all(not s.configured for s in statuses)


def test_set_get_delete_provider_key_round_trip(ai_router):
    from api.schemas import AiProviderKeyIn

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))
    statuses = ai_router.list_providers()
    assert next(s for s in statuses if s.provider == "openai").configured is True

    ai_router.delete_provider_key("openai")
    statuses = ai_router.list_providers()
    assert next(s for s in statuses if s.provider == "openai").configured is False


def test_set_provider_key_rejects_unknown_provider(ai_router):
    from api.schemas import AiProviderKeyIn

    with pytest.raises(HTTPException) as exc_info:
        ai_router.set_provider_key("not-a-real-provider", AiProviderKeyIn(api_key="sk-test"))
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_recommend_nl_happy_path(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, RecommendNlRequest

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return json.dumps(
            {
                "model": MODEL_JSON,
                "precision": "bf16",
                "tokens_per_step": 32768,
                "total_training_tokens": 1e11,
                "objective": "cost",
                "batch_size": 1,
                "seq_len": 2048,
            }
        )

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    res = await ai_router.recommend_nl(RecommendNlRequest(provider="openai", model="gpt-4o", prompt="cheap 7B training run"))
    assert len(res.candidates) > 0


@pytest.mark.anyio
async def test_recommend_nl_traces_the_call(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, RecommendNlRequest

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return json.dumps({"model": MODEL_JSON, "objective": "cost"})

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    traced = {}
    monkeypatch.setattr(ai_router.langfuse_client, "trace_ai_request", lambda **kwargs: traced.update(kwargs))

    await ai_router.recommend_nl(RecommendNlRequest(provider="openai", model="gpt-4o", prompt="cheap 7B training run"))

    assert traced["name"] == "ai-recommend-nl"
    assert traced["provider"] == "openai"
    assert traced["model"] == "gpt-4o"
    assert traced["input_data"] == {"prompt": "cheap 7B training run"}
    assert traced["output_data"]["candidates_count"] > 0


@pytest.mark.anyio
async def test_chat_stream_traces_the_call(ai_router, monkeypatch):
    from api.schemas import ChatMessage, ChatRequest

    async def fake_stream_chat_completion(provider, api_key, model, messages):
        for chunk in ["Hel", "lo"]:
            yield chunk

    monkeypatch.setattr(ai_router, "stream_chat_completion", fake_stream_chat_completion)

    traced = {}
    monkeypatch.setattr(ai_router.langfuse_client, "trace_ai_request", lambda **kwargs: traced.update(kwargs))

    req = ChatRequest(provider="openai", model="gpt-4o", messages=[ChatMessage(role="user", content="hi")])
    events = [e async for e in ai_router._chat_stream(req.provider, "sk-test", req.model, [{"role": "user", "content": "hi"}])]

    assert [e["event"] for e in events] == ["token", "token", "done"]
    assert traced["name"] == "ai-chat"
    assert traced["provider"] == "openai"
    assert traced["model"] == "gpt-4o"
    assert traced["output_data"] == {"response": "Hello"}


@pytest.mark.anyio
async def test_recommend_nl_strips_preamble_before_fenced_json(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, RecommendNlRequest

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        body = json.dumps(
            {
                "model": MODEL_JSON,
                "precision": "bf16",
                "tokens_per_step": 32768,
                "total_training_tokens": 1e11,
                "objective": "cost",
                "batch_size": 1,
                "seq_len": 2048,
            }
        )
        return f"Sure, here's a config for that:\n\n```json\n{body}\n```\n"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    res = await ai_router.recommend_nl(RecommendNlRequest(provider="openai", model="gpt-4o", prompt="cheap 7B training run"))
    assert len(res.candidates) > 0


@pytest.mark.anyio
async def test_recommend_nl_422_on_malformed_model_output(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, RecommendNlRequest

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return "not json at all"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    with pytest.raises(HTTPException) as exc_info:
        await ai_router.recommend_nl(RecommendNlRequest(provider="openai", model="gpt-4o", prompt="cheap 7B training run"))
    assert exc_info.value.status_code == 422


@pytest.mark.anyio
async def test_recommend_nl_400_when_no_key_configured(ai_router):
    from api.schemas import RecommendNlRequest

    with pytest.raises(HTTPException) as exc_info:
        await ai_router.recommend_nl(RecommendNlRequest(provider="openai", model="gpt-4o", prompt="cheap 7B training run"))
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_provider_models_happy_path(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_list_models(provider, api_key):
        return ["gpt-4o", "gpt-4o-mini"]

    monkeypatch.setattr(ai_router, "list_models", fake_list_models)

    models = await ai_router.provider_models("openai")
    assert models == ["gpt-4o", "gpt-4o-mini"]


@pytest.mark.anyio
async def test_provider_models_400_when_no_key_configured(ai_router):
    with pytest.raises(HTTPException) as exc_info:
        await ai_router.provider_models("openai")
    assert exc_info.value.status_code == 400


@pytest.mark.anyio
async def test_provider_models_502_on_provider_failure(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn

    ai_router.set_provider_key("openai", AiProviderKeyIn(api_key="sk-test"))

    async def fake_list_models(provider, api_key):
        raise RuntimeError("boom")

    monkeypatch.setattr(ai_router, "list_models", fake_list_models)

    with pytest.raises(HTTPException) as exc_info:
        await ai_router.provider_models("openai")
    assert exc_info.value.status_code == 502


@pytest.mark.anyio
async def test_trace_generate_nl_inserts_missing_header(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, TraceGenerateNlRequest

    ai_router.set_provider_key("groq", AiProviderKeyIn(api_key="gk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        # No header row at all — observed from gpt-4o-mini in the wild.
        return "job-1,team-a,2,2,0,300\njob-2,team-b,5,1,30,600"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    res = await ai_router.trace_generate_nl(TraceGenerateNlRequest(provider="groq", model="gpt-4o-mini", prompt="a few small jobs"))
    assert res.trace_csv.startswith("job_id,team,priority,gpu_count,submit_time,duration\n")
    assert "job-1,team-a,2,2,0,300" in res.trace_csv


@pytest.mark.anyio
async def test_trace_generate_nl_happy_path(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, TraceGenerateNlRequest

    ai_router.set_provider_key("groq", AiProviderKeyIn(api_key="gk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return "job_id,team,priority,gpu_count,submit_time,duration\nj1,team-a,5,8,0,10\n"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    res = await ai_router.trace_generate_nl(TraceGenerateNlRequest(provider="groq", model="llama-3.3-70b", prompt="a small burst of jobs"))
    assert "j1,team-a" in res.trace_csv


@pytest.mark.anyio
async def test_trace_generate_nl_traces_the_call(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, TraceGenerateNlRequest

    ai_router.set_provider_key("groq", AiProviderKeyIn(api_key="gk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return "job_id,team,priority,gpu_count,submit_time,duration\nj1,team-a,5,8,0,10\n"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    traced = {}
    monkeypatch.setattr(ai_router.langfuse_client, "trace_ai_request", lambda **kwargs: traced.update(kwargs))

    await ai_router.trace_generate_nl(TraceGenerateNlRequest(provider="groq", model="llama-3.3-70b", prompt="a small burst of jobs"))

    assert traced["name"] == "ai-trace-generate-nl"
    assert traced["provider"] == "groq"
    assert traced["model"] == "llama-3.3-70b"
    assert traced["input_data"] == {"prompt": "a small burst of jobs"}
    assert "j1,team-a" in traced["output_data"]["trace_csv"]


@pytest.mark.anyio
async def test_trace_generate_nl_strips_preamble_before_fenced_csv(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, TraceGenerateNlRequest

    ai_router.set_provider_key("groq", AiProviderKeyIn(api_key="gk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return (
            "Here's a trace based on your request:\n\n"
            "```csv\n"
            "job_id,team,priority,gpu_count,submit_time,duration\n"
            "j1,team-a,5,8,0,10\n"
            "```\n"
        )

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    res = await ai_router.trace_generate_nl(TraceGenerateNlRequest(provider="groq", model="llama-3.3-70b", prompt="a small burst of jobs"))
    assert "job_id,team,priority,gpu_count,submit_time,duration" in res.trace_csv
    assert "Here's a trace" not in res.trace_csv


@pytest.mark.anyio
async def test_trace_generate_nl_422_on_malformed_csv(ai_router, monkeypatch):
    from api.schemas import AiProviderKeyIn, TraceGenerateNlRequest

    ai_router.set_provider_key("groq", AiProviderKeyIn(api_key="gk-test"))

    async def fake_complete_once(provider, api_key, model, messages):
        return "not,a,valid,trace"

    monkeypatch.setattr(ai_router, "complete_once", fake_complete_once)

    with pytest.raises(HTTPException) as exc_info:
        await ai_router.trace_generate_nl(TraceGenerateNlRequest(provider="groq", model="llama-3.3-70b", prompt="a small burst of jobs"))
    assert exc_info.value.status_code == 422
