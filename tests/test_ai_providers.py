import json

import httpx
import pytest

from api import ai_providers


def _openai_sse(deltas: list[str]) -> bytes:
    lines = []
    for d in deltas:
        lines.append(f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def _anthropic_sse(deltas: list[str]) -> bytes:
    lines = []
    for d in deltas:
        chunk = {"type": "content_block_delta", "delta": {"text": d}}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    return "".join(lines).encode()


def _patch_client(monkeypatch, handler):
    class _MockAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(ai_providers.httpx, "AsyncClient", _MockAsyncClient)


@pytest.mark.anyio
async def test_stream_openai_compatible_yields_text_deltas(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, content=_openai_sse(["Hel", "lo"]))

    _patch_client(monkeypatch, handler)
    deltas = [d async for d in ai_providers.stream_chat_completion("openai", "sk-test", "gpt-4o", [{"role": "user", "content": "hi"}])]
    assert deltas == ["Hel", "lo"]


@pytest.mark.anyio
async def test_stream_anthropic_yields_text_deltas(monkeypatch):
    def handler(request):
        assert request.headers["x-api-key"] == "sk-ant-test"
        return httpx.Response(200, content=_anthropic_sse(["Hi", " there"]))

    _patch_client(monkeypatch, handler)
    deltas = [
        d async for d in ai_providers.stream_chat_completion("anthropic", "sk-ant-test", "claude-sonnet", [{"role": "user", "content": "hi"}])
    ]
    assert deltas == ["Hi", " there"]


@pytest.mark.anyio
async def test_complete_once_joins_deltas(monkeypatch):
    def handler(request):
        return httpx.Response(200, content=_openai_sse(["foo", "bar"]))

    _patch_client(monkeypatch, handler)
    text = await ai_providers.complete_once("groq", "gk-test", "llama-3.3-70b", [{"role": "user", "content": "hi"}])
    assert text == "foobar"


@pytest.mark.anyio
async def test_list_models_returns_sorted_ids(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer sk-test"
        return httpx.Response(200, json={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}, {"id": "gpt-3.5-turbo"}]})

    _patch_client(monkeypatch, handler)
    models = await ai_providers.list_models("openai", "sk-test")
    assert models == ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]


@pytest.mark.anyio
async def test_list_models_anthropic_uses_x_api_key_header(monkeypatch):
    def handler(request):
        assert request.headers["x-api-key"] == "sk-ant-test"
        return httpx.Response(200, json={"data": [{"id": "claude-sonnet-4-5"}]})

    _patch_client(monkeypatch, handler)
    models = await ai_providers.list_models("anthropic", "sk-ant-test")
    assert models == ["claude-sonnet-4-5"]


@pytest.mark.anyio
async def test_list_models_raises_with_key_redacted_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(401, content=b"invalid key sk-test")

    _patch_client(monkeypatch, handler)
    with pytest.raises(RuntimeError) as exc_info:
        await ai_providers.list_models("openai", "sk-test")
    assert "sk-test" not in str(exc_info.value)


@pytest.mark.anyio
async def test_stream_raises_with_key_redacted_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(401, content=b"invalid key sk-test")

    _patch_client(monkeypatch, handler)
    with pytest.raises(RuntimeError) as exc_info:
        async for _ in ai_providers.stream_chat_completion("openai", "sk-test", "gpt-4o", [{"role": "user", "content": "hi"}]):
            pass
    assert "sk-test" not in str(exc_info.value)
    assert "***" in str(exc_info.value)
