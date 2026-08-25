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
async def test_stream_raises_with_key_redacted_on_http_error(monkeypatch):
    def handler(request):
        return httpx.Response(401, content=b"invalid key sk-test")

    _patch_client(monkeypatch, handler)
    with pytest.raises(RuntimeError) as exc_info:
        async for _ in ai_providers.stream_chat_completion("openai", "sk-test", "gpt-4o", [{"role": "user", "content": "hi"}]):
            pass
    assert "sk-test" not in str(exc_info.value)
    assert "***" in str(exc_info.value)
