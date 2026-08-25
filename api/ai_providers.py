"""Thin, provider-agnostic chat-completion client for the bring-your-own-key
AI features (chat assistant, NL recommender, NL trace generator). Six of the
seven candidate providers speak the same OpenAI-compatible chat/completions
shape; only Anthropic's Messages API differs, so there are exactly two
adapters, dispatched by `style`.

This module never decides what the AI features *do* with the model's
output — routers/ai.py owns that (e.g. validating NL-recommend output
against the real engine before trusting it). This module's only job is:
given messages, get text out, streamed or all at once.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    style: str  # "openai" | "anthropic"


PROVIDERS: dict[str, ProviderConfig] = {
    "anthropic": ProviderConfig("https://api.anthropic.com/v1/messages", "anthropic"),
    "openai": ProviderConfig("https://api.openai.com/v1/chat/completions", "openai"),
    "deepseek": ProviderConfig("https://api.deepseek.com/v1/chat/completions", "openai"),
    "kimi": ProviderConfig("https://api.moonshot.cn/v1/chat/completions", "openai"),
    "groq": ProviderConfig("https://api.groq.com/openai/v1/chat/completions", "openai"),
    "nvidia_nim": ProviderConfig("https://integrate.api.nvidia.com/v1/chat/completions", "openai"),
    "openrouter": ProviderConfig("https://openrouter.ai/api/v1/chat/completions", "openai"),
}


def _redact(api_key: str, text: str) -> str:
    return text.replace(api_key, "***") if api_key else text


async def _stream_openai_compatible(
    cfg: ProviderConfig, api_key: str, model: str, messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "stream": True}
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        async with client.stream("POST", cfg.base_url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                raise RuntimeError(_redact(api_key, f"Provider request failed ({resp.status_code}): {raw.decode(errors='replace')[:500]}"))
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    yield delta


async def _stream_anthropic(
    cfg: ProviderConfig, api_key: str, model: str, messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
    turns = [m for m in messages if m["role"] != "system"]
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    body: dict = {"model": model, "messages": turns, "max_tokens": 4096, "stream": True}
    if system:
        body["system"] = system
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        async with client.stream("POST", cfg.base_url, headers=headers, json=body) as resp:
            if resp.status_code >= 400:
                raw = await resp.aread()
                raise RuntimeError(_redact(api_key, f"Provider request failed ({resp.status_code}): {raw.decode(errors='replace')[:500]}"))
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {}).get("text")
                    if delta:
                        yield delta


def stream_chat_completion(
    provider: str, api_key: str, model: str, messages: list[dict[str, str]]
) -> AsyncIterator[str]:
    cfg = PROVIDERS[provider]
    if cfg.style == "anthropic":
        return _stream_anthropic(cfg, api_key, model, messages)
    return _stream_openai_compatible(cfg, api_key, model, messages)


async def complete_once(provider: str, api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    parts = [chunk async for chunk in stream_chat_completion(provider, api_key, model, messages)]
    return "".join(parts)
