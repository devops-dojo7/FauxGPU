"""Optional Langfuse tracing for individual simulated (or real, if pointed
at the k3s/inference-server) inference requests. Entirely optional: every
function here is a no-op if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY aren't
set, so the simulator works standalone with zero Langfuse connection
required — same shape as api/grafana_push.py's optional Grafana Cloud
integration, just for per-request LLM traces instead of infra gauges.
"""

from __future__ import annotations

import os

_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

_client = None
if _PUBLIC_KEY and _SECRET_KEY:
    from langfuse import Langfuse

    _client = Langfuse(public_key=_PUBLIC_KEY, secret_key=_SECRET_KEY, host=_HOST)


def tracing_available() -> bool:
    return _client is not None


def trace_inference_request(
    *,
    request_id: str,
    prompt: str,
    model_label: str,
    gpu_id: str,
    precision: str,
    ttft_s: float,
    decode_step_s: float,
    output_tokens: int,
    tokens_per_sec: float,
    cost_usd: float,
) -> None:
    """Best-effort: a Langfuse outage should never fail a simulated request.
    Emits one trace per inference request with a prefill span (TTFT) and a
    decode span (all output tokens), plus a generation for the simulated
    output — real Langfuse trace/span/generation concepts on top of the
    simulator's synthetic timing/content.
    """
    if _client is None:
        return
    from langfuse import propagate_attributes

    try:
        with _client.start_as_current_observation(
            name="inference-request", as_type="span", input={"prompt": prompt}
        ) as root:
            with propagate_attributes(
                session_id=request_id,
                metadata={"model": model_label, "gpu": gpu_id, "precision": precision},
            ):
                with _client.start_as_current_observation(name="prefill", as_type="span") as prefill_span:
                    prefill_span.update(output={"ttft_s": ttft_s})
                with _client.start_as_current_observation(
                    name="decode",
                    as_type="generation",
                    model=model_label,
                    usage_details={"output": output_tokens},
                    output={"tokens_per_sec": tokens_per_sec, "decode_step_s": decode_step_s},
                ):
                    pass
            root.update(output={"tokens_per_sec": tokens_per_sec, "cost_usd": cost_usd})
        _client.flush()
    except Exception:
        pass  # best-effort; never fail a request over a tracing hiccup
