import api.langfuse_client as langfuse_client


def test_tracing_unavailable_by_default():
    assert langfuse_client.tracing_available() is False


def test_trace_inference_request_is_a_silent_noop_when_not_configured():
    langfuse_client.trace_inference_request(
        request_id="inf-test",
        prompt="should not raise",
        model_label="llama2-7b",
        gpu_id="h100-sxm",
        precision="bf16",
        ttft_s=0.01,
        decode_step_s=0.005,
        output_tokens=10,
        tokens_per_sec=50.0,
        cost_usd=0.001,
    )  # no exception, no network call


def test_trace_inference_request_calls_client_when_configured(monkeypatch):
    calls = {"spans": [], "updates": []}

    class FakeObservation:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, **kwargs):
            calls["updates"].append(kwargs)

    class FakeClient:
        def start_as_current_observation(self, **kwargs):
            calls["spans"].append(kwargs)
            return FakeObservation()

        def flush(self):
            calls["flushed"] = True

    monkeypatch.setattr(langfuse_client, "_client", FakeClient())
    assert langfuse_client.tracing_available() is True

    langfuse_client.trace_inference_request(
        request_id="inf-test",
        prompt="hello",
        model_label="llama2-7b",
        gpu_id="h100-sxm",
        precision="bf16",
        ttft_s=0.01,
        decode_step_s=0.005,
        output_tokens=10,
        tokens_per_sec=50.0,
        cost_usd=0.001,
    )
    span_types = {kw.get("as_type") for kw in calls["spans"]}
    assert "span" in span_types
    assert "generation" in span_types
    assert calls.get("flushed") is True


def test_trace_ai_request_is_a_silent_noop_when_not_configured():
    langfuse_client.trace_ai_request(
        name="ai-chat",
        provider="openai",
        model="gpt-4o",
        input_data={"messages": []},
        output_data={"response": "should not raise"},
    )  # no exception, no network call


def test_trace_ai_request_calls_client_when_configured(monkeypatch):
    calls = {"spans": [], "updates": []}

    class FakeObservation:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, **kwargs):
            calls["updates"].append(kwargs)

    class FakeClient:
        def start_as_current_observation(self, **kwargs):
            calls["spans"].append(kwargs)
            return FakeObservation()

        def flush(self):
            calls["flushed"] = True

    monkeypatch.setattr(langfuse_client, "_client", FakeClient())

    langfuse_client.trace_ai_request(
        name="ai-recommend-nl",
        provider="openai",
        model="gpt-4o",
        input_data={"prompt": "cheap 7B run"},
        output_data={"candidates_count": 3},
    )

    assert len(calls["spans"]) == 1
    span = calls["spans"][0]
    assert span["name"] == "ai-recommend-nl"
    assert span["as_type"] == "generation"
    assert span["model"] == "gpt-4o"
    assert span["metadata"] == {"provider": "openai"}
    assert calls["updates"] == [{"output": {"candidates_count": 3}}]
    assert calls.get("flushed") is True
