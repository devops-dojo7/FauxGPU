import api.grafana_push as grafana_push
from api.runs_store import RunsStore


def test_push_unavailable_by_default():
    assert grafana_push.push_available() is False


def test_annotations_unavailable_by_default():
    assert grafana_push.annotations_available() is False


def test_push_run_metrics_returns_zero_when_not_configured():
    assert grafana_push.push_run_metrics() == 0


def test_post_annotation_is_a_silent_noop_when_not_configured():
    grafana_push.post_annotation("should not raise", tags=["test"])  # no exception


def test_push_run_metrics_calls_writer_when_configured(monkeypatch):
    test_store = RunsStore()
    test_store.start(
        "run-1",
        {
            "model": "llama2-7b",
            "gpu": "h100-sxm",
            "topology": "nvlink_node",
            "compute_s_per_step": 0.5,
            "communication_s_per_step": 0.2,
        },
    )
    test_store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 0.1})
    monkeypatch.setattr(grafana_push, "store", test_store)

    sent_items = []

    class FakeResult:
        samples_sent = 4

    class FakeWriter:
        def send(self, items):
            sent_items.extend(items)
            return FakeResult()

    monkeypatch.setattr(grafana_push, "_writer", FakeWriter())
    assert grafana_push.push_available() is True

    samples = grafana_push.push_run_metrics()
    assert samples == 4
    names = {item["metric"]["__name__"] for item in sent_items}
    assert "simgpu_run_status" in names
    assert "simgpu_run_power_watts" in names
