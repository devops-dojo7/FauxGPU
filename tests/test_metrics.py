from api.metrics import render_prometheus_metrics
from api.runs_store import RunsStore
import api.metrics as metrics_module


def test_render_empty_store_has_only_help_type_lines(monkeypatch):
    monkeypatch.setattr(metrics_module, "store", RunsStore())
    text = render_prometheus_metrics()
    assert "# HELP simgpu_run_status" in text
    assert "simgpu_run_status{" not in text


def test_render_includes_running_run_with_power(monkeypatch):
    test_store = RunsStore()
    test_store.start(
        "run-1",
        {
            "model": "llama2-7b",
            "gpu": "h100-sxm",
            "topology": "nvlink_node",
            "total_steps": 10,
            "compute_s_per_step": 0.5,
            "communication_s_per_step": 0.2,
        },
    )
    test_store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 0.1})
    monkeypatch.setattr(metrics_module, "store", test_store)

    text = render_prometheus_metrics()
    assert 'simgpu_run_status{run_id="run-1",model="llama2-7b",gpu="h100-sxm",topology="nvlink_node"} 1' in text
    assert "simgpu_run_power_watts{" in text
    assert "simgpu_run_step{" in text and "} 1" in text


def test_render_skips_power_for_unknown_gpu(monkeypatch):
    test_store = RunsStore()
    test_store.start("run-1", {"model": "custom", "gpu": "not-a-real-gpu", "topology": "single_gpu"})
    monkeypatch.setattr(metrics_module, "store", test_store)

    text = render_prometheus_metrics()
    assert "simgpu_run_power_watts{" not in text
