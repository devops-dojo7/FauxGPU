from api.dcgm_metrics import compute_dcgm_series, dcgm_metric_items, render_dcgm_prometheus_text
from api.runs_store import RunsStore


def _running_store():
    store = RunsStore()
    store.start(
        "run-1",
        {
            "model": "llama2-7b",
            "gpu": "h100-sxm",
            "topology": "nvlink_node",
            "total_gpus": 4,
            "compute_s_per_step": 0.5,
            "communication_s_per_step": 0.2,
        },
    )
    store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 1.5})
    return store


def test_finished_run_emits_no_dcgm_series():
    store = RunsStore()
    run = store.start("run-1", {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1})
    store.finish("run-1")
    run = store.get("run-1")
    assert compute_dcgm_series(run) == []


def test_running_run_emits_series_per_simulated_gpu():
    store = _running_store()
    run = store.get("run-1")
    series = compute_dcgm_series(run)
    gpu_indices = {s["labels"]["gpu"] for s in series}
    assert gpu_indices == {"0", "1", "2", "3"}


def test_expected_metric_names_present():
    store = _running_store()
    run = store.get("run-1")
    names = {s["name"] for s in compute_dcgm_series(run)}
    assert "DCGM_FI_DEV_POWER_USAGE" in names
    assert "DCGM_FI_DEV_GPU_TEMP" in names
    assert "DCGM_FI_DEV_FB_USED" in names
    assert "DCGM_FI_PROF_GR_ENGINE_ACTIVE" in names
    assert "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION" in names


def test_temperature_rises_with_power_draw():
    store = RunsStore()
    store.start(
        "idle",
        {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1, "compute_s_per_step": 0.0, "communication_s_per_step": 1.0},
    )
    store.start(
        "busy",
        {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1, "compute_s_per_step": 1.0, "communication_s_per_step": 0.0},
    )
    idle_temp = next(s["value"] for s in compute_dcgm_series(store.get("idle")) if s["name"] == "DCGM_FI_DEV_GPU_TEMP")
    busy_temp = next(s["value"] for s in compute_dcgm_series(store.get("busy")) if s["name"] == "DCGM_FI_DEV_GPU_TEMP")
    assert busy_temp > idle_temp


def test_unknown_gpu_id_yields_no_series():
    store = RunsStore()
    store.start("run-1", {"gpu": "not-a-real-gpu", "topology": "single_gpu", "total_gpus": 1})
    assert compute_dcgm_series(store.get("run-1")) == []


def test_render_prometheus_text_includes_dcgm_lines():
    store = _running_store()
    text = render_dcgm_prometheus_text(store.list())
    assert 'DCGM_FI_DEV_POWER_USAGE{Hostname="simgpu-nvlink_node",gpu="0"' in text


def test_metric_items_shaped_for_remote_write():
    store = _running_store()
    items = dcgm_metric_items(store.list())
    assert len(items) > 0
    assert all("timestamps" in i and "values" in i for i in items)
