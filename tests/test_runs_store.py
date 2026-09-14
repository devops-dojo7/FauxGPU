from api.runs_store import RunsStore


def test_start_then_steps_then_finish():
    store = RunsStore()
    store.start("run-1", {"model": "llama2-7b"})
    store.add_step("run-1", {"step": 1, "tokens_seen": 100, "elapsed_s": 0.1})
    store.add_step("run-1", {"step": 2, "tokens_seen": 200, "elapsed_s": 0.2})
    run = store.finish("run-1")

    assert run.status == "done"
    assert len(run.steps) == 2
    assert run.steps[-1]["step"] == 2


def test_add_step_to_unknown_run_returns_none():
    store = RunsStore()
    assert store.add_step("nope", {"step": 1, "tokens_seen": 1, "elapsed_s": 0.1}) is None


def test_list_orders_newest_first():
    store = RunsStore()
    store.start("older", {})
    store.start("newer", {})
    ids = [r.run_id for r in store.list()]
    assert ids[0] == "newer"


def test_max_runs_evicts_oldest():
    store = RunsStore(max_runs=2)
    store.start("a", {})
    store.start("b", {})
    store.start("c", {})
    ids = {r.run_id for r in store.list()}
    assert "a" not in ids
    assert ids == {"b", "c"}


def test_add_checkpoint_event_appends_with_generated_id():
    store = RunsStore()
    store.start("run-1", {"gpu": "h100-sxm", "topology": "single_gpu", "total_gpus": 1})
    store.add_checkpoint_event("run-1", {"kind": "save", "step": 10, "size_gb": 5.0, "overhead_s": 1.0, "steps_lost": None})
    store.add_checkpoint_event("run-1", {"kind": "restore", "step": 15, "size_gb": 5.0, "overhead_s": 2.0, "steps_lost": 5})

    run = store.get("run-1")
    assert len(run.checkpoint_events) == 2
    assert run.checkpoint_events[0]["kind"] == "save"
    assert run.checkpoint_events[1]["kind"] == "restore"
    assert run.checkpoint_events[1]["steps_lost"] == 5
    assert all("event_id" in e for e in run.checkpoint_events)


def test_add_checkpoint_event_to_unknown_run_returns_none():
    store = RunsStore()
    assert store.add_checkpoint_event("nope", {"kind": "save", "step": 1, "size_gb": 1.0, "overhead_s": 0.1, "steps_lost": None}) is None


def test_active_on_fabric_returns_other_running_runs_on_same_fabric():
    store = RunsStore()
    store.start("run-a", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.2})
    store.start("run-b", {"fabric_id": "infiniband-hdr", "total_gpus": 16, "payload_gb": 2.4})

    others = store.active_on_fabric("infiniband-hdr", exclude_run_id="run-a")

    assert len(others) == 1
    assert others[0] == {"run_id": "run-b", "total_gpus": 16, "payload_gb": 2.4}


def test_active_on_fabric_excludes_different_fabric_and_finished_runs():
    store = RunsStore()
    store.start("same-fabric", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    store.start("other-fabric", {"fabric_id": "roce-100g", "total_gpus": 8, "payload_gb": 1.0})
    store.start("finished", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    store.finish("finished")

    others = store.active_on_fabric("infiniband-hdr", exclude_run_id="probe")

    assert {o["run_id"] for o in others} == {"same-fabric"}


def test_active_on_fabric_with_no_matches_returns_empty_list():
    store = RunsStore()
    store.start("run-a", {"fabric_id": "infiniband-hdr", "total_gpus": 8, "payload_gb": 1.0})
    assert store.active_on_fabric("infiniband-hdr", exclude_run_id="run-a") == []
    assert store.active_on_fabric("no-such-fabric") == []
