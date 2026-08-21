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
