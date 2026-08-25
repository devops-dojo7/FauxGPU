from engine.chaos import ChaosEvent, effective_step_seconds, effective_topology, is_active, stall_multiplier
from engine.compute import estimate_step_time
from engine.memory import ModelShape
from engine.topology import build_topology

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)


def _event(kind, severity, injected_at_step=0, duration_steps=None, event_id="e1"):
    return ChaosEvent(event_id=event_id, kind=kind, injected_at_step=injected_at_step, duration_steps=duration_steps, severity=severity)


def test_transient_event_is_only_active_within_its_window():
    e = _event("xid_error", 3.0, injected_at_step=5, duration_steps=3)
    assert not is_active(e, 4)
    assert is_active(e, 5)
    assert is_active(e, 7)
    assert not is_active(e, 8)


def test_permanent_event_stays_active_forever_after_injection():
    e = _event("node_drain", 1, injected_at_step=5, duration_steps=None)
    assert not is_active(e, 4)
    assert is_active(e, 5)
    assert is_active(e, 1000)


def test_no_active_events_returns_base_topology_values_unchanged():
    base = build_topology("multi_node", gpu_id="h100-sxm", gpus_per_node=8, num_nodes=4, fabric_id="infiniband-hdr")
    result = effective_topology(base, [])
    assert result.num_nodes == base.num_nodes
    assert result.gpus_per_node == base.gpus_per_node
    assert result.gpu.nvlink_gbps == base.gpu.nvlink_gbps


def test_node_drain_reduces_node_count_and_clamps_at_one():
    base = build_topology("multi_node", gpu_id="h100-sxm", gpus_per_node=8, num_nodes=4, fabric_id="infiniband-hdr")
    result = effective_topology(base, [_event("node_drain", 2)])
    assert result.num_nodes == 2

    drain_everything = effective_topology(base, [_event("node_drain", 10)])
    assert drain_everything.num_nodes == 1


def test_nvlink_degradation_applies_worst_active_fraction_not_compounded():
    base = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    result = effective_topology(base, [_event("nvlink_degradation", 0.5), _event("nvlink_degradation", 0.2)])
    assert result.gpu.nvlink_gbps == base.gpu.nvlink_gbps * 0.2  # worst (min) wins, not 0.5*0.2


def test_multiple_xid_errors_compound_multiplicatively():
    events = [_event("xid_error", 3.0, event_id="a"), _event("xid_error", 2.0, event_id="b")]
    assert stall_multiplier(events) == 6.0


def test_stall_multiplier_ignores_non_xid_events():
    events = [_event("node_drain", 1), _event("nvlink_degradation", 0.5)]
    assert stall_multiplier(events) == 1.0


def test_effective_step_seconds_increases_for_each_fault_kind():
    topo = build_topology("nvlink_node", gpu_id="h100-sxm", gpus_per_node=8)
    baseline = estimate_step_time(LLAMA2_7B, topo, tokens_per_step=32768).total_s

    xid = effective_step_seconds(LLAMA2_7B, topo, [_event("xid_error", 5.0)], tokens_per_step=32768)
    assert xid == baseline * 5.0

    degraded = effective_step_seconds(LLAMA2_7B, topo, [_event("nvlink_degradation", 0.1)], tokens_per_step=32768)
    assert degraded > baseline

    drained_topo = build_topology("multi_node", gpu_id="h100-sxm", gpus_per_node=8, num_nodes=4, fabric_id="infiniband-hdr")
    drained_baseline = estimate_step_time(LLAMA2_7B, drained_topo, tokens_per_step=32768).total_s
    drained = effective_step_seconds(LLAMA2_7B, drained_topo, [_event("node_drain", 2)], tokens_per_step=32768)
    assert drained > drained_baseline
