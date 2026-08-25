"""Failure/chaos injection for a live training run: Xid errors (transient
step stalls), NVLink degradation (temporary bandwidth loss), and node drain
(permanent GPU loss). Modeled by perturbing the topology/step-time inputs
fed into the existing engine.compute.estimate_step_time formula — no
changes to that formula itself. Pure Python, no I/O.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

from engine.compute import estimate_step_time
from engine.memory import ModelShape
from engine.topology import ClusterTopology

ChaosKind = Literal["xid_error", "nvlink_degradation", "node_drain"]


@dataclass(frozen=True)
class ChaosEvent:
    event_id: str
    kind: ChaosKind
    injected_at_step: int
    duration_steps: int | None  # None = permanent (node_drain only)
    severity: float
    # xid_error: stall multiplier applied to that step's time (e.g. 5.0 = step takes 5x as long)
    # nvlink_degradation: fraction of nominal NVLink bandwidth remaining (e.g. 0.3 = 70% bandwidth loss)
    # node_drain: number of nodes removed from the cluster (permanent for the rest of the run)


def is_active(event: ChaosEvent, step: int) -> bool:
    if step < event.injected_at_step:
        return False
    if event.duration_steps is None:
        return True
    return step < event.injected_at_step + event.duration_steps


def effective_topology(base: ClusterTopology, active_events: list[ChaosEvent]) -> ClusterTopology:
    """Applies currently-active node_drain/nvlink_degradation events to the
    base topology. xid_error is handled separately by stall_multiplier,
    since it doesn't change the cluster's steady-state shape.
    """
    topology = base

    drained_nodes = sum(e.severity for e in active_events if e.kind == "node_drain")
    if drained_nodes:
        remaining = max(1, int(topology.num_nodes - drained_nodes))
        topology = dataclasses.replace(topology, num_nodes=remaining)

    degradations = [e.severity for e in active_events if e.kind == "nvlink_degradation"]
    if degradations and topology.gpu.nvlink_gbps:
        factor = min(degradations)  # worst active degradation wins, doesn't compound
        degraded_gpu = dataclasses.replace(topology.gpu, nvlink_gbps=topology.gpu.nvlink_gbps * factor)
        topology = dataclasses.replace(topology, gpu=degraded_gpu)

    return topology


def stall_multiplier(active_events: list[ChaosEvent]) -> float:
    """Extra one-step time multiplier from active Xid error(s) — models a
    driver reset/step retry. Multiple simultaneous Xid events compound.
    """
    multiplier = 1.0
    for e in active_events:
        if e.kind == "xid_error":
            multiplier *= e.severity
    return multiplier


def effective_step_seconds(
    model: ModelShape,
    base_topology: ClusterTopology,
    active_events: list[ChaosEvent],
    tokens_per_step: int,
    precision: str = "bf16",
    utilization: float = 0.35,
) -> float:
    topology = effective_topology(base_topology, active_events)
    step = estimate_step_time(model, topology, tokens_per_step, precision, utilization)
    return step.total_s * stall_multiplier(active_events)
