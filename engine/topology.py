"""Cluster topology shapes: how GPUs within a node and nodes within a cluster are
wired together, and the effective bandwidth used for collective communication
(e.g. gradient all-reduce) at each scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.gpu_specs import Fabric, GpuSpec, get_fabric, get_gpu


@dataclass(frozen=True)
class ClusterTopology:
    shape: str  # "single_gpu" | "nvlink_node" | "multi_node"
    gpu: GpuSpec
    gpus_per_node: int
    num_nodes: int
    inter_node_fabric: Fabric | None  # None for single_gpu / single-node shapes

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_node * self.num_nodes

    @property
    def intra_node_bandwidth_gbps(self) -> float | None:
        """NVLink (or equivalent) bandwidth available for intra-node collectives."""
        return self.gpu.nvlink_gbps

    @property
    def inter_node_bandwidth_gbps(self) -> float | None:
        return self.inter_node_fabric.bandwidth_gbps if self.inter_node_fabric else None

    def bottleneck_bandwidth_gbps(self) -> float:
        """The slowest link a cross-GPU collective has to cross at this scale.
        Single node -> NVLink (or PCIe-shared-host, modeled as a low default).
        Multi node -> the inter-node fabric always dominates (InfiniBand/Ethernet
        is far slower than NVLink), since gradients must cross it each step.
        """
        if self.num_nodes <= 1:
            return self.intra_node_bandwidth_gbps or 32.0  # PCIe4 x16 ballpark, GB/s->Gb/s below
        return self.inter_node_bandwidth_gbps


def build_topology(
    shape: str,
    gpu_id: str,
    gpus_per_node: int = 1,
    num_nodes: int = 1,
    fabric_id: str | None = None,
) -> ClusterTopology:
    gpu = get_gpu(gpu_id)

    if shape == "single_gpu":
        return ClusterTopology(shape, gpu, gpus_per_node=1, num_nodes=1, inter_node_fabric=None)

    if shape == "nvlink_node":
        if gpu.nvlink_gbps is None:
            raise ValueError(f"{gpu.name} has no NVLink — cannot form an nvlink_node topology")
        return ClusterTopology(
            shape, gpu, gpus_per_node=gpus_per_node, num_nodes=1, inter_node_fabric=None
        )

    if shape == "multi_node":
        fabric = get_fabric(fabric_id or "infiniband-hdr")
        return ClusterTopology(
            shape,
            gpu,
            gpus_per_node=gpus_per_node,
            num_nodes=num_nodes,
            inter_node_fabric=fabric,
        )

    raise ValueError(f"Unknown topology shape: {shape!r}")
