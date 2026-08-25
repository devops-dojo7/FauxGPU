"""Network/InfiniBand congestion modeling: what happens when multiple
concurrent training jobs share one physical interconnect fabric, versus
engine.compute.ring_all_reduce_seconds's default assumption that a job gets
the fabric's full static bandwidth to itself.

Contention model, deliberately simple: each job's fair-share bandwidth is
proportional to its GPU count (more GPUs generally means more NICs/links
pulling from the shared fabric). Real InfiniBand contention also depends on
adaptive routing, QoS traffic classes, and topology-specific (fat-tree vs.
dragonfly) effects — none of that is modeled here, same "teaching
approximation" tier as the rest of engine/.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.compute import ring_all_reduce_seconds
from engine.gpu_specs import Fabric


@dataclass(frozen=True)
class NetworkJob:
    job_id: str
    team: str
    num_gpus: int
    payload_gb: float  # gradient all-reduce payload size per step


@dataclass(frozen=True)
class NetworkJobResult:
    job_id: str
    team: str
    num_gpus: int
    bandwidth_share_gbps: float
    isolated_comm_s: float
    contended_comm_s: float
    slowdown_factor: float


@dataclass(frozen=True)
class NetworkContentionResult:
    fabric_id: str
    fabric_bandwidth_gbps: float
    total_gpus_sharing_fabric: int
    jobs: tuple[NetworkJobResult, ...]


def simulate_network_contention(fabric: Fabric, jobs: list[NetworkJob]) -> NetworkContentionResult:
    if not jobs:
        raise ValueError("At least one job is required")
    ids = [j.job_id for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("job_id values must be unique")

    total_gpus = sum(j.num_gpus for j in jobs)

    results = []
    for job in jobs:
        payload_bytes = job.payload_gb * 1e9
        share_gbps = fabric.bandwidth_gbps * (job.num_gpus / total_gpus)
        isolated_s = ring_all_reduce_seconds(payload_bytes, job.num_gpus, fabric.bandwidth_gbps)
        contended_s = ring_all_reduce_seconds(payload_bytes, job.num_gpus, share_gbps)
        results.append(
            NetworkJobResult(
                job_id=job.job_id,
                team=job.team,
                num_gpus=job.num_gpus,
                bandwidth_share_gbps=share_gbps,
                isolated_comm_s=isolated_s,
                contended_comm_s=contended_s,
                slowdown_factor=(contended_s / isolated_s) if isolated_s > 0 else 1.0,
            )
        )

    return NetworkContentionResult(
        fabric_id=fabric.id,
        fabric_bandwidth_gbps=fabric.bandwidth_gbps,
        total_gpus_sharing_fabric=total_gpus,
        jobs=tuple(results),
    )
