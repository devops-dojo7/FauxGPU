"""NVIDIA Multi-Instance GPU (MIG) capacity planning: partitioning a
physical GPU into isolated slices, and packing multiple tenants' slice
requests across a pool of physical GPUs.

Scope, deliberately simplified for a teaching tool:
- Uses the well-known **slice-budget model** that's accurate for
  Ampere/Hopper MIG GPUs: each physical GPU has a 7-slot compute budget and
  an 8-slot memory budget. A profile named "Ng.Xgb" consumes N compute
  slots; its memory-slot cost follows NVIDIA's real (slightly asymmetric)
  table — 1g/2g consume as many memory slots as compute slots, 3g/4g both
  consume 4 memory slots, and 7g consumes all 8. This is real MIG behavior,
  not an approximation.
- Does NOT model NVIDIA's exact placement-ID legality table (which precise
  profile *combinations* can physically coexist in which slots) — this
  module only tracks aggregate remaining compute/memory budget per GPU,
  which is accurate for capacity/utilization math but not every
  placement-adjacency rule.
- Packing is greedy first-fit (place each request on the first pool GPU
  with enough remaining budget in both dimensions), not globally-optimal
  bin-packing (NP-hard in general) — good enough to visualize fragmentation,
  not guaranteed to use the minimum possible GPU count.
- B200/Blackwell is excluded: its MIG profile grid isn't published at the
  same confidence tier as Ampere/Hopper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

COMPUTE_SLOTS_TOTAL = 7
MEMORY_SLOTS_TOTAL = 8


@dataclass(frozen=True)
class MigProfile:
    id: str  # e.g. "1g.10gb"
    compute_slots: int  # out of COMPUTE_SLOTS_TOTAL
    memory_slots: int  # out of MEMORY_SLOTS_TOTAL
    memory_gb: float


def _standard_profiles(vram_gb: float) -> tuple[MigProfile, ...]:
    """The standard 5-profile MIG ladder, scaled to a GPU's VRAM capacity.
    Memory-per-slot = vram_gb / MEMORY_SLOTS_TOTAL, matching NVIDIA's real
    published profile names for every 8-memory-slot MIG GPU (verified
    against A100-80GB, A100-40GB, and H200's published profile tables).
    """
    per_slot_gb = vram_gb / MEMORY_SLOTS_TOTAL

    def gb(memory_slots: int) -> float:
        # Round-half-up (not Python's round-half-to-even) to match NVIDIA's
        # published naming, e.g. H200's 141/8*4=70.5 is published as "71gb".
        return math.floor(per_slot_gb * memory_slots + 0.5)

    return (
        MigProfile(f"1g.{gb(1):.0f}gb", compute_slots=1, memory_slots=1, memory_gb=gb(1)),
        MigProfile(f"2g.{gb(2):.0f}gb", compute_slots=2, memory_slots=2, memory_gb=gb(2)),
        MigProfile(f"3g.{gb(4):.0f}gb", compute_slots=3, memory_slots=4, memory_gb=gb(4)),
        MigProfile(f"4g.{gb(4):.0f}gb", compute_slots=4, memory_slots=4, memory_gb=gb(4)),
        MigProfile(f"7g.{gb(8):.0f}gb", compute_slots=7, memory_slots=8, memory_gb=gb(8)),
    )


# GPUs with a real, documented MIG profile table (Ampere/Hopper datacenter
# parts). B200/Blackwell and every non-datacenter/consumer GPU are excluded.
MIG_PROFILES: dict[str, tuple[MigProfile, ...]] = {
    "a100-80gb-sxm": _standard_profiles(80),
    "a100-40gb-sxm": _standard_profiles(40),
    "h100-sxm": _standard_profiles(80),
    "h100-pcie": _standard_profiles(80),
    "h200-sxm": _standard_profiles(141),
}


def supports_mig(gpu_id: str) -> bool:
    return gpu_id in MIG_PROFILES


def list_mig_profiles(gpu_id: str) -> tuple[MigProfile, ...]:
    try:
        return MIG_PROFILES[gpu_id]
    except KeyError:
        raise ValueError(f"{gpu_id!r} does not support MIG. Known: {sorted(MIG_PROFILES)}") from None


@dataclass(frozen=True)
class MigRequest:
    request_id: str
    tenant: str
    profile_id: str


@dataclass(frozen=True)
class MigPlacement:
    request_id: str
    tenant: str
    profile_id: str
    gpu_index: int
    compute_slots: int
    memory_slots: int


@dataclass(frozen=True)
class MigPackingResult:
    placements: tuple[MigPlacement, ...]
    unplaced: tuple[MigRequest, ...]
    pool_size: int
    gpus_used: int
    compute_utilization_pct: float
    memory_utilization_pct: float


def pack_mig_requests(gpu_id: str, pool_size: int, requests: list[MigRequest]) -> MigPackingResult:
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")
    profiles_by_id = {p.id: p for p in list_mig_profiles(gpu_id)}

    unknown = sorted({r.profile_id for r in requests if r.profile_id not in profiles_by_id})
    if unknown:
        raise ValueError(f"Profile(s) not valid for {gpu_id!r}: {unknown}. Known: {sorted(profiles_by_id)}")

    remaining_compute = [COMPUTE_SLOTS_TOTAL] * pool_size
    remaining_memory = [MEMORY_SLOTS_TOTAL] * pool_size
    placements: list[MigPlacement] = []
    unplaced: list[MigRequest] = []

    for req in requests:
        profile = profiles_by_id[req.profile_id]
        for gpu_index in range(pool_size):
            if remaining_compute[gpu_index] >= profile.compute_slots and remaining_memory[gpu_index] >= profile.memory_slots:
                remaining_compute[gpu_index] -= profile.compute_slots
                remaining_memory[gpu_index] -= profile.memory_slots
                placements.append(
                    MigPlacement(
                        request_id=req.request_id,
                        tenant=req.tenant,
                        profile_id=req.profile_id,
                        gpu_index=gpu_index,
                        compute_slots=profile.compute_slots,
                        memory_slots=profile.memory_slots,
                    )
                )
                break
        else:
            unplaced.append(req)

    used_compute = sum(p.compute_slots for p in placements)
    used_memory = sum(p.memory_slots for p in placements)
    gpus_used = len({p.gpu_index for p in placements})

    return MigPackingResult(
        placements=tuple(placements),
        unplaced=tuple(unplaced),
        pool_size=pool_size,
        gpus_used=gpus_used,
        compute_utilization_pct=(used_compute / (pool_size * COMPUTE_SLOTS_TOTAL)) * 100,
        memory_utilization_pct=(used_memory / (pool_size * MEMORY_SLOTS_TOTAL)) * 100,
    )
