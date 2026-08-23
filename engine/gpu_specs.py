"""Loads the GPU/vendor spec + pricing table and interconnect fabric table from data/gpus.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DATA_PATH = Path(__file__).parent / "data" / "gpus.yaml"


@dataclass(frozen=True)
class GpuSpec:
    id: str
    vendor: str
    name: str
    vram_gb: float
    mem_bandwidth_gbps: float
    bf16_tflops: float
    fp8_tflops: float | None
    nvlink_gbps: float | None
    price_per_hr_usd: float
    tdp_watts: float
    idle_watts: float
    # Compute-die internals, for readers who want to look past the top-line
    # throughput numbers. Left null rather than guessed where a vendor
    # hasn't published an exact figure (e.g. Blackwell's per-die SM/core
    # counts, which NVIDIA has not disclosed for the dual-die B200).
    architecture: str = ""
    process_node: str = ""
    launch_year: int | None = None
    transistors_b: float | None = None
    sm_count: int | None = None
    core_count: int | None = None
    core_label: str = "CUDA Cores"
    matrix_core_count: int | None = None
    matrix_core_label: str | None = None
    boost_clock_ghz: float | None = None
    l2_cache_mb: float | None = None
    interconnect_name: str | None = None
    # Set only for devices priced as a one-time purchase rather than a cloud
    # rental (e.g. DGX Spark) — explains what price_per_hr_usd actually means
    # for that entry, since it's an amortized figure, not a market rate.
    price_note: str | None = None


@dataclass(frozen=True)
class Fabric:
    id: str
    name: str
    bandwidth_gbps: float
    latency_us: float


def _load() -> tuple[dict[str, GpuSpec], dict[str, Fabric]]:
    raw = yaml.safe_load(_DATA_PATH.read_text())
    gpus = {g["id"]: GpuSpec(**g) for g in raw["gpus"]}
    fabrics = {f["id"]: Fabric(**f) for f in raw["fabrics"]}
    return gpus, fabrics


GPUS, FABRICS = _load()


def get_gpu(gpu_id: str) -> GpuSpec:
    try:
        return GPUS[gpu_id]
    except KeyError:
        raise ValueError(f"Unknown GPU id: {gpu_id!r}. Known: {sorted(GPUS)}") from None


def get_fabric(fabric_id: str) -> Fabric:
    try:
        return FABRICS[fabric_id]
    except KeyError:
        raise ValueError(f"Unknown fabric id: {fabric_id!r}. Known: {sorted(FABRICS)}") from None


def list_gpus() -> list[GpuSpec]:
    return list(GPUS.values())
