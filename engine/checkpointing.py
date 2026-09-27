"""Checkpoint save/restore overhead and failure-recovery cost for a live
training run. Distinct from engine.memory's `checkpointing` flag, which is
*activation* checkpointing (a memory-saving recompute technique) — this
module models writing model state to durable storage for fault tolerance,
an unrelated concept. Pure Python, no I/O, no framework dependency.
"""

from __future__ import annotations

from engine.memory import (
    ModelShape,
    bytes_per_param,
    lora_trainable_params,
    optimizer_state_bytes,
    weights_bytes,
)

CHECKPOINT_WRITE_GBPS = 5.0  # illustrative shared-storage write throughput (teaching approximation)
CHECKPOINT_RESTORE_GBPS = 8.0  # reads are typically faster than writes on shared storage

# Faults that crash a real training process and require a restart from the
# last checkpoint. nvlink_degradation is a slowdown, not a crash — training
# continues, just slower, so it never triggers a recovery.
RECOVERY_TRIGGER_KINDS = {"xid_error", "node_drain"}


def checkpoint_size_gb(
    model: ModelShape,
    precision: str,
    optimizer: str = "adam",
    fp32_master_copy: bool = True,
    peft_method: str = "full",
    peft_rank: int = 8,
    peft_target_modules: int = 2,
) -> float:
    """Bytes of persisted state in a checkpoint: weights + optimizer state.
    Deliberately excludes gradients/activations/kv_cache — none of those are
    part of what gets written to a training checkpoint.

    peft_method="lora"/"qlora" reproduce LoRA's own headline storage story
    (Hu et al. 2021, Sec 4.2): only the tiny trainable adapter needs saving
    per task, not the full model — the paper's own GPT-3 175B example
    (r=4, Wq+Wv) puts this at ~35MB, a ~10,000x reduction from the 350GB
    full checkpoint, letting many task-specific checkpoints share one
    base-model copy instead of each paying the full checkpoint size. QLoRA
    shares this same adapter-only checkpoint (only the frozen base's
    storage format differs, and the frozen base isn't part of a
    checkpoint's persisted trainable state in the first place) — see
    tests/test_checkpointing.py.
    """
    if peft_method in ("lora", "qlora"):
        adapter_params = lora_trainable_params(model, peft_rank, peft_target_modules)
        total_bytes = adapter_params * bytes_per_param(precision)
    else:
        total_bytes = weights_bytes(model, precision) + optimizer_state_bytes(model, optimizer, fp32_master_copy)
    return total_bytes / 1e9


def checkpoint_write_seconds(size_gb: float) -> float:
    return size_gb / CHECKPOINT_WRITE_GBPS


def checkpoint_restore_seconds(size_gb: float) -> float:
    return size_gb / CHECKPOINT_RESTORE_GBPS


def recovery_overhead_seconds(size_gb: float, steps_lost: int, step_time_s: float) -> float:
    """Wall-clock cost of recovering from a crash: time to restore the
    checkpoint back onto the GPU, plus redoing every step since the last
    checkpoint (or since the start, if size_gb == 0 — checkpointing was
    never enabled, so there's no restore I/O, but the full run so far is
    lost. That's a deliberate, pedagogically useful edge case demonstrating
    the cost of running with no fault tolerance, not a bug to special-case.)
    """
    return checkpoint_restore_seconds(size_gb) + steps_lost * step_time_s


def is_recovery_trigger(kind: str) -> bool:
    return kind in RECOVERY_TRIGGER_KINDS
