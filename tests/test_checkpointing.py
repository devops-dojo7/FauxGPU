import pytest

from engine.checkpointing import (
    checkpoint_restore_seconds,
    checkpoint_size_gb,
    checkpoint_write_seconds,
    is_recovery_trigger,
    recovery_overhead_seconds,
)
from engine.memory import ModelShape, lora_trainable_params, optimizer_state_bytes, weights_bytes

LLAMA2_7B = ModelShape(params=6.74e9, num_layers=32, hidden_dim=4096, num_heads=32, head_dim=128)

# The LoRA paper's own GPT-3 175B shape (Hu et al. 2021) — used for its own
# "~35MB adapter checkpoint vs. 350GB full checkpoint" worked example.
GPT3_175B = ModelShape(params=175.2558e9, num_layers=96, hidden_dim=12288, num_heads=96, head_dim=128)


def test_checkpoint_size_matches_weights_plus_optimizer_state():
    expected_bytes = weights_bytes(LLAMA2_7B, "bf16") + optimizer_state_bytes(LLAMA2_7B, "adam", True)
    assert checkpoint_size_gb(LLAMA2_7B, "bf16") == expected_bytes / 1e9


def test_checkpoint_size_excludes_gradients_and_activations():
    # weights + optimizer state only — much smaller than a full VRAM breakdown would be
    size = checkpoint_size_gb(LLAMA2_7B, "bf16")
    weights_only_gb = weights_bytes(LLAMA2_7B, "bf16") / 1e9
    assert size > weights_only_gb  # optimizer state adds to it
    assert size < weights_only_gb * 10  # but stays in a sane ballpark, not double-counting VRAM categories


def test_write_and_restore_scale_with_size():
    small = checkpoint_write_seconds(10.0)
    large = checkpoint_write_seconds(20.0)
    assert large == small * 2
    assert checkpoint_restore_seconds(20.0) == checkpoint_restore_seconds(10.0) * 2


def test_restore_is_faster_than_write_for_the_same_size():
    size = 50.0
    assert checkpoint_restore_seconds(size) < checkpoint_write_seconds(size)


def test_recovery_overhead_scales_with_steps_lost():
    base = recovery_overhead_seconds(10.0, steps_lost=5, step_time_s=2.0)
    more_lost = recovery_overhead_seconds(10.0, steps_lost=10, step_time_s=2.0)
    assert more_lost > base
    assert more_lost - base == 5 * 2.0


def test_recovery_overhead_with_no_checkpoint_is_pure_lost_time():
    # size_gb=0 means checkpointing was never enabled: no restore I/O, but
    # every step since the start counts as lost.
    overhead = recovery_overhead_seconds(0.0, steps_lost=20, step_time_s=1.5)
    assert overhead == 20 * 1.5


def test_is_recovery_trigger_only_for_crash_kinds():
    assert is_recovery_trigger("xid_error") is True
    assert is_recovery_trigger("node_drain") is True
    assert is_recovery_trigger("nvlink_degradation") is False


# --- LoRA checkpoint sizing (Hu et al. 2021) ---


def test_lora_checkpoint_matches_the_papers_own_35mb_gpt3_example():
    # Sec 4.2 footnote's own worked example: r=4, Wq+Wv adapted, fp16 ->
    # "the checkpoint size is reduced by roughly 10,000x (from 350GB to
    # 35MB)". The paper rounds both ends, so a loose tolerance matches its
    # own precision.
    size_mb = checkpoint_size_gb(GPT3_175B, precision="fp16", peft_method="lora", peft_rank=4, peft_target_modules=2) * 1000
    assert size_mb == pytest.approx(35, rel=0.1)


def test_lora_checkpoint_is_far_smaller_than_full_checkpoint():
    # 350GB -> 35MB is a ~10,000x reduction; the full checkpoint here also
    # includes optimizer state (not just weights), so the actual ratio runs
    # somewhat higher than the paper's own weights-only 350GB/35MB figure —
    # assert a directional, order-of-magnitude reduction instead of pinning
    # the paper's exact (weights-only) ratio.
    full_gb = checkpoint_size_gb(GPT3_175B, precision="fp16")
    lora_gb = checkpoint_size_gb(GPT3_175B, precision="fp16", peft_method="lora", peft_rank=4, peft_target_modules=2)
    assert full_gb / lora_gb > 5_000


def test_lora_checkpoint_matches_raw_adapter_param_count():
    rank, target_modules = 8, 2
    adapter_params = lora_trainable_params(LLAMA2_7B, rank, target_modules)
    size_gb = checkpoint_size_gb(LLAMA2_7B, precision="fp16", peft_method="lora", peft_rank=rank, peft_target_modules=target_modules)
    assert size_gb == pytest.approx(adapter_params * 2.0 / 1e9)  # fp16 = 2 bytes/param
