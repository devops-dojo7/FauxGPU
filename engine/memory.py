"""VRAM breakdown: weights, optimizer states, gradients, activations, and KV cache.

All formulas are standard textbook/paper approximations (see comments), not exact
values from any particular framework — good enough to teach *why* VRAM runs out,
not to reproduce a specific runtime's allocator behavior byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_ELEMENT = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "int4": 0.5,
}


def bytes_per_param(precision: str) -> float:
    try:
        return BYTES_PER_ELEMENT[precision]
    except KeyError:
        raise ValueError(
            f"Unknown precision: {precision!r}. Known: {sorted(BYTES_PER_ELEMENT)}"
        ) from None


@dataclass(frozen=True)
class ModelShape:
    """Minimal transformer shape needed for the memory/compute formulas.

    Three optional fields cover attention/MoE variants beyond plain
    multi-head attention (MHA) dense models — each defaults to the MHA/dense
    behavior when left unset, so existing presets don't need to change:

    - num_kv_heads: for grouped-query attention (GQA), the number of KV heads
      is smaller than the number of query heads (num_heads), shrinking KV
      cache proportionally. None = MHA (num_kv_heads == num_heads).
    - active_params: for Mixture-of-Experts (MoE) models, only a subset of
      total params is touched per token (routed experts + shared layers).
      All params must still be resident in VRAM (weights/optimizer state),
      but compute (FLOPs) and per-token memory traffic scale with the
      *active* count, not the total. None = dense (active == total params).
    - kv_latent_dim: for multi-head latent attention (MLA, e.g. DeepSeek-V3
      and models sharing its architecture), K/V are compressed into one
      shared latent vector per token per layer instead of per-head K/V
      tensors, which is a fundamentally different KV cache formula. None =
      use the standard (GQA/MHA) head-count-based KV cache formula.
    """

    params: float  # total parameter count (all experts, if MoE)
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int  # usually hidden_dim // num_heads, kept explicit for GQA/MQA models
    num_kv_heads: int | None = None
    active_params: float | None = None
    kv_latent_dim: int | None = None

    @property
    def effective_kv_heads(self) -> int:
        return self.num_kv_heads if self.num_kv_heads is not None else self.num_heads

    @property
    def effective_active_params(self) -> float:
        return self.active_params if self.active_params is not None else self.params

    @property
    def is_moe(self) -> bool:
        return self.active_params is not None and self.active_params < self.params

    @property
    def uses_mla(self) -> bool:
        return self.kv_latent_dim is not None


@dataclass(frozen=True)
class VramBreakdown:
    weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    kv_cache_gb: float

    @property
    def total_gb(self) -> float:
        return (
            self.weights_gb
            + self.gradients_gb
            + self.optimizer_states_gb
            + self.activations_gb
            + self.kv_cache_gb
        )


def weights_bytes(model: ModelShape, precision: str) -> float:
    return model.params * bytes_per_param(precision)


def active_weights_bytes(model: ModelShape, precision: str) -> float:
    """Bytes of weights actually touched per token. For dense models this
    equals weights_bytes; for MoE models it's the (much smaller) active
    param subset — the right quantity for per-token memory-bandwidth-bound
    decode cost, as opposed to weights_bytes (all experts, for VRAM sizing).
    """
    return model.effective_active_params * bytes_per_param(precision)


def gradients_bytes(model: ModelShape, precision: str) -> float:
    # Gradients are typically kept in the same precision as the weights being trained.
    return model.params * bytes_per_param(precision)


def optimizer_state_bytes(
    model: ModelShape, optimizer: str = "adam", fp32_master_copy: bool = True
) -> float:
    """Optimizer state size, independent of training precision (mixed-precision
    training keeps optimizer moments/master weights in fp32 for stability).
    """
    if optimizer == "adam":
        moments = 2 * model.params * 4.0  # first + second moment, fp32
    elif optimizer == "sgd_momentum":
        moments = 1 * model.params * 4.0
    elif optimizer == "sgd":
        moments = 0.0
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    master_copy = model.params * 4.0 if fp32_master_copy else 0.0
    return moments + master_copy


def activation_bytes(
    model: ModelShape,
    batch_size: int,
    seq_len: int,
    precision: str,
    checkpointing: bool = False,
) -> float:
    """Per-batch activation memory for training, Megatron-LM approximation
    (Korthikanti et al., "Reducing Activation Recomputation in Large Transformer
    Models"): per-layer bytes ~= seq*batch*hidden*(34 + 5*heads*seq/hidden).
    Activation checkpointing recomputes activations instead of storing them,
    cutting this roughly to the single-layer cost regardless of depth.
    """
    elem_bytes = bytes_per_param(precision)
    per_layer_units = seq_len * batch_size * model.hidden_dim * (
        34 + 5 * model.num_heads * seq_len / model.hidden_dim
    )
    # The "34" constant already assumes ~2-byte activations; rescale for other precisions.
    per_layer_bytes = per_layer_units * (elem_bytes / 2.0)
    if checkpointing:
        return per_layer_bytes  # only one layer's activations live at a time
    return per_layer_bytes * model.num_layers


def kv_cache_bytes_per_token(model: ModelShape, precision: str) -> float:
    """Bytes of KV cache added per token, per sequence (all layers).

    MLA models (kv_latent_dim set) compress K and V into one shared latent
    vector per token per layer — no separate K/V tensors and no per-head
    scaling, so the formula is fundamentally smaller than GQA/MHA. Otherwise
    this is the standard 2 (K&V) * layers * kv_heads * head_dim formula,
    using num_kv_heads instead of num_heads for GQA models (MHA when unset).
    """
    elem_bytes = bytes_per_param(precision)
    if model.uses_mla:
        return model.num_layers * model.kv_latent_dim * elem_bytes
    return 2 * model.num_layers * model.effective_kv_heads * model.head_dim * elem_bytes


def kv_cache_bytes(
    model: ModelShape,
    batch_size: int,
    seq_len: int,
    precision: str,
) -> float:
    """KV cache for autoregressive inference, summed over a batch of sequences."""
    return kv_cache_bytes_per_token(model, precision) * seq_len * batch_size


def _to_gb(b: float) -> float:
    return b / 1e9


def compute_vram_breakdown(
    model: ModelShape,
    precision: str,
    batch_size: int,
    seq_len: int,
    optimizer: str = "adam",
    fp32_master_copy: bool = True,
    checkpointing: bool = False,
    training: bool = True,
) -> VramBreakdown:
    """Full VRAM picture for either a training step or inference-only serving.

    training=False zeroes out gradients/optimizer state (not needed for inference)
    and reports KV cache instead of full-batch training activations.
    """
    weights = weights_bytes(model, precision)

    if training:
        grads = gradients_bytes(model, precision)
        opt = optimizer_state_bytes(model, optimizer, fp32_master_copy)
        acts = activation_bytes(model, batch_size, seq_len, precision, checkpointing)
        kv = 0.0
    else:
        grads = 0.0
        opt = 0.0
        acts = 0.0
        kv = kv_cache_bytes(model, batch_size, seq_len, precision)

    return VramBreakdown(
        weights_gb=_to_gb(weights),
        gradients_gb=_to_gb(grads),
        optimizer_states_gb=_to_gb(opt),
        activations_gb=_to_gb(acts),
        kv_cache_gb=_to_gb(kv),
    )
