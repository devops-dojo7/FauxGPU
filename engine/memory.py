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


def _optimizer_state_bytes_for_params(
    num_params: float, optimizer: str = "adam", fp32_master_copy: bool = True
) -> float:
    """Core optimizer-state formula, parametrized on a raw param count rather
    than a full ModelShape — shared by optimizer_state_bytes (the whole
    model) and the LoRA/QLoRA path below (just the adapter's much smaller
    trainable-param count, per Hu et al. 2021 — see lora_trainable_params).
    """
    if optimizer == "adam":
        moments = 2 * num_params * 4.0  # first + second moment, fp32
    elif optimizer == "sgd_momentum":
        moments = 1 * num_params * 4.0
    elif optimizer == "sgd":
        moments = 0.0
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")
    master_copy = num_params * 4.0 if fp32_master_copy else 0.0
    return moments + master_copy


def optimizer_state_bytes(
    model: ModelShape, optimizer: str = "adam", fp32_master_copy: bool = True
) -> float:
    """Optimizer state size, independent of training precision (mixed-precision
    training keeps optimizer moments/master weights in fp32 for stability).
    """
    return _optimizer_state_bytes_for_params(model.params, optimizer, fp32_master_copy)


# --- LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685) / QLoRA
# (Dettmers et al. 2023, https://arxiv.org/abs/2305.14314) parameter-efficient
# fine-tuning ---

def lora_trainable_params(model: ModelShape, rank: int, target_modules: int = 2) -> float:
    """Number of trainable LoRA parameters across the adapted attention
    projection matrices. For a linear layer W in R^(d_out x d_in), LoRA's
    B*A decomposition (B in R^(d_out x r), A in R^(r x d_in)) contributes
    r*(d_out + d_in) trainable params (Hu et al. 2021, Sec 4.1). Summed
    across every adapted layer:

      |Theta| = num_layers * sum_over_adapted_matrices(r * (d_out + d_in))

    target_modules selects which matrices are adapted, matching this
    project's own UI dropdown mapping: 1=Wq, 2=+Wv, 3=+Wk, 4=+Wo — the
    paper's own formula (Sec 5.1) is a special case of this for plain MHA
    (multi-head attention, all four matrices square at hidden_dim x
    hidden_dim): |Theta| = 2 * L_hat * d_model * r, where L_hat =
    num_layers * target_modules. Verified against the paper's own Table 5
    worked example on GPT-3 175B (a plain-MHA model, hidden_dim 12288, 96
    layers): target_modules=1, rank=8 gives exactly 18,874,368 (paper
    reports "~18M") — see tests/test_memory.py.

    For grouped-query attention (GQA, num_kv_heads set below num_heads —
    the majority of this project's own model catalog, e.g. Llama-3.x,
    Qwen, Falcon, Gemma), Wk/Wv project to a *smaller* dimension than Wq/Wo
    (num_kv_heads * head_dim, not hidden_dim) — the same distinction
    engine.memory.kv_cache_bytes_per_token already makes via
    effective_kv_heads. Modeling every adapted matrix as square would
    overestimate a GQA model's adapter size by 25-33% for target_modules
    in {2, 3} (verified against real catalog presets in
    tests/test_memory.py) — this function gets it right for both MHA and
    GQA, reducing to the paper's own exact formula when num_kv_heads is
    unset (MHA) and num_heads * head_dim == hidden_dim.

    For multi-head latent attention (MLA, kv_latent_dim set — DeepSeek-V3/
    R1, Kimi K2/K3 in this project's catalog), there is no conventional
    per-head Wk/Wv at all: K/V are compressed through a shared down-
    projection to kv_latent_dim before being up-projected back per-head
    (DeepSeek-AI et al. 2024, DeepSeek-V2, https://arxiv.org/abs/2405.04434).
    Adapting the down-projection (hidden_dim -> kv_latent_dim, the
    dominant KV-side parameter/adapter cost and the same "K/V compressed
    to one shared latent" simplification engine.memory.
    kv_cache_bytes_per_token already uses) is this teaching tool's level
    of MLA fidelity for target_modules 2/3 — using kv_latent_dim in place
    of effective_kv_heads*head_dim avoids a real ~50% overestimate a naive
    head-count-based formula would otherwise produce for these presets
    (kv_latent_dim=576 is far smaller than num_heads*head_dim=16384 for
    DeepSeek-V3's published shape).
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1; got {rank}")
    if target_modules not in (1, 2, 3, 4):
        raise ValueError(f"target_modules must be 1-4 (of Wq/Wk/Wv/Wo); got {target_modules}")

    query_output_dim = model.num_heads * model.head_dim  # Wq's (and Wo's input's) actual output dim
    # Wk/Wv's actual output dim: MLA's compressed latent when present
    # (kv_latent_dim, far smaller than a head-count-based formula would
    # give), else GQA's (possibly shrunk) effective_kv_heads * head_dim.
    kv_output_dim = model.kv_latent_dim if model.uses_mla else model.effective_kv_heads * model.head_dim

    # (d_out, d_in) per adapted matrix, in this project's own UI ordering.
    matrix_shapes = [(query_output_dim, model.hidden_dim)]  # Wq
    if target_modules >= 2:
        matrix_shapes.append((kv_output_dim, model.hidden_dim))  # Wv
    if target_modules >= 3:
        matrix_shapes.append((kv_output_dim, model.hidden_dim))  # Wk (same shape as Wv)
    if target_modules >= 4:
        matrix_shapes.append((model.hidden_dim, query_output_dim))  # Wo

    per_layer_params = sum(rank * (d_out + d_in) for d_out, d_in in matrix_shapes)
    return model.num_layers * per_layer_params


# QLoRA's block-wise quantization overhead (Dettmers et al. 2023, Sec 3):
# a 32-bit quantization constant per 64-value block costs 32/64 = 0.5 bits/
# param on top of the storage dtype itself. Double Quantization re-quantizes
# those constants (8-bit codes, blocksize 256), cutting the overhead to
# 8/64 + 32/(64*256) = 0.127 bits/param — the paper's own reported "average
# of about 0.37 bits per parameter" saved (0.5 - 0.127 = 0.373).
QLORA_BLOCKWISE_OVERHEAD_BITS = 0.5
QLORA_DOUBLE_QUANT_OVERHEAD_BITS = 0.127


def quantized_weight_bytes(model: ModelShape, quant_bits: float = 4.0, double_quant: bool = True) -> float:
    """Frozen base-model weight size under QLoRA's block-wise k-bit
    quantization (default 4-bit NormalFloat, quant_bits=4.0), including the
    per-parameter quantization-constant storage overhead. Verified against
    the paper's own reported ~3GB saving from Double Quantization at 65B
    params (65e9 * 0.373 bits / 8 == ~3.03GB) — see tests/test_memory.py.
    """
    overhead_bits = QLORA_DOUBLE_QUANT_OVERHEAD_BITS if double_quant else QLORA_BLOCKWISE_OVERHEAD_BITS
    bits_per_param = quant_bits + overhead_bits
    return model.params * bits_per_param / 8.0


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


PEFT_METHODS = ("full", "lora", "qlora")


def compute_vram_breakdown(
    model: ModelShape,
    precision: str,
    batch_size: int,
    seq_len: int,
    optimizer: str = "adam",
    fp32_master_copy: bool = True,
    checkpointing: bool = False,
    training: bool = True,
    zero_stage: int = 0,
    dp_size: int = 1,
    peft_method: str = "full",
    peft_rank: int = 8,
    peft_target_modules: int = 2,
) -> VramBreakdown:
    """Full VRAM picture for either a training step or inference-only serving.

    training=False zeroes out gradients/optimizer state (not needed for inference)
    and reports KV cache instead of full-batch training activations.

    zero_stage/dp_size model ZeRO-DP (Rajbhandari et al. 2020,
    https://arxiv.org/abs/1910.02054) / FSDP's equivalent sharding of model
    states across a data-parallel group of `dp_size` ranks — each rank's
    per-GPU memory drops as more of the model's states get sharded rather
    than replicated:

      0 (off, default): every rank holds a full copy of everything —
        today's non-ZeRO default DP behavior, and the only mode inference
        serving (training=False) supports (ZeRO is a training-time
        optimizer/gradient sharding technique; nothing to shard once
        there's no optimizer state or gradients).
      1 (Pos): optimizer states sharded across dp_size ranks. 4x memory
        reduction at Nd=64 in the paper's own worked example (Adam,
        mixed-precision, K=12) — same communication volume as plain DP.
      2 (Pos+g): + gradients also sharded. 8x reduction at Nd=64 — still
        the same communication volume as plain DP.
      3 (Pos+g+p): + parameters (weights) also sharded — this is what
        FSDP calls "full sharding" and is its default mode. Memory drops
        linearly with dp_size (up to ~64x at Nd=64 in the paper's
        example), at the cost of a "modest 50% increase" in communication
        volume the paper itself reports (see
        engine.compute.zero_communication_multiplier for where that's
        applied to step time).

    Verified against the paper's own Table 1 (a 7.5B model, Nd=64, mixed-
    precision Adam): this function's stage-1/2/3 outputs land at 31.4GB /
    16.6GB / 1.88GB, matching the paper exactly (see tests/test_memory.py).

    Only ZeRO-DP (model-state sharding) is modeled here — ZeRO-R's other
    optimizations (partitioned activation checkpointing, which additionally
    requires a model-parallel/tensor-parallel degree, not just dp_size) are
    out of scope.

    peft_method selects a parameter-efficient fine-tuning strategy in place
    of full fine-tuning — the other reason (besides ZeRO) frontier-scale
    fine-tuning is affordable at all:

      "full" (default): every parameter is trainable — today's existing
        behavior, unaffected by peft_rank/peft_target_modules.
      "lora": LoRA (Hu et al. 2021, https://arxiv.org/abs/2106.09685).
        Base weights stay frozen (still fully resident, in `precision`);
        only a tiny pair of rank-`peft_rank` decomposition matrices per
        adapted matrix are trainable (see engine.memory.lora_trainable_params
        for the exact count, verified against the paper's own Table 5).
        Gradients and optimizer state are computed for that tiny trainable
        count only, not the full model — this is the "up to 2/3 VRAM
        reduction" the paper itself reports, since there's no optimizer
        state to keep for the (vast majority) frozen parameters.
      "qlora": QLoRA (Dettmers et al. 2023, https://arxiv.org/abs/2305.14314).
        LoRA's same tiny trainable adapter, but the frozen base weights are
        additionally stored in 4-bit NormalFloat with Double Quantization
        (see engine.memory.quantized_weight_bytes) instead of `precision` —
        the combination that makes a 65B model fit on a single 48GB GPU,
        per the paper's own headline result.

    peft_rank/peft_target_modules only affect the "lora"/"qlora" paths —
    LoRA's adapter weights are always kept in `precision` (the paper's own
    "computation data type", e.g. bf16) regardless of the base weights'
    storage format, since the adapter is what actually receives gradients.

    zero_stage/dp_size and peft_method are mutually exclusive here (a
    zero_stage other than 0 combined with a non-"full" peft_method raises
    ValueError) — real systems do combine ZeRO/FSDP with LoRA at very large
    scale, but modeling that interaction precisely is out of scope for this
    teaching tool; each is fully modeled on its own.
    """
    if zero_stage not in (0, 1, 2, 3):
        raise ValueError(f"zero_stage must be 0, 1, 2, or 3; got {zero_stage}")
    if dp_size < 1:
        raise ValueError(f"dp_size must be >= 1; got {dp_size}")
    if peft_method not in PEFT_METHODS:
        raise ValueError(f"peft_method must be one of {PEFT_METHODS}; got {peft_method!r}")
    if peft_method != "full" and zero_stage != 0:
        raise ValueError(
            f"zero_stage={zero_stage} combined with peft_method={peft_method!r} is not modeled "
            "(mutually exclusive here) — see compute_vram_breakdown's docstring."
        )

    if peft_method == "qlora":
        weights = quantized_weight_bytes(model)
    else:
        weights = weights_bytes(model, precision)

    if peft_method in ("lora", "qlora"):
        adapter_params = lora_trainable_params(model, peft_rank, peft_target_modules)
        weights += adapter_params * bytes_per_param(precision)

    if training:
        if peft_method in ("lora", "qlora"):
            # Only the tiny adapter receives gradients/optimizer state — the
            # frozen base model needs neither, which is LoRA's whole point.
            grads = adapter_params * bytes_per_param(precision)
            opt = _optimizer_state_bytes_for_params(adapter_params, optimizer, fp32_master_copy)
        else:
            grads = gradients_bytes(model, precision)
            opt = optimizer_state_bytes(model, optimizer, fp32_master_copy)
        acts = activation_bytes(model, batch_size, seq_len, precision, checkpointing)
        kv = 0.0

        if zero_stage >= 1:
            opt /= dp_size
        if zero_stage >= 2:
            grads /= dp_size
        if zero_stage >= 3:
            weights /= dp_size
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
