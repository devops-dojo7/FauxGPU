"""Pydantic request/response models for the API layer. Kept separate from
engine/ so the calc engine itself has zero web-framework dependency."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelShapeIn(BaseModel):
    params: float = Field(..., description="Total parameter count, e.g. 6.74e9 for Llama-2 7B")
    num_layers: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int | None = Field(None, description="GQA KV head count; omit for MHA (num_kv_heads == num_heads)")
    active_params: float | None = Field(None, description="MoE active param count per token; omit for dense models")
    kv_latent_dim: int | None = Field(None, description="MLA compressed KV latent dim per layer; omit for GQA/MHA")
    num_experts: int | None = Field(None, description="MoE total routed-expert count; omit for dense/non-EP models")
    top_k: int | None = Field(None, description="MoE experts routed to per token; omit for dense/non-EP models")


class VramRequest(BaseModel):
    model: ModelShapeIn
    precision: str = "bf16"
    batch_size: int = 1
    seq_len: int = 2048
    optimizer: str = "adam"
    fp32_master_copy: bool = True
    checkpointing: bool = False
    training: bool = True
    zero_stage: int = Field(
        0, ge=0, le=3, description="ZeRO-DP/FSDP model-state sharding stage: 0=off, 1=Pos, 2=Pos+g, 3=Pos+g+p"
    )
    dp_size: int = Field(1, ge=1, description="Data-parallel replica count zero_stage shards model states across")
    peft_method: str = Field(
        "full",
        description=(
            "Parameter-efficient fine-tuning strategy: full (default, every param trainable) | "
            "lora (Hu et al. 2021) | qlora (Dettmers et al. 2023, 4-bit NF4 base + LoRA). "
            "Mutually exclusive with zero_stage != 0 — see engine.memory.compute_vram_breakdown."
        ),
    )
    peft_rank: int = Field(8, ge=1, description="LoRA/QLoRA rank r — only used when peft_method is lora/qlora")
    peft_target_modules: int = Field(
        2, ge=1, le=4, description="Number of attention projection matrices (of Wq/Wk/Wv/Wo) adapted, 1-4"
    )


class VramResponse(BaseModel):
    weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    kv_cache_gb: float
    total_gb: float


class TopologyRequest(BaseModel):
    shape: str = Field(..., description="single_gpu | nvlink_node | multi_node")
    gpu_id: str
    gpus_per_node: int = 1
    num_nodes: int = 1
    fabric_id: str | None = None


class TopologyResponse(BaseModel):
    shape: str
    gpu_id: str
    gpu_name: str
    total_gpus: int
    intra_node_bandwidth_gbps: float | None
    inter_node_bandwidth_gbps: float | None
    bottleneck_bandwidth_gbps: float


class CostRequest(BaseModel):
    model: ModelShapeIn
    topology: TopologyRequest
    tokens_per_step: int = 32768
    total_training_tokens: float = 1e12
    precision: str = "bf16"
    utilization: float = 0.35
    tp_degree: int = 1
    pp_degree: int = 1
    ep_degree: int = Field(
        1,
        ge=1,
        description=(
            "Expert-parallel degree: shards an MoE model's expert pool across this many ranks, "
            "each needing a dispatch+combine all-to-all per MoE layer. A no-op for models with no "
            "num_experts/top_k set (dense models), regardless of this value — see "
            "engine.parallelism.expert_all_to_all_seconds."
        ),
    )
    batch_size: int = 1
    seq_len: int = 2048
    num_microbatches: int = 1
    zero_stage: int = Field(
        0,
        ge=0,
        le=3,
        description=(
            "ZeRO-DP/FSDP model-state sharding stage: 0=off, 1=Pos, 2=Pos+g, 3=Pos+g+p. "
            "Shards across topology's data-parallel replica group (topology.gpus_per_node * "
            "topology.num_nodes, before any tp_degree/pp_degree multiplier) — see engine.memory."
        ),
    )
    carbon_region: str = Field(
        "global-avg", description="Grid carbon-intensity region — see engine.carbon.GRID_CARBON_INTENSITY_G_PER_KWH"
    )
    peft_method: str = Field(
        "full",
        description=(
            "Parameter-efficient fine-tuning strategy: full | lora | qlora. Shrinks the DP "
            "gradient all-reduce to the tiny trainable adapter only. Mutually exclusive with "
            "zero_stage != 0 — see engine.compute.estimate_step_time."
        ),
    )
    peft_rank: int = Field(8, ge=1, description="LoRA/QLoRA rank r — only used when peft_method is lora/qlora")
    peft_target_modules: int = Field(
        2, ge=1, le=4, description="Number of attention projection matrices (of Wq/Wk/Wv/Wo) adapted, 1-4"
    )


class CostResponse(BaseModel):
    compute_s_per_step: float
    communication_s_per_step: float
    tp_communication_s_per_step: float
    pipeline_bubble_s_per_step: float
    expert_communication_s_per_step: float
    total_s_per_step: float
    total_gpus: int
    total_steps: int
    total_time_hours: float
    total_cost_usd: float
    cost_per_1k_tokens_usd: float
    power_watts_per_gpu: float
    total_power_kw: float
    total_energy_kwh: float
    co2e_kg: float
    carbon_region: str
    grid_intensity_g_per_kwh: float
    equivalent_car_km: float
    equivalent_flights_ny_london: float


class SpotPricingRequest(BaseModel):
    on_demand_price_per_hr_usd: float = Field(..., gt=0)
    total_gpus: int = Field(..., gt=0)
    base_time_hours: float = Field(..., ge=0)
    step_time_s: float = Field(..., gt=0)
    checkpoint_interval_steps: int = Field(..., gt=0)
    checkpoint_size_gb: float = Field(..., ge=0)
    preemptions_per_1000_gpu_hours: float = Field(1.0, ge=0)
    discount: float = Field(0.65, ge=0, lt=1)


class SpotPricingResponse(BaseModel):
    on_demand_cost_usd: float
    spot_price_per_hr_usd: float
    expected_preemptions: float
    expected_lost_steps: float
    expected_recovery_overhead_hours: float
    expected_wall_clock_hours: float
    expected_total_cost_usd: float
    savings_usd: float
    savings_pct: float


class InferenceRequest(BaseModel):
    model: ModelShapeIn
    gpu_id: str
    precision: str = "bf16"
    prompt_tokens: int = 2048
    output_tokens: int = 256
    decode_batch_size: int = 8
    requests_per_sec: float = 1.0
    cache_hit_fraction: float = 0.0
    utilization: float = 0.35
    paged_attention: bool = False
    block_size: int = 16
    gpu_memory_utilization: float = 0.9
    tp_degree: int = 1


class InferenceResponse(BaseModel):
    ttft_ms: float
    decode_step_ms: float
    tokens_per_sec_per_gpu: float
    prefill_interference_fraction: float
    colocated_tokens_per_sec_per_gpu: float
    disaggregated_tokens_per_sec_per_gpu: float
    tp_communication_overhead_fraction: float
    tokens_per_sec_per_gpu_amortized: float
    usable_vram_gb: float
    weights_gb: float
    kv_budget_gb: float
    max_concurrent_sequences: int


class SpeculativeDecodingRequest(BaseModel):
    draft_model: ModelShapeIn
    target_model: ModelShapeIn
    gpu_id: str
    precision: str = "bf16"
    gamma: int = 4
    acceptance_rate: float = 0.7
    avg_kv_tokens: float = 1024
    batch_size: int = 1


class SpeculativeDecodingResponse(BaseModel):
    draft_step_ms: float
    verify_step_ms: float
    round_ms: float
    expected_tokens_per_round: float
    speculative_tokens_per_sec: float
    baseline_tokens_per_sec: float
    speedup: float


class RecommendRequest(BaseModel):
    model: ModelShapeIn
    precision: str = "bf16"
    tokens_per_step: int = 32768
    total_training_tokens: float = 1e12
    objective: str = Field("cost", description="cost | time")
    batch_size: int = 1
    seq_len: int = 2048
    utilization: float = 0.35
    num_microbatches: int = 1
    max_gpus: int = 64
    budget_usd: float | None = None
    max_time_hours: float | None = None
    candidate_gpu_ids: list[str] | None = None


class RecommendationCandidateOut(BaseModel):
    gpu_id: str
    gpu_name: str
    num_gpus: int
    tp_degree: int
    pp_degree: int
    ep_degree: int
    total_cost_usd: float
    total_time_hours: float
    cost_per_1k_tokens_usd: float
    vram_per_gpu_gb: float
    vram_headroom_gb: float


class RecommendResponse(BaseModel):
    candidates: list[RecommendationCandidateOut]


class EconomicsPointOut(BaseModel):
    step: int
    elapsed_hours: float
    cost_usd: float
    cumulative_cost_usd: float
    utilization_pct: float


class EconomicsSummaryOut(BaseModel):
    total_cost_usd: float
    total_gpu_hours: float
    avg_utilization_pct: float
    idle_cost_usd: float
    idle_pct: float


class EconomicsResponse(BaseModel):
    points: list[EconomicsPointOut]
    summary: EconomicsSummaryOut


class SchedJobIn(BaseModel):
    job_id: str
    team: str
    priority: int = Field(..., description="Higher = more important")
    gpu_count: int = Field(..., gt=0)
    submit_time: float = Field(..., ge=0)
    duration: float = Field(..., gt=0)


class SchedulerRequest(BaseModel):
    jobs: list[SchedJobIn]
    gpu_id: str
    total_gpus: int = Field(..., gt=0)
    preemption_enabled: bool = True
    horizon: float | None = None


class TimelineSegmentOut(BaseModel):
    start: float
    end: float


class JobOutcomeOut(BaseModel):
    job_id: str
    team: str
    segments: list[TimelineSegmentOut]
    final_status: str
    wait_time_total: float
    preempted_count: int


class SchedulerResponse(BaseModel):
    jobs: list[JobOutcomeOut]
    pool_total_gpus: int
    makespan: float
    gpu_utilization_pct: float


class TraceReplayRequest(BaseModel):
    trace_csv: str
    gpu_id: str
    total_gpus: int = Field(..., gt=0)
    preemption_enabled: bool = True
    horizon: float | None = None


class RunStartRequest(BaseModel):
    model: str
    gpu: str
    topology: str
    total_gpus: int
    compute_s_per_step: float
    communication_s_per_step: float
    total_s_per_step: float
    total_steps: int
    fabric_id: str | None = None
    payload_gb: float = 0.0


class FabricContentionStatus(BaseModel):
    communication_s: float
    bandwidth_share_gbps: float
    gpus_sharing_fabric: int
    jobs_sharing_fabric: int


class RunStepRequest(BaseModel):
    step: int
    tokens_seen: int
    elapsed_s: float
    active_gpus: int | None = None


class ChaosInjectRequest(BaseModel):
    kind: str = Field(..., description="xid_error | nvlink_degradation | node_drain")
    severity: float
    duration_steps: int | None = None


class ChaosEventOut(BaseModel):
    event_id: str
    kind: str
    injected_at_step: int
    duration_steps: int | None
    severity: float


class RunSummary(BaseModel):
    run_id: str
    status: str
    meta: RunStartRequest | None
    latest_step: RunStepRequest | None
    started_at: float
    updated_at: float


class CheckpointEventOut(BaseModel):
    event_id: str
    kind: str = Field(..., description="save | restore")
    step: int
    size_gb: float
    overhead_s: float
    steps_lost: int | None = None


class RunDetail(RunSummary):
    steps: list[RunStepRequest]
    events: list[ChaosEventOut]
    checkpoint_events: list[CheckpointEventOut]


class SimulateRunRequest(BaseModel):
    model: ModelShapeIn
    model_label: str = "custom"
    topology: TopologyRequest
    precision: str = "bf16"
    tokens_per_step: int = 32768
    total_steps: int = 50
    speedup: float = 20.0
    utilization: float = 0.35
    checkpoint_interval_steps: int | None = None


class InferenceStreamRequest(BaseModel):
    model: ModelShapeIn
    model_label: str = "custom"
    gpu_id: str
    precision: str = "bf16"
    prompt: str
    prompt_tokens: int = Field(..., description="Pre-tokenized prompt length; playground computes this client-side")
    max_output_tokens: int = 80
    cache_hit_fraction: float = 0.0
    utilization: float = 0.35
    tp_degree: int = 1


class K8sAvailabilityResponse(BaseModel):
    available: bool


class LaunchInferenceServerRequest(BaseModel):
    model_preset: str = "llama2-7b"
    gpu_id: str
    precision: str = "bf16"


class LaunchInferenceServerResponse(BaseModel):
    name: str


class K8sCompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 80


class K8sCompletionResult(BaseModel):
    text: str
    ttft_s: float
    tokens_per_sec: float
    cost_usd: float
    round_trip_ms: float


class LaunchK8sJobResponse(BaseModel):
    job_name: str
    run_id: str


class MigGpuOut(BaseModel):
    gpu_id: str
    gpu_name: str


class MigProfileOut(BaseModel):
    id: str
    compute_slots: int
    memory_slots: int
    memory_gb: float


class MigRequestIn(BaseModel):
    request_id: str
    tenant: str
    profile_id: str


class MigPlacementOut(BaseModel):
    request_id: str
    tenant: str
    profile_id: str
    gpu_index: int
    compute_slots: int
    memory_slots: int


class MigPackRequest(BaseModel):
    gpu_id: str
    pool_size: int = Field(..., gt=0)
    requests: list[MigRequestIn]


class MigPackResponse(BaseModel):
    placements: list[MigPlacementOut]
    unplaced: list[MigRequestIn]
    pool_size: int
    gpus_used: int
    compute_utilization_pct: float
    memory_utilization_pct: float


class TrafficStageIn(BaseModel):
    duration_s: float = Field(..., gt=0)
    target_rps: float = Field(..., ge=0)


class AutoscalingConfigIn(BaseModel):
    min_replicas: int = Field(..., ge=1)
    max_replicas: int = Field(..., ge=1)
    target_utilization_pct: float = 70.0
    eval_interval_s: float = 15.0
    scale_down_stabilization_s: float = 300.0


class AutoscalingRequest(BaseModel):
    model: ModelShapeIn
    gpu_id: str
    precision: str = "bf16"
    prompt_tokens: int = 2048
    output_tokens: int = 256
    decode_batch_size: int = 8
    cache_hit_fraction: float = 0.0
    utilization: float = 0.35
    paged_attention: bool = False
    block_size: int = 16
    stages: list[TrafficStageIn]
    config: AutoscalingConfigIn
    tick_s: float = 5.0


class AutoscalingPointOut(BaseModel):
    t: float
    demand_rps: float
    replicas: int
    capacity_rps: float
    backlog_requests: float
    est_queue_delay_s: float


class AutoscalingResponse(BaseModel):
    points: list[AutoscalingPointOut]
    per_replica_capacity_rps: float
    peak_replicas: int
    peak_backlog_requests: float
    peak_queue_delay_s: float


class NetworkJobIn(BaseModel):
    job_id: str
    team: str
    num_gpus: int = Field(..., gt=0)
    payload_gb: float = Field(..., gt=0)


class NetworkContentionRequest(BaseModel):
    fabric_id: str
    jobs: list[NetworkJobIn]


class NetworkJobResultOut(BaseModel):
    job_id: str
    team: str
    num_gpus: int
    bandwidth_share_gbps: float
    isolated_comm_s: float
    contended_comm_s: float
    slowdown_factor: float


class NetworkContentionResponse(BaseModel):
    fabric_id: str
    fabric_bandwidth_gbps: float
    total_gpus_sharing_fabric: int
    jobs: list[NetworkJobResultOut]


# --- Bring-your-own-key AI features (chat assistant, NL recommender, NL trace generator) ---


class AiProviderStatus(BaseModel):
    provider: str
    configured: bool


class AiProviderKeyIn(BaseModel):
    api_key: str


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str


class ChatRequest(BaseModel):
    provider: str
    model: str
    messages: list[ChatMessage]


class RecommendNlRequest(BaseModel):
    provider: str
    model: str
    prompt: str


class TraceGenerateNlRequest(BaseModel):
    provider: str
    model: str
    prompt: str


class TraceGenerateNlResponse(BaseModel):
    trace_csv: str
