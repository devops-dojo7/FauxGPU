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


class VramRequest(BaseModel):
    model: ModelShapeIn
    precision: str = "bf16"
    batch_size: int = 1
    seq_len: int = 2048
    optimizer: str = "adam"
    fp32_master_copy: bool = True
    checkpointing: bool = False
    training: bool = True


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
    batch_size: int = 1
    seq_len: int = 2048
    num_microbatches: int = 1


class CostResponse(BaseModel):
    compute_s_per_step: float
    communication_s_per_step: float
    tp_communication_s_per_step: float
    pipeline_bubble_s_per_step: float
    total_s_per_step: float
    total_gpus: int
    total_steps: int
    total_time_hours: float
    total_cost_usd: float
    cost_per_1k_tokens_usd: float
    power_watts_per_gpu: float
    total_power_kw: float
    total_energy_kwh: float


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


class InferenceResponse(BaseModel):
    ttft_ms: float
    decode_step_ms: float
    tokens_per_sec_per_gpu: float
    prefill_interference_fraction: float
    colocated_tokens_per_sec_per_gpu: float
    disaggregated_tokens_per_sec_per_gpu: float
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


class RunStartRequest(BaseModel):
    model: str
    gpu: str
    topology: str
    total_gpus: int
    compute_s_per_step: float
    communication_s_per_step: float
    total_s_per_step: float
    total_steps: int


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


class K8sAvailabilityResponse(BaseModel):
    available: bool


class LaunchInferenceServerRequest(BaseModel):
    model_preset: str = "llama2-7b"
    gpu_id: str
    precision: str = "bf16"


class LaunchInferenceServerResponse(BaseModel):
    name: str


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
