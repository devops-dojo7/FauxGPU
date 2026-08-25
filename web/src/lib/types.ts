export interface GpuSpec {
  id: string;
  vendor: string;
  name: string;
  vram_gb: number;
  mem_bandwidth_gbps: number;
  bf16_tflops: number;
  fp8_tflops: number | null;
  nvlink_gbps: number | null;
  price_per_hr_usd: number;
  tdp_watts: number;
  idle_watts: number;
  // Compute-die internals — null where the vendor hasn't published an exact
  // figure (e.g. Blackwell's per-die SM/core counts for the dual-die B200).
  architecture: string;
  process_node: string;
  launch_year: number | null;
  transistors_b: number | null;
  sm_count: number | null;
  core_count: number | null;
  core_label: string;
  matrix_core_count: number | null;
  matrix_core_label: string | null;
  boost_clock_ghz: number | null;
  l2_cache_mb: number | null;
  interconnect_name: string | null;
  // Set only for devices priced as a one-time purchase rather than a cloud
  // rental (e.g. DGX Spark) — explains what price_per_hr_usd means there.
  price_note: string | null;
}

export interface Fabric {
  id: string;
  name: string;
  bandwidth_gbps: number;
  latency_us: number;
}

export interface ModelShape {
  params: number;
  num_layers: number;
  hidden_dim: number;
  num_heads: number;
  head_dim: number;
  // Optional: unset = plain dense MHA model.
  num_kv_heads?: number | null; // GQA: KV head count, smaller than num_heads
  active_params?: number | null; // MoE: params touched per token (vs. total `params`)
  kv_latent_dim?: number | null; // MLA: compressed KV latent dim per layer (replaces head-count KV formula)
}

export interface VramRequest {
  model: ModelShape;
  precision: string;
  batch_size: number;
  seq_len: number;
  optimizer: string;
  fp32_master_copy: boolean;
  checkpointing: boolean;
  training: boolean;
}

export interface VramResponse {
  weights_gb: number;
  gradients_gb: number;
  optimizer_states_gb: number;
  activations_gb: number;
  kv_cache_gb: number;
  total_gb: number;
}

export type TopologyShape = "single_gpu" | "nvlink_node" | "multi_node";

export interface TopologyRequest {
  shape: TopologyShape;
  gpu_id: string;
  gpus_per_node: number;
  num_nodes: number;
  fabric_id: string | null;
}

export interface TopologyResponse {
  shape: TopologyShape;
  gpu_id: string;
  gpu_name: string;
  total_gpus: number;
  intra_node_bandwidth_gbps: number | null;
  inter_node_bandwidth_gbps: number | null;
  bottleneck_bandwidth_gbps: number;
}

export interface CostRequest {
  model: ModelShape;
  topology: TopologyRequest;
  tokens_per_step: number;
  total_training_tokens: number;
  precision: string;
  utilization: number;
  tp_degree: number;
  pp_degree: number;
  batch_size: number;
  seq_len: number;
  num_microbatches: number;
}

export interface CostResponse {
  compute_s_per_step: number;
  communication_s_per_step: number;
  tp_communication_s_per_step: number;
  pipeline_bubble_s_per_step: number;
  total_s_per_step: number;
  total_gpus: number;
  total_steps: number;
  total_time_hours: number;
  total_cost_usd: number;
  cost_per_1k_tokens_usd: number;
  power_watts_per_gpu: number;
  total_power_kw: number;
  total_energy_kwh: number;
}

export interface InferenceRequest {
  model: ModelShape;
  gpu_id: string;
  precision: string;
  prompt_tokens: number;
  output_tokens: number;
  decode_batch_size: number;
  requests_per_sec: number;
  cache_hit_fraction: number;
  utilization: number;
  paged_attention: boolean;
  block_size: number;
  gpu_memory_utilization: number;
}

export interface InferenceResponse {
  ttft_ms: number;
  decode_step_ms: number;
  tokens_per_sec_per_gpu: number;
  prefill_interference_fraction: number;
  colocated_tokens_per_sec_per_gpu: number;
  disaggregated_tokens_per_sec_per_gpu: number;
  usable_vram_gb: number;
  weights_gb: number;
  kv_budget_gb: number;
  max_concurrent_sequences: number;
}

export interface RunMeta {
  model: string;
  gpu: string;
  topology: string;
  total_gpus: number;
  compute_s_per_step: number;
  communication_s_per_step: number;
  total_s_per_step: number;
  total_steps: number;
}

export interface RunStep {
  step: number;
  tokens_seen: number;
  elapsed_s: number;
  active_gpus: number | null;
}

export type ChaosKind = "xid_error" | "nvlink_degradation" | "node_drain";

export interface ChaosInjectRequest {
  kind: ChaosKind;
  severity: number;
  duration_steps: number | null;
}

export interface ChaosEvent {
  event_id: string;
  kind: ChaosKind;
  injected_at_step: number;
  duration_steps: number | null;
  severity: number;
}

export interface RunSummary {
  run_id: string;
  status: "running" | "done" | "stopped";
  meta: RunMeta | null;
  latest_step: RunStep | null;
  started_at: number;
  updated_at: number;
}

export interface CheckpointEvent {
  event_id: string;
  kind: "save" | "restore";
  step: number;
  size_gb: number;
  overhead_s: number;
  steps_lost: number | null;
}

export interface RunDetail extends RunSummary {
  steps: RunStep[];
  events: ChaosEvent[];
  checkpoint_events: CheckpointEvent[];
}

export interface SpeculativeDecodingRequest {
  draft_model: ModelShape;
  target_model: ModelShape;
  gpu_id: string;
  precision: string;
  gamma: number;
  acceptance_rate: number;
  avg_kv_tokens: number;
  batch_size: number;
}

export interface SpeculativeDecodingResponse {
  draft_step_ms: number;
  verify_step_ms: number;
  round_ms: number;
  expected_tokens_per_round: number;
  speculative_tokens_per_sec: number;
  baseline_tokens_per_sec: number;
  speedup: number;
}

export interface InferenceStreamRequest {
  model: ModelShape;
  model_label: string;
  gpu_id: string;
  precision: string;
  prompt: string;
  prompt_tokens: number;
  max_output_tokens: number;
  cache_hit_fraction: number;
  utilization: number;
}

export interface K8sAvailability {
  available: boolean;
}

export interface LaunchK8sJobResponse {
  job_name: string;
  run_id: string;
}

export interface RecommendRequest {
  model: ModelShape;
  precision: string;
  tokens_per_step: number;
  total_training_tokens: number;
  objective: "cost" | "time";
  batch_size: number;
  seq_len: number;
  utilization: number;
  num_microbatches: number;
  max_gpus: number;
  budget_usd: number | null;
  max_time_hours: number | null;
  candidate_gpu_ids: string[] | null;
}

export interface RecommendationCandidate {
  gpu_id: string;
  gpu_name: string;
  num_gpus: number;
  tp_degree: number;
  pp_degree: number;
  total_cost_usd: number;
  total_time_hours: number;
  cost_per_1k_tokens_usd: number;
  vram_per_gpu_gb: number;
  vram_headroom_gb: number;
}

export interface RecommendResponse {
  candidates: RecommendationCandidate[];
}

export interface EconomicsPoint {
  step: number;
  elapsed_hours: number;
  cost_usd: number;
  cumulative_cost_usd: number;
  utilization_pct: number;
}

export interface EconomicsSummary {
  total_cost_usd: number;
  total_gpu_hours: number;
  avg_utilization_pct: number;
  idle_cost_usd: number;
  idle_pct: number;
}

export interface EconomicsResponse {
  points: EconomicsPoint[];
  summary: EconomicsSummary;
}

export interface SchedJob {
  job_id: string;
  team: string;
  priority: number;
  gpu_count: number;
  submit_time: number;
  duration: number;
}

export interface SchedulerRequest {
  jobs: SchedJob[];
  gpu_id: string;
  total_gpus: number;
  preemption_enabled: boolean;
  horizon: number | null;
}

export interface TimelineSegment {
  start: number;
  end: number;
}

export interface JobOutcome {
  job_id: string;
  team: string;
  segments: TimelineSegment[];
  final_status: "completed" | "incomplete" | "never_started";
  wait_time_total: number;
  preempted_count: number;
}

export interface SchedulerResponse {
  jobs: JobOutcome[];
  pool_total_gpus: number;
  makespan: number;
  gpu_utilization_pct: number;
}

export interface TraceReplayRequest {
  trace_csv: string;
  gpu_id: string;
  total_gpus: number;
  preemption_enabled: boolean;
  horizon: number | null;
}

export interface SimulateRunRequest {
  model: ModelShape;
  model_label: string;
  topology: TopologyRequest;
  precision: string;
  tokens_per_step: number;
  total_steps: number;
  speedup: number;
  utilization: number;
  checkpoint_interval_steps: number | null;
}

export interface MigGpu {
  gpu_id: string;
  gpu_name: string;
}

export interface MigProfile {
  id: string;
  compute_slots: number;
  memory_slots: number;
  memory_gb: number;
}

export interface MigRequest {
  request_id: string;
  tenant: string;
  profile_id: string;
}

export interface MigPlacement {
  request_id: string;
  tenant: string;
  profile_id: string;
  gpu_index: number;
  compute_slots: number;
  memory_slots: number;
}

export interface MigPackRequest {
  gpu_id: string;
  pool_size: number;
  requests: MigRequest[];
}

export interface MigPackResponse {
  placements: MigPlacement[];
  unplaced: MigRequest[];
  pool_size: number;
  gpus_used: number;
  compute_utilization_pct: number;
  memory_utilization_pct: number;
}

export interface TrafficStage {
  duration_s: number;
  target_rps: number;
}

export interface AutoscalingConfig {
  min_replicas: number;
  max_replicas: number;
  target_utilization_pct: number;
  eval_interval_s: number;
  scale_down_stabilization_s: number;
}

export interface AutoscalingRequest {
  model: ModelShape;
  gpu_id: string;
  precision: string;
  prompt_tokens: number;
  output_tokens: number;
  decode_batch_size: number;
  cache_hit_fraction: number;
  utilization: number;
  paged_attention: boolean;
  block_size: number;
  stages: TrafficStage[];
  config: AutoscalingConfig;
  tick_s: number;
}

export interface AutoscalingPoint {
  t: number;
  demand_rps: number;
  replicas: number;
  capacity_rps: number;
  backlog_requests: number;
  est_queue_delay_s: number;
}

export interface AutoscalingResponse {
  points: AutoscalingPoint[];
  per_replica_capacity_rps: number;
  peak_replicas: number;
  peak_backlog_requests: number;
  peak_queue_delay_s: number;
}

export interface NetworkJob {
  job_id: string;
  team: string;
  num_gpus: number;
  payload_gb: number;
}

export interface NetworkContentionRequest {
  fabric_id: string;
  jobs: NetworkJob[];
}

export interface NetworkJobResult {
  job_id: string;
  team: string;
  num_gpus: number;
  bandwidth_share_gbps: number;
  isolated_comm_s: number;
  contended_comm_s: number;
  slowdown_factor: number;
}

export interface NetworkContentionResponse {
  fabric_id: string;
  fabric_bandwidth_gbps: number;
  total_gpus_sharing_fabric: number;
  jobs: NetworkJobResult[];
}

export interface ModelPreset extends ModelShape {
  id: string;
  label: string;
}

// Specs below are approximate, from public model cards/papers — illustrative
// for comparing relative scale and architecture (dense vs. GQA vs. MoE vs.
// MLA), not guaranteed to reproduce exact vendor numbers.
export const MODEL_PRESETS: ModelPreset[] = [
  // Small / draft-sized — speculative-decoding drafts and quick sanity checks.
  { id: "qwen2.5-0.5b", label: "Qwen2.5 0.5B (draft-sized)", params: 0.49e9, num_layers: 24, hidden_dim: 896, num_heads: 14, head_dim: 64, num_kv_heads: 2 },
  { id: "qwen3-0.6b", label: "Qwen3 0.6B (draft-sized)", params: 0.6e9, num_layers: 28, hidden_dim: 1024, num_heads: 16, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.2-1b", label: "Llama-3.2 1B (draft-sized)", params: 1.24e9, num_layers: 16, hidden_dim: 2048, num_heads: 32, head_dim: 64, num_kv_heads: 8 },
  { id: "qwen2.5-1.5b", label: "Qwen2.5 1.5B", params: 1.54e9, num_layers: 28, hidden_dim: 1536, num_heads: 12, head_dim: 128, num_kv_heads: 2 },
  { id: "qwen3-1.7b", label: "Qwen3 1.7B", params: 1.7e9, num_layers: 28, hidden_dim: 2048, num_heads: 16, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma3-1b", label: "Gemma 3 1B", params: 1.0e9, num_layers: 26, hidden_dim: 1152, num_heads: 4, head_dim: 256, num_kv_heads: 1 },
  { id: "llama3.2-3b", label: "Llama-3.2 3B", params: 3.21e9, num_layers: 28, hidden_dim: 3072, num_heads: 24, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen2.5-3b", label: "Qwen2.5 3B", params: 3.09e9, num_layers: 36, hidden_dim: 2048, num_heads: 16, head_dim: 128, num_kv_heads: 2 },
  { id: "qwen3-4b", label: "Qwen3 4B", params: 4.02e9, num_layers: 36, hidden_dim: 2560, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma3-4b", label: "Gemma 3 4B", params: 4.3e9, num_layers: 34, hidden_dim: 2560, num_heads: 8, head_dim: 256, num_kv_heads: 4 },
  { id: "phi-3-mini", label: "Phi-3-mini 3.8B", params: 3.8e9, num_layers: 32, hidden_dim: 3072, num_heads: 32, head_dim: 96 },

  // Classic dense open models (Llama-2 era and earlier) — plain MHA, no GQA.
  { id: "gpt2-xl", label: "GPT-2 XL 1.5B", params: 1.5e9, num_layers: 48, hidden_dim: 1600, num_heads: 25, head_dim: 64 },
  { id: "gptj-6b", label: "GPT-J 6B", params: 6.05e9, num_layers: 28, hidden_dim: 4096, num_heads: 16, head_dim: 256 },
  { id: "llama2-7b", label: "Llama-2 7B", params: 6.74e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
  { id: "mpt-7b", label: "MPT-7B", params: 6.65e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
  { id: "mistral-7b", label: "Mistral 7B", params: 7.3e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
  { id: "falcon-7b", label: "Falcon-7B", params: 7.22e9, num_layers: 32, hidden_dim: 4544, num_heads: 71, head_dim: 64, num_kv_heads: 1 },
  { id: "llama2-13b", label: "Llama-2 13B", params: 13.0e9, num_layers: 40, hidden_dim: 5120, num_heads: 40, head_dim: 128 },
  { id: "gpt-neox-20b", label: "GPT-NeoX-20B", params: 20.0e9, num_layers: 44, hidden_dim: 6144, num_heads: 64, head_dim: 96 },
  { id: "mpt-30b", label: "MPT-30B", params: 30.0e9, num_layers: 48, hidden_dim: 7168, num_heads: 64, head_dim: 112 },
  { id: "falcon-40b", label: "Falcon-40B", params: 40.0e9, num_layers: 60, hidden_dim: 8192, num_heads: 128, head_dim: 64, num_kv_heads: 8 },
  { id: "llama2-70b", label: "Llama-2 70B", params: 70.0e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128 },
  { id: "gpt3-175b", label: "GPT-3 175B", params: 175.0e9, num_layers: 96, hidden_dim: 12288, num_heads: 96, head_dim: 128 },
  { id: "falcon-180b", label: "Falcon-180B", params: 180.0e9, num_layers: 80, hidden_dim: 14848, num_heads: 232, head_dim: 64, num_kv_heads: 8 },

  // Dense, GQA (smaller num_kv_heads shrinks KV cache vs. plain MHA above)
  { id: "qwen2.5-7b", label: "Qwen2.5 7B", params: 7.6e9, num_layers: 28, hidden_dim: 3584, num_heads: 28, head_dim: 128, num_kv_heads: 4 },
  { id: "qwen3-8b", label: "Qwen3 8B", params: 8.19e9, num_layers: 36, hidden_dim: 4096, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.1-8b", label: "Llama-3.1 8B", params: 8.03e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma2-9b", label: "Gemma 2 9B", params: 9.24e9, num_layers: 42, hidden_dim: 3584, num_heads: 16, head_dim: 256, num_kv_heads: 8 },
  { id: "mistral-nemo-12b", label: "Mistral Nemo 12B", params: 12.2e9, num_layers: 40, hidden_dim: 5120, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma3-12b", label: "Gemma 3 12B", params: 12.0e9, num_layers: 48, hidden_dim: 3840, num_heads: 16, head_dim: 256, num_kv_heads: 8 },
  { id: "phi-3-medium", label: "Phi-3-medium 14B", params: 14.0e9, num_layers: 40, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 10 },
  { id: "qwen2.5-14b", label: "Qwen2.5 14B", params: 14.8e9, num_layers: 48, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen3-14b", label: "Qwen3 14B", params: 14.8e9, num_layers: 40, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },
  { id: "mistral-small-22b", label: "Mistral Small 22B", params: 22.2e9, num_layers: 56, hidden_dim: 6144, num_heads: 48, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma3-27b", label: "Gemma 3 27B", params: 27.0e9, num_layers: 62, hidden_dim: 5376, num_heads: 32, head_dim: 128, num_kv_heads: 16 },
  { id: "gemma2-27b", label: "Gemma 2 27B", params: 27.2e9, num_layers: 46, hidden_dim: 4608, num_heads: 32, head_dim: 128, num_kv_heads: 16 },
  { id: "qwen2.5-32b", label: "Qwen2.5 32B", params: 32.5e9, num_layers: 64, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen3-32b", label: "Qwen3 32B", params: 32.8e9, num_layers: 64, hidden_dim: 5120, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.1-70b", label: "Llama-3.1 70B", params: 70.6e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.3-70b", label: "Llama-3.3 70B", params: 70.6e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen2.5-72b", label: "Qwen2.5 72B", params: 72.7e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "nemotron-4-340b", label: "NVIDIA Nemotron-4 340B", params: 340.0e9, num_layers: 96, hidden_dim: 18432, num_heads: 96, head_dim: 192, num_kv_heads: 8 },
  { id: "mistral-large-2", label: "Mistral Large 2 123B", params: 123.0e9, num_layers: 88, hidden_dim: 12288, num_heads: 96, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.1-405b", label: "Llama-3.1 405B", params: 405.0e9, num_layers: 126, hidden_dim: 16384, num_heads: 128, head_dim: 128, num_kv_heads: 8 },

  // MoE — `active_params` is what's touched per token; `params` is the total resident-in-VRAM count.
  { id: "mixtral-8x7b", label: "Mixtral 8x7B (MoE)", params: 46.7e9, active_params: 12.9e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen3-30b-a3b", label: "Qwen3 30B-A3B (MoE)", params: 30.5e9, active_params: 3.3e9, num_layers: 48, hidden_dim: 2048, num_heads: 32, head_dim: 128, num_kv_heads: 4 },
  { id: "gpt-oss-20b", label: "gpt-oss-20B (MoE)", params: 21.0e9, active_params: 3.6e9, num_layers: 24, hidden_dim: 2880, num_heads: 64, head_dim: 64, num_kv_heads: 8 },
  { id: "mixtral-8x22b", label: "Mixtral 8x22B (MoE)", params: 141.0e9, active_params: 39.0e9, num_layers: 56, hidden_dim: 6144, num_heads: 48, head_dim: 128, num_kv_heads: 8 },
  { id: "dbrx", label: "DBRX 132B-A36B (MoE)", params: 132.0e9, active_params: 36.0e9, num_layers: 40, hidden_dim: 6144, num_heads: 48, head_dim: 128, num_kv_heads: 8 },
  { id: "gpt-oss-120b", label: "gpt-oss-120B (MoE)", params: 117.0e9, active_params: 5.1e9, num_layers: 36, hidden_dim: 2880, num_heads: 64, head_dim: 64, num_kv_heads: 8 },
  { id: "llama4-scout", label: "Llama 4 Scout 109B-A17B (MoE)", params: 109.0e9, active_params: 17.0e9, num_layers: 48, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },
  { id: "grok-1", label: "Grok-1 314B-A86B (MoE)", params: 314.0e9, active_params: 86.0e9, num_layers: 64, hidden_dim: 6144, num_heads: 48, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen3-235b-a22b", label: "Qwen3 235B-A22B (MoE)", params: 235.0e9, active_params: 22.0e9, num_layers: 94, hidden_dim: 4096, num_heads: 64, head_dim: 128, num_kv_heads: 4 },
  { id: "llama4-maverick", label: "Llama 4 Maverick 400B-A17B (MoE)", params: 400.0e9, active_params: 17.0e9, num_layers: 48, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },

  // MoE + MLA — `kv_latent_dim` replaces the head-count KV formula with DeepSeek-V3's compressed latent.
  { id: "deepseek-v3", label: "DeepSeek-V3 671B-A37B (MoE+MLA)", params: 671.0e9, active_params: 37.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
  { id: "deepseek-r1", label: "DeepSeek-R1 671B-A37B (MoE+MLA)", params: 671.0e9, active_params: 37.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
  { id: "kimi-k2", label: "Kimi K2 1T-A32B (MoE+MLA)", params: 1000.0e9, active_params: 32.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
  // Kimi K3's real published config (moonshotai/Kimi-K3) is a hybrid of 69
  // linear-attention "KDA" layers + 24 gated-MLA layers across 93 layers,
  // 896 experts (16 active) — a mix this engine's single-attention-type
  // model can't represent. Approximated as uniform MLA (same simplification
  // tier as DeepSeek/Kimi K2 above); head_dim/kv_latent_dim carried over
  // from Kimi K2 since K3's exact values for those two fields weren't
  // cleanly confirmed. params/active_params/layers/hidden_dim/heads are
  // from the real published config, so weights-VRAM math (the dominant
  // cost) is accurate — only the KV-cache portion is a simplification.
  { id: "kimi-k3", label: "Kimi K3 2.8T-A104B (MoE+MLA)", params: 2800.0e9, active_params: 104.0e9, num_layers: 93, hidden_dim: 7168, num_heads: 96, head_dim: 128, kv_latent_dim: 576 },

  // Closed-weight frontier models — the vendor has never disclosed architecture
  // (layer count, hidden dim, head count) for these, unlike every entry above,
  // which comes from a real published config/paper. These three are synthetic
  // placeholders: total size is loosely anchored to press/analyst *rumors*
  // (themselves unconfirmed), and the internal shape is invented to be a
  // plausible transformer at that scale — not sourced data. Much lower
  // confidence than the rest of this list; labeled "(est., unofficial)" so
  // that's visible in the picker itself, not just here.
  { id: "claude-3-haiku-est", label: "Claude 3 Haiku (est., unofficial)", params: 20.0e9, num_layers: 40, hidden_dim: 5120, num_heads: 40, head_dim: 128, num_kv_heads: 8 },
  { id: "claude-3.5-sonnet-est", label: "Claude 3.5 Sonnet (est., unofficial)", params: 70.0e9, num_layers: 76, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "grok-2-est", label: "Grok-2 (est., unofficial, MoE)", params: 270.0e9, active_params: 115.0e9, num_layers: 64, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  // ox-alpha: an anonymous "stealth" model benchmarked publicly on OpenRouter
  // (Aug 2026) — maker unconfirmed (fingerprinting analysis guesses Zhipu/
  // GLM-class with ~90% confidence, per third-party blog speculation, not a
  // vendor statement). Unlike the three estimates above, there isn't even a
  // rumored parameter count to anchor to — every spec field on its tracker
  // page reads "Unknown". This entry is a round-number placeholder so it
  // shows up in the list, not a size/architecture estimate of any kind.
  { id: "ox-alpha-placeholder", label: "ox-alpha (placeholder — no public specs exist)", params: 100.0e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128 },
];
