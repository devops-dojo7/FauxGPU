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
}

export interface RunSummary {
  run_id: string;
  status: "running" | "done";
  meta: RunMeta | null;
  latest_step: RunStep | null;
  started_at: number;
  updated_at: number;
}

export interface RunDetail extends RunSummary {
  steps: RunStep[];
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

export interface SimulateRunRequest {
  model: ModelShape;
  model_label: string;
  topology: TopologyRequest;
  precision: string;
  tokens_per_step: number;
  total_steps: number;
  speedup: number;
  utilization: number;
}

export interface ModelPreset extends ModelShape {
  id: string;
  label: string;
}

// Specs below are approximate, from public model cards/papers — illustrative
// for comparing relative scale and architecture (dense vs. GQA vs. MoE vs.
// MLA), not guaranteed to reproduce exact vendor numbers.
export const MODEL_PRESETS: ModelPreset[] = [
  { id: "qwen2.5-0.5b", label: "Qwen2.5 0.5B (draft-sized)", params: 0.49e9, num_layers: 24, hidden_dim: 896, num_heads: 14, head_dim: 64, num_kv_heads: 2 },
  { id: "llama3.2-1b", label: "Llama-3.2 1B (draft-sized)", params: 1.24e9, num_layers: 16, hidden_dim: 2048, num_heads: 32, head_dim: 64, num_kv_heads: 8 },
  { id: "llama2-7b", label: "Llama-2 7B", params: 6.74e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
  { id: "llama2-13b", label: "Llama-2 13B", params: 13.0e9, num_layers: 40, hidden_dim: 5120, num_heads: 40, head_dim: 128 },
  { id: "llama2-70b", label: "Llama-2 70B", params: 70.0e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128 },
  { id: "mistral-7b", label: "Mistral 7B", params: 7.3e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
  { id: "gpt3-175b", label: "GPT-3 175B", params: 175.0e9, num_layers: 96, hidden_dim: 12288, num_heads: 96, head_dim: 128 },
  // Dense, GQA (smaller num_kv_heads shrinks KV cache vs. plain MHA above)
  { id: "qwen2.5-7b", label: "Qwen2.5 7B", params: 7.6e9, num_layers: 28, hidden_dim: 3584, num_heads: 28, head_dim: 128, num_kv_heads: 4 },
  { id: "qwen2.5-72b", label: "Qwen2.5 72B", params: 72.7e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "nemotron-4-340b", label: "NVIDIA Nemotron-4 340B", params: 340.0e9, num_layers: 96, hidden_dim: 18432, num_heads: 96, head_dim: 192, num_kv_heads: 8 },
  { id: "llama3.1-8b", label: "Llama-3.1 8B", params: 8.03e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.1-70b", label: "Llama-3.1 70B", params: 70.6e9, num_layers: 80, hidden_dim: 8192, num_heads: 64, head_dim: 128, num_kv_heads: 8 },
  { id: "llama3.1-405b", label: "Llama-3.1 405B", params: 405.0e9, num_layers: 126, hidden_dim: 16384, num_heads: 128, head_dim: 128, num_kv_heads: 8 },
  { id: "gemma2-27b", label: "Gemma 2 27B", params: 27.2e9, num_layers: 46, hidden_dim: 4608, num_heads: 32, head_dim: 128, num_kv_heads: 16 },
  // MoE — `active_params` is what's touched per token; `params` is the total resident-in-VRAM count.
  { id: "mixtral-8x7b", label: "Mixtral 8x7B (MoE)", params: 46.7e9, active_params: 12.9e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128, num_kv_heads: 8 },
  { id: "qwen3-235b-a22b", label: "Qwen3 235B-A22B (MoE)", params: 235.0e9, active_params: 22.0e9, num_layers: 94, hidden_dim: 4096, num_heads: 64, head_dim: 128, num_kv_heads: 4 },
  // MoE + MLA — `kv_latent_dim` replaces the head-count KV formula with DeepSeek-V3's compressed latent.
  { id: "deepseek-v3", label: "DeepSeek-V3 671B-A37B (MoE+MLA)", params: 671.0e9, active_params: 37.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
  { id: "deepseek-r1", label: "DeepSeek-R1 671B-A37B (MoE+MLA)", params: 671.0e9, active_params: 37.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
  { id: "kimi-k2", label: "Kimi K2 1T-A32B (MoE+MLA)", params: 1000.0e9, active_params: 32.0e9, num_layers: 61, hidden_dim: 7168, num_heads: 128, head_dim: 128, kv_latent_dim: 576 },
];
