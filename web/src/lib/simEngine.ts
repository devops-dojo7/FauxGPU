/**
 * Client-side mirror of engine/memory.py + engine/inference.py's prefill/decode
 * formulas — including GQA (num_kv_heads), MoE (active_params), MLA
 * (kv_latent_dim), and PagedAttention block rounding. Duplicated (not
 * fetched) so the live prompt playground can advance token-by-token smoothly
 * in the browser without a network round trip per token.
 * Keep in sync with the Python engine if those formulas change.
 */
import { GpuSpec, ModelShape } from "./types";

const BYTES_PER_ELEMENT: Record<string, number> = {
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
};

export function bytesPerParam(precision: string): number {
  return BYTES_PER_ELEMENT[precision] ?? 2;
}

/** All params resident in VRAM (every MoE expert included) — for VRAM sizing. */
export function weightsBytes(model: ModelShape, precision: string): number {
  return model.params * bytesPerParam(precision);
}

/** Params actually touched per token — active experts for MoE, all params for dense. */
export function activeWeightsBytes(model: ModelShape, precision: string): number {
  const active = model.active_params ?? model.params;
  return active * bytesPerParam(precision);
}

export function isMoe(model: ModelShape): boolean {
  return model.active_params != null && model.active_params < model.params;
}

export function usesMla(model: ModelShape): boolean {
  return model.kv_latent_dim != null;
}

export function effectiveKvHeads(model: ModelShape): number {
  return model.num_kv_heads ?? model.num_heads;
}

/** Bytes of KV cache added per token, per sequence (all layers). MLA models
 * (kv_latent_dim set) use a compressed shared latent instead of per-head K/V. */
export function kvCacheBytesPerToken(model: ModelShape, precision: string): number {
  const elemBytes = bytesPerParam(precision);
  if (usesMla(model)) {
    return model.num_layers * (model.kv_latent_dim as number) * elemBytes;
  }
  return 2 * model.num_layers * effectiveKvHeads(model) * model.head_dim * elemBytes;
}

export function achievableTflops(peakTflops: number, utilization = 0.35): number {
  return peakTflops * utilization;
}

export function prefillMs(
  model: ModelShape,
  gpu: GpuSpec,
  precision: string,
  promptTokens: number,
  utilization = 0.35,
  cacheHitFraction = 0,
): number {
  const effectiveTokens = promptTokens * (1 - cacheHitFraction);
  const activeParams = model.active_params ?? model.params;
  const flops = 2 * activeParams * effectiveTokens;
  const tflops = achievableTflops(gpu.bf16_tflops, utilization);
  return (flops / (tflops * 1e12)) * 1000;
}

function pagedKvTokens(kvTokens: number, pagedAttention: boolean, blockSize: number): number {
  if (!pagedAttention || blockSize <= 0) return kvTokens;
  return Math.ceil(kvTokens / blockSize) * blockSize;
}

/** Time for one decode step (one new token for every sequence in the batch),
 * given the average KV length across the batch. Mirrors
 * engine/inference.py:decode_step_seconds. */
export function decodeStepMs(
  model: ModelShape,
  gpu: GpuSpec,
  precision: string,
  avgKvTokens: number,
  batchSize = 1,
  pagedAttention = false,
  blockSize = 16,
): number {
  const weights = activeWeightsBytes(model, precision);
  const kvTokens = pagedKvTokens(avgKvTokens, pagedAttention, blockSize);
  const kvBytes = kvCacheBytesPerToken(model, precision) * kvTokens * batchSize;
  const totalBytes = weights + kvBytes;
  const bandwidthBytesPerSec = gpu.mem_bandwidth_gbps * 1e9;
  return (totalBytes / bandwidthBytesPerSec) * 1000;
}

/** Rough token-count estimate from raw text (~4 chars/token), for a
 * playground where we don't have a real tokenizer. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.trim().length / 4));
}

export interface VramFitCheck {
  fits: boolean;
  requiredGb: number;
  capacityGb: number;
}

/** Whether weights (all experts, if MoE) + the given total KV cache tokens
 * (summed across however many sequences are meant to be resident at once)
 * fit in the GPU's VRAM. */
/** tpDegree > 1 shards weights evenly across the group (mirrors
 * engine.estimate_serving_capacity) — this is a per-GPU fit check, same as
 * the backend's, so a model that doesn't fit at tpDegree=1 can correctly
 * pass once sharded. */
export function checkVramFit(model: ModelShape, gpu: GpuSpec, precision: string, totalKvTokens: number, tpDegree = 1): VramFitCheck {
  const requiredBytes = weightsBytes(model, precision) / Math.max(tpDegree, 1) + kvCacheBytesPerToken(model, precision) * totalKvTokens;
  const requiredGb = requiredBytes / 1e9;
  return { fits: requiredGb <= gpu.vram_gb, requiredGb, capacityGb: gpu.vram_gb };
}

/** Mirrors engine/power.py. busyFraction in [0,1]: 0 = idle, 1 = full TDP. */
export function powerWatts(gpu: GpuSpec, busyFraction: number): number {
  const f = Math.max(0, Math.min(1, busyFraction));
  return gpu.idle_watts + f * (gpu.tdp_watts - gpu.idle_watts);
}

export const DECODE_POWER_FRACTION = 0.55;

export function decodePowerWatts(gpu: GpuSpec): number {
  return powerWatts(gpu, DECODE_POWER_FRACTION);
}

export function prefillPowerWatts(gpu: GpuSpec, utilization = 0.35): number {
  return powerWatts(gpu, utilization);
}

const COMMUNICATION_POWER_FRACTION = 0.25;

/** Time-weighted average power across a training step's compute + comm phases. */
export function trainingStepPowerWatts(gpu: GpuSpec, computeS: number, communicationS: number, utilization = 0.35): number {
  const totalS = computeS + communicationS;
  if (totalS <= 0) return gpu.idle_watts;
  const computePower = powerWatts(gpu, utilization);
  const commPower = powerWatts(gpu, COMMUNICATION_POWER_FRACTION);
  return (computeS * computePower + communicationS * commPower) / totalS;
}
