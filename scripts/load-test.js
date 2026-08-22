// k6 load test against a live simgpu API (default http://localhost:8000).
// Run: k6 run scripts/load-test.js
// Or via docker compose (no local k6 install needed):
//   docker compose --profile loadtest run k6
//
// This deliberately tests two DIFFERENT kinds of "limit", and the two
// scenarios below are honest about which is which:
//
//   gpuLimits       — sweeps decode_batch_size through POST /calculate/inference,
//                     the same analytic prefill/decode math the website's
//                     Inference tab calls. This finds the simulated GPU's
//                     actual limit as this project models it: KV cache growth
//                     vs. VRAM capacity (max_concurrent_sequences hits 0) and
//                     decode_step_ms degrading as batch size grows. It's cheap
//                     synchronous math, so 1 VU sweeping batch sizes is the
//                     point, not concurrency.
//
//   apiConcurrency  — ramps real concurrent virtual users against
//                     POST /inference/stream, the SSE endpoint the live
//                     playground uses, where each request paces itself with
//                     the model's real prefill/decode timing via
//                     asyncio.sleep. This finds the API SERVER's concurrency
//                     ceiling (uvicorn/event-loop capacity) — NOT the GPU's,
//                     since the simulator doesn't model shared-GPU contention
//                     across concurrent requests (each request's simulated
//                     timing is independent of how many others are in
//                     flight). Both are real, useful limits — they're just
//                     not the same limit.
//
// Env vars: BASE_URL, GPU_ID, MODEL_PARAMS_B, MODEL_LAYERS, MODEL_HIDDEN,
// MODEL_HEADS, MODEL_HEAD_DIM, MODEL_KV_HEADS (all optional, default to a
// Llama-3.1-8B-shaped dense model on an H100).

import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Rate } from "k6/metrics";

const BASE_URL = __ENV.BASE_URL || "http://localhost:8000";
const GPU_ID = __ENV.GPU_ID || "h100-sxm";

const MODEL = {
  params: Number(__ENV.MODEL_PARAMS_B || 8.03) * 1e9,
  num_layers: Number(__ENV.MODEL_LAYERS || 32),
  hidden_dim: Number(__ENV.MODEL_HIDDEN || 4096),
  num_heads: Number(__ENV.MODEL_HEADS || 32),
  head_dim: Number(__ENV.MODEL_HEAD_DIM || 128),
  num_kv_heads: Number(__ENV.MODEL_KV_HEADS || 8),
};

const decodeStepMs = new Trend("gpu_decode_step_ms", true);
const maxConcurrentSeqs = new Trend("gpu_max_concurrent_sequences");
const batchFitsGpu = new Rate("gpu_batch_fits_rate");

const streamDurationMs = new Trend("api_stream_duration_ms", true);
const streamErrorRate = new Rate("api_stream_error_rate");

export const options = {
  scenarios: {
    gpu_limits: {
      executor: "shared-iterations",
      exec: "gpuLimits",
      vus: 1,
      iterations: 5,
      maxDuration: "2m",
    },
    api_concurrency: {
      executor: "ramping-vus",
      exec: "apiConcurrency",
      startVUs: 0,
      startTime: "1m",
      stages: [
        { duration: "20s", target: 10 },
        { duration: "30s", target: 10 },
        { duration: "20s", target: 40 },
        { duration: "30s", target: 40 },
        { duration: "20s", target: 100 },
        { duration: "30s", target: 100 },
        { duration: "15s", target: 0 },
      ],
    },
  },
  thresholds: {
    api_stream_error_rate: ["rate<0.05"],
  },
};

const BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

export function gpuLimits() {
  for (const batchSize of BATCH_SIZES) {
    const res = http.post(
      `${BASE_URL}/calculate/inference`,
      JSON.stringify({
        model: MODEL,
        gpu_id: GPU_ID,
        precision: "bf16",
        prompt_tokens: 2048,
        output_tokens: 256,
        decode_batch_size: batchSize,
        requests_per_sec: 1,
        cache_hit_fraction: 0,
        utilization: 0.35,
        paged_attention: true,
        block_size: 16,
        gpu_memory_utilization: 0.9,
      }),
      { headers: { "Content-Type": "application/json" }, tags: { batch_size: String(batchSize) } },
    );

    const ok = check(res, { "gpu_limits: status 200": (r) => r.status === 200 });
    if (ok) {
      const body = res.json();
      const tags = { batch_size: String(batchSize) };
      decodeStepMs.add(body.decode_step_ms, tags);
      maxConcurrentSeqs.add(body.max_concurrent_sequences, tags);
      batchFitsGpu.add(body.max_concurrent_sequences > 0, tags);
    }
    sleep(0.2);
  }
}

export function apiConcurrency() {
  const start = Date.now();
  const res = http.post(
    `${BASE_URL}/inference/stream`,
    JSON.stringify({
      model: MODEL,
      model_label: "k6-load-test",
      gpu_id: GPU_ID,
      precision: "bf16",
      prompt: "Summarize the tradeoffs of tensor parallelism vs. pipeline parallelism.",
      prompt_tokens: 256,
      max_output_tokens: 64,
      cache_hit_fraction: 0,
      utilization: 0.35,
    }),
    { headers: { "Content-Type": "application/json" }, timeout: "60s" },
  );
  streamDurationMs.add(Date.now() - start);
  const ok = check(res, { "api_concurrency: status 200": (r) => r.status === 200 });
  streamErrorRate.add(!ok);
}
