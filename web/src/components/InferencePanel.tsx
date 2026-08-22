"use client";

import { useEffect, useState } from "react";
import { calculateInference } from "@/lib/api";
import { GpuSpec, InferenceResponse, ModelShape } from "@/lib/types";
import { formatUsd } from "@/lib/format";
import { Card, Field, ModelArchBadges, NumberInput, Select, Stat, Toggle } from "./ui";

export interface InferenceInputsState {
  precision: string;
  promptTokens: number;
  outputTokens: number;
  decodeBatchSize: number;
  requestsPerSec: number;
  cacheHitPct: number;
  decodeGpus: number;
  pagedAttention: boolean;
  gpuMemoryUtilizationPct: number;
}

interface WellLitPath {
  id: string;
  label: string;
  description: string;
  config: Omit<InferenceInputsState, "precision">;
}

const WELL_LIT_PATHS: WellLitPath[] = [
  {
    id: "chat",
    label: "Chat — low latency",
    description: "Short prompts, small batches, tuned for fast TTFT over raw throughput.",
    config: { promptTokens: 512, outputTokens: 256, decodeBatchSize: 1, requestsPerSec: 1, cacheHitPct: 50, decodeGpus: 1, pagedAttention: true, gpuMemoryUtilizationPct: 85 },
  },
  {
    id: "rag",
    label: "RAG — long context",
    description: "Large retrieved-context prompts with a shared/cached system prefix.",
    config: { promptTokens: 8192, outputTokens: 512, decodeBatchSize: 4, requestsPerSec: 2, cacheHitPct: 70, decodeGpus: 2, pagedAttention: true, gpuMemoryUtilizationPct: 90 },
  },
  {
    id: "batch",
    label: "Batch — high throughput",
    description: "Large decode batches, many concurrent requests, offline/async workloads.",
    config: { promptTokens: 1024, outputTokens: 1024, decodeBatchSize: 32, requestsPerSec: 8, cacheHitPct: 10, decodeGpus: 4, pagedAttention: true, gpuMemoryUtilizationPct: 95 },
  },
  {
    id: "code",
    label: "Code completion — bursty",
    description: "Short outputs, very high request rate, heavy prefix-cache reuse from shared repo context.",
    config: { promptTokens: 2048, outputTokens: 64, decodeBatchSize: 4, requestsPerSec: 10, cacheHitPct: 60, decodeGpus: 2, pagedAttention: true, gpuMemoryUtilizationPct: 85 },
  },
];

function ThroughputBar({ label, tps, maxTps, color }: { label: string; tps: number; maxTps: number; color: string }) {
  const pct = maxTps > 0 ? Math.max(2, (tps / maxTps) * 100) : 0;
  return (
    <div>
      <div className="flex justify-between text-xs text-body mb-1">
        <span>{label}</span>
        <span className="tabular-nums">{tps.toFixed(0)} tok/s</span>
      </div>
      <div className="h-5 w-full rounded-md bg-surface-strong overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function InferencePanel({
  model,
  gpu,
  state,
  onChange,
}: {
  model: ModelShape;
  gpu: GpuSpec | undefined;
  state: InferenceInputsState;
  onChange: (s: InferenceInputsState) => void;
}) {
  const set = (patch: Partial<InferenceInputsState>) => onChange({ ...state, ...patch });

  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!gpu) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for an outbound fetch, not derived state
    setLoading(true);
    setError(null);
    calculateInference({
      model,
      gpu_id: gpu.id,
      precision: state.precision,
      prompt_tokens: state.promptTokens,
      output_tokens: state.outputTokens,
      decode_batch_size: state.decodeBatchSize,
      requests_per_sec: state.requestsPerSec,
      cache_hit_fraction: state.cacheHitPct / 100,
      utilization: 0.35,
      paged_attention: state.pagedAttention,
      block_size: 16,
      gpu_memory_utilization: state.gpuMemoryUtilizationPct / 100,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [model, gpu, state]);

  const maxTps = result ? result.disaggregated_tokens_per_sec_per_gpu : 0;
  const clusterTps = result ? result.disaggregated_tokens_per_sec_per_gpu * state.decodeGpus : 0;
  const costPer1kTokens =
    result && gpu && clusterTps > 0 ? ((gpu.price_per_hr_usd * state.decodeGpus) / 3600 / clusterTps) * 1000 : 0;

  return (
    <Card title="Inference & serving">
      <div className="mb-4 rounded-lg border border-hairline bg-surface-strong p-3 text-xs leading-relaxed text-body">
        Modeled on{" "}
        <a href="https://llm-d.ai/" target="_blank" rel="noreferrer" className="text-blue-500 underline underline-offset-2">
          llm-d
        </a>
        , a Kubernetes-native distributed inference stack: <strong>prefill</strong> (time-to-first-token) is
        compute-bound — one forward pass over the prompt — while <strong>decode</strong> (time-per-output-token) is
        memory-bandwidth-bound, streaming weights + KV cache from HBM one token at a time. Serving both phases on the
        same GPUs (&quot;colocated&quot;) lets prefill bursts stall decode throughput; llm-d&apos;s disaggregated
        pattern runs them on separate GPU pools so decode stays steady. Cache hit % models a prefix-cache routing hit.
        Toggle <strong>vLLM PagedAttention</strong> below to see KV cache allocated in fixed-size blocks (some
        rounding waste) instead of exact per-token — the engine llm-d itself builds on.
      </div>

      <div className="mb-4">
        <ModelArchBadges model={model} />
      </div>

      <div className="mb-4">
        <Field label="Well-lit path (preset workload)">
          <Select
            value=""
            onChange={(id) => {
              const path = WELL_LIT_PATHS.find((p) => p.id === id);
              if (path) set(path.config);
            }}
          >
            <option value="">Choose a preset…</option>
            {WELL_LIT_PATHS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </Select>
        </Field>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <Field label="Precision">
          <Select value={state.precision} onChange={(v) => set({ precision: v })}>
            <option value="fp32">fp32</option>
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
            <option value="fp8">fp8</option>
          </Select>
        </Field>
        <Field label="Prompt tokens">
          <NumberInput value={state.promptTokens} min={1} step={128} onChange={(v) => set({ promptTokens: v })} />
        </Field>
        <Field label="Output tokens">
          <NumberInput value={state.outputTokens} min={1} step={32} onChange={(v) => set({ outputTokens: v })} />
        </Field>
        <Field label="Decode batch size">
          <NumberInput value={state.decodeBatchSize} min={1} onChange={(v) => set({ decodeBatchSize: v })} />
        </Field>
        <Field label="Requests / sec">
          <NumberInput value={state.requestsPerSec} min={0} step={0.5} onChange={(v) => set({ requestsPerSec: v })} />
        </Field>
        <Field label="Prefix cache hit %">
          <NumberInput value={state.cacheHitPct} min={0} max={100} step={5} onChange={(v) => set({ cacheHitPct: v })} />
        </Field>
        <Field label="Decode pool GPUs">
          <NumberInput value={state.decodeGpus} min={1} onChange={(v) => set({ decodeGpus: v })} />
        </Field>
        <Field label="GPU memory utilization %">
          <NumberInput value={state.gpuMemoryUtilizationPct} min={10} max={100} step={5} onChange={(v) => set({ gpuMemoryUtilizationPct: v })} />
        </Field>
        <div className="flex items-end pb-1.5">
          <Toggle checked={state.pagedAttention} onChange={(v) => set({ pagedAttention: v })} label="vLLM PagedAttention" />
        </div>
      </div>

      {error && <p className="text-sm text-error">{error}</p>}
      {result && !error && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            <Stat label="TTFT" value={`${result.ttft_ms.toFixed(0)} ms`} sub="time to first token" />
            <Stat label="Decode step" value={`${result.decode_step_ms.toFixed(1)} ms`} sub={`batch ${state.decodeBatchSize}`} />
            <Stat label="Peak throughput" value={`${result.tokens_per_sec_per_gpu.toFixed(0)} tok/s`} sub="per GPU, undisturbed" />
            <Stat label="Prefill interference" value={`${(result.prefill_interference_fraction * 100).toFixed(0)}%`} sub="of GPU time, if colocated" />
          </div>

          <div className="space-y-3 mb-5">
            <ThroughputBar label="Colocated (prefill + decode share GPUs)" tps={result.colocated_tokens_per_sec_per_gpu} maxTps={maxTps} color="bg-amber-500" />
            <ThroughputBar label="Disaggregated (llm-d style — separate pools)" tps={result.disaggregated_tokens_per_sec_per_gpu} maxTps={maxTps} color="bg-emerald-500" />
          </div>

          <div className="border-t border-hairline pt-4 grid grid-cols-2 md:grid-cols-3 gap-4">
            <Stat label="Decode pool cost" value={`$${(gpu ? gpu.price_per_hr_usd * state.decodeGpus : 0).toFixed(2)}/hr`} sub={`${state.decodeGpus} GPU${state.decodeGpus > 1 ? "s" : ""}`} />
            <Stat label="Cluster throughput" value={`${clusterTps.toFixed(0)} tok/s`} sub="disaggregated decode pool" />
            <Stat label="Cost / 1K tokens" value={formatUsd(costPer1kTokens)} sub="served, decode pool only" />
          </div>

          <div className="border-t border-hairline pt-4 mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <Stat label="Usable VRAM" value={`${result.usable_vram_gb.toFixed(0)} GB`} sub={`${state.gpuMemoryUtilizationPct}% of ${gpu?.vram_gb ?? 0} GB`} />
            <Stat label="Weights" value={`${result.weights_gb.toFixed(0)} GB`} sub="all experts resident, if MoE" />
            <Stat label="KV cache budget" value={`${result.kv_budget_gb.toFixed(1)} GB`} sub="usable VRAM minus weights" />
            <Stat
              label="Max concurrent sequences"
              value={`${result.max_concurrent_sequences}`}
              sub={result.max_concurrent_sequences === 0 ? "weights alone exceed this GPU" : "per GPU, at this avg length"}
            />
          </div>
        </>
      )}
      {loading && <p className="text-sm text-muted mt-2">Calculating…</p>}
    </Card>
  );
}
