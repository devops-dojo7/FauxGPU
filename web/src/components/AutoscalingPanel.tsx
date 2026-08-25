"use client";

import { useState } from "react";
import { simulateAutoscaling } from "@/lib/api";
import { AutoscalingResponse, GpuSpec, MODEL_PRESETS } from "@/lib/types";
import { formatCompact } from "@/lib/format";
import { Card, Field, NumberInput, Select, Stat, Toggle } from "./ui";
import { LineChart } from "./LineChart";

let nextStageId = 1;

interface StageRow {
  id: number;
  duration_s: number;
  target_rps: number;
}

function makeStage(duration_s: number, target_rps: number): StageRow {
  return { id: nextStageId++, duration_s, target_rps };
}

// Mirrors scripts/load-test.js's api_concurrency scenario stages exactly.
const K6_SHAPED_STAGES: [number, number][] = [
  [20, 10],
  [30, 10],
  [20, 40],
  [30, 40],
  [20, 100],
  [30, 100],
  [15, 0],
];

export function AutoscalingPanel({ gpus }: { gpus: GpuSpec[] }) {
  const [presetId, setPresetId] = useState(MODEL_PRESETS.find((p) => p.id === "llama3.1-8b")?.id ?? MODEL_PRESETS[0].id);
  const [gpuId, setGpuId] = useState(gpus[0]?.id ?? "");
  const [precision, setPrecision] = useState("bf16");
  const [promptTokens, setPromptTokens] = useState(2048);
  const [outputTokens, setOutputTokens] = useState(256);
  const [decodeBatchSize, setDecodeBatchSize] = useState(8);
  const [pagedAttention, setPagedAttention] = useState(true);

  const [stages, setStages] = useState<StageRow[]>(() => K6_SHAPED_STAGES.map(([d, r]) => makeStage(d, r)));
  const [minReplicas, setMinReplicas] = useState(1);
  const [maxReplicas, setMaxReplicas] = useState(8);
  const [targetUtilizationPct, setTargetUtilizationPct] = useState(70);
  const [evalIntervalS, setEvalIntervalS] = useState(15);
  const [scaleDownStabilizationS, setScaleDownStabilizationS] = useState(60);

  const [result, setResult] = useState<AutoscalingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const preset = MODEL_PRESETS.find((p) => p.id === presetId) ?? MODEL_PRESETS[0];

  const updateStage = (id: number, patch: Partial<StageRow>) => setStages((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));
  const removeStage = (id: number) => setStages((prev) => prev.filter((s) => s.id !== id));
  const addStage = () => setStages((prev) => [...prev, makeStage(30, prev[prev.length - 1]?.target_rps ?? 10)]);

  const runSimulation = () => {
    setLoading(true);
    setError(null);
    simulateAutoscaling({
      model: preset,
      gpu_id: gpuId,
      precision,
      prompt_tokens: promptTokens,
      output_tokens: outputTokens,
      decode_batch_size: decodeBatchSize,
      cache_hit_fraction: 0,
      utilization: 0.35,
      paged_attention: pagedAttention,
      block_size: 16,
      stages: stages.map((s) => ({ duration_s: s.duration_s, target_rps: s.target_rps })),
      config: {
        min_replicas: minReplicas,
        max_replicas: maxReplicas,
        target_utilization_pct: targetUtilizationPct,
        eval_interval_s: evalIntervalS,
        scale_down_stabilization_s: scaleDownStabilizationS,
      },
      tick_s: 5,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="flex flex-col gap-6">
      <Card title="Workload">
        <p className="text-xs text-muted max-w-2xl mb-4">
          Simulates a Kubernetes HorizontalPodAutoscaler/KEDA-style control loop scaling inference-serving replicas
          against a traffic pattern — pre-seeded below with the exact stages from{" "}
          <code>scripts/load-test.js</code>&apos;s own k6 ramp.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="col-span-2">
            <Field label="Model">
              <Select value={presetId} onChange={setPresetId}>
                {MODEL_PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <div className="col-span-2">
            <Field label="GPU (per replica)">
              <Select value={gpuId} onChange={setGpuId}>
                {gpus.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.name}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Precision">
            <Select value={precision} onChange={setPrecision}>
              <option value="fp32">fp32</option>
              <option value="bf16">bf16</option>
              <option value="fp16">fp16</option>
              <option value="fp8">fp8</option>
            </Select>
          </Field>
          <Field label="Prompt tokens">
            <NumberInput value={promptTokens} min={1} onChange={setPromptTokens} />
          </Field>
          <Field label="Output tokens">
            <NumberInput value={outputTokens} min={1} onChange={setOutputTokens} />
          </Field>
          <Field label="Decode batch size">
            <NumberInput value={decodeBatchSize} min={1} onChange={setDecodeBatchSize} />
          </Field>
          <div className="flex items-end">
            <Toggle checked={pagedAttention} onChange={setPagedAttention} label="PagedAttention" />
          </div>
        </div>
      </Card>

      <Card title="Traffic pattern (k6-shaped stages)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                <th className="py-2 pr-3">Duration (s)</th>
                <th className="py-2 pr-3">Target RPS</th>
                <th className="py-2 pr-3" />
              </tr>
            </thead>
            <tbody>
              {stages.map((s) => (
                <tr key={s.id} className="border-b border-hairline-soft last:border-0">
                  <td className="py-2 pr-3 w-28">
                    <NumberInput value={s.duration_s} min={1} onChange={(v) => updateStage(s.id, { duration_s: v })} />
                  </td>
                  <td className="py-2 pr-3 w-28">
                    <NumberInput value={s.target_rps} min={0} onChange={(v) => updateStage(s.id, { target_rps: v })} />
                  </td>
                  <td className="py-2 pr-3">
                    <button onClick={() => removeStage(s.id)} disabled={stages.length <= 1} className="text-muted-soft hover:text-error disabled:opacity-30 px-1">
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <button onClick={addStage} className="mt-4 text-xs px-2.5 py-1 rounded-full border border-hairline-strong hover:bg-surface-strong transition-colors">
          + Add stage
        </button>
      </Card>

      <Card title="Autoscaler config">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          <Field label="Min replicas">
            <NumberInput value={minReplicas} min={1} onChange={setMinReplicas} />
          </Field>
          <Field label="Max replicas">
            <NumberInput value={maxReplicas} min={1} onChange={setMaxReplicas} />
          </Field>
          <Field label="Target utilization %">
            <NumberInput value={targetUtilizationPct} min={1} max={100} onChange={setTargetUtilizationPct} />
          </Field>
          <Field label="Eval interval (s) — HPA default: 15">
            <NumberInput value={evalIntervalS} min={1} onChange={setEvalIntervalS} />
          </Field>
          <Field label="Scale-down stabilization (s) — HPA default: 300">
            <NumberInput value={scaleDownStabilizationS} min={0} onChange={setScaleDownStabilizationS} />
          </Field>
        </div>
        <div className="flex justify-end mt-4">
          <button
            onClick={runSimulation}
            disabled={loading || !gpuId}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            {loading ? "Simulating…" : "Simulate"}
          </button>
        </div>
      </Card>

      {error && <p className="text-sm text-error">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <Stat label="Per-replica capacity" value={`${result.per_replica_capacity_rps.toFixed(2)} req/s`} />
            </Card>
            <Card>
              <Stat label="Peak replicas" value={String(result.peak_replicas)} />
            </Card>
            <Card>
              <Stat label="Peak backlog" value={formatCompact(result.peak_backlog_requests)} sub="requests" />
            </Card>
            <Card>
              <Stat label="Peak queue delay" value={`${result.peak_queue_delay_s.toFixed(1)}s`} />
            </Card>
          </div>

          <Card title="Demand vs. capacity">
            <LineChart
              series={[
                { label: "Demand (req/s)", color: "#f43f5e", points: result.points.map((p) => ({ x: p.t, y: p.demand_rps })) },
                { label: "Capacity (req/s)", color: "#10b981", points: result.points.map((p) => ({ x: p.t, y: p.capacity_rps })) },
              ]}
              xLabel="time"
              yLabel="req/s"
              formatX={(v) => `${v.toFixed(0)}s`}
            />
          </Card>

          <Card title="Replica count over time">
            <LineChart
              series={[{ label: "Replicas", color: "#3b82f6", points: result.points.map((p) => ({ x: p.t, y: p.replicas })) }]}
              xLabel="time"
              yLabel="replicas"
              formatX={(v) => `${v.toFixed(0)}s`}
            />
          </Card>

          <Card title="Backlog (queueing impact of under-provisioning)">
            <LineChart
              series={[{ label: "Backlog (requests)", color: "#f59e0b", points: result.points.map((p) => ({ x: p.t, y: p.backlog_requests })) }]}
              xLabel="time"
              yLabel="backlog"
              formatX={(v) => `${v.toFixed(0)}s`}
            />
          </Card>
        </>
      )}
    </div>
  );
}
