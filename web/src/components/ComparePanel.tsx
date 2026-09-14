"use client";

import { useEffect, useMemo, useState } from "react";
import { calculateCost, calculateInference, calculateVram } from "@/lib/api";
import { CostResponse, GpuSpec, InferenceResponse, MODEL_PRESET_GROUPS, MODEL_PRESETS, ModelShape, VramResponse } from "@/lib/types";
import { isMoe, usesMla } from "@/lib/simEngine";
import { formatGb, formatUsd } from "@/lib/format";
import { ModelPanelState } from "./ModelPanel";
import { TopologyState } from "./TopologyPanel";
import { CostInputsState } from "./CostPanel";
import { InferenceInputsState } from "./InferencePanel";
import { Card, Field, NumberInput, Select, Toggle } from "./ui";
import { RadarChart, RadarAxis, RadarSeries } from "./RadarChart";

export interface CompareSlot {
  id: number;
  presetId: string;
  model: ModelShape;
  customName: string;
  precision: string;
}

export interface ColumnResult {
  vram: VramResponse | null;
  cost: CostResponse | null;
  inference: InferenceResponse | null;
  loading: boolean;
  error: string | null;
}

const MAX_SLOTS = 4;
const SLOT_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e"];
let nextSlotId = 1;

function makeSlot(presetId: string): CompareSlot {
  const preset = MODEL_PRESETS.find((p) => p.id === presetId);
  return {
    id: nextSlotId++,
    presetId,
    model: preset ? { ...preset } : { params: 7e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 },
    customName: "",
    precision: "bf16",
  };
}

function slotLabel(slot: CompareSlot): string {
  if (slot.presetId === "custom") return slot.customName.trim() || "Custom";
  return MODEL_PRESETS.find((p) => p.id === slot.presetId)?.label ?? slot.presetId;
}

function archSummary(model: ModelShape): string {
  const parts = [isMoe(model) ? "MoE" : "Dense"];
  if (usesMla(model)) parts.push("MLA");
  else if (model.num_kv_heads != null && model.num_kv_heads < model.num_heads) parts.push("GQA");
  else parts.push("MHA");
  return parts.join(" · ");
}

interface Metric {
  key: string;
  label: string;
  lowerBetter: boolean;
  raw: (r: ColumnResult | undefined) => number | null;
  display: (r: ColumnResult | undefined) => string;
}

function buildMetrics(gpu: GpuSpec | undefined, numGpus: number): Metric[] {
  const capacityPerGpu = gpu?.vram_gb ?? 0;
  const perGpuVram = (r: ColumnResult | undefined) => (r?.vram ? r.vram.total_gb / numGpus : null);

  return [
    {
      key: "vram",
      label: "VRAM / GPU",
      lowerBetter: true,
      raw: perGpuVram,
      display: (r) => {
        const v = perGpuVram(r);
        if (v === null) return "—";
        return `${formatGb(v)}${capacityPerGpu > 0 ? ` / ${capacityPerGpu} GB` : ""}`;
      },
    },
    {
      key: "cost",
      label: "Training cost",
      lowerBetter: true,
      raw: (r) => r?.cost?.total_cost_usd ?? null,
      display: (r) => (r?.cost ? formatUsd(r.cost.total_cost_usd) : "—"),
    },
    {
      key: "costPer1k",
      label: "$ / 1K tokens (train)",
      lowerBetter: true,
      raw: (r) => r?.cost?.cost_per_1k_tokens_usd ?? null,
      display: (r) => (r?.cost ? formatUsd(r.cost.cost_per_1k_tokens_usd) : "—"),
    },
    {
      key: "wallClock",
      label: "Wall-clock time",
      lowerBetter: true,
      raw: (r) => r?.cost?.total_time_hours ?? null,
      display: (r) => (r?.cost ? `${r.cost.total_time_hours.toFixed(1)} hr` : "—"),
    },
    {
      key: "power",
      label: "Power / GPU",
      lowerBetter: true,
      raw: (r) => r?.cost?.power_watts_per_gpu ?? null,
      display: (r) => (r?.cost ? `${r.cost.power_watts_per_gpu.toFixed(0)} W` : "—"),
    },
    {
      key: "ttft",
      label: "TTFT",
      lowerBetter: true,
      raw: (r) => r?.inference?.ttft_ms ?? null,
      display: (r) => (r?.inference ? `${r.inference.ttft_ms.toFixed(0)} ms` : "—"),
    },
    {
      key: "throughput",
      label: "Tokens/sec per GPU",
      lowerBetter: false,
      raw: (r) => r?.inference?.disaggregated_tokens_per_sec_per_gpu ?? null,
      display: (r) => (r?.inference ? r.inference.disaggregated_tokens_per_sec_per_gpu.toFixed(0) : "—"),
    },
    {
      key: "concurrency",
      label: "Max concurrent sequences",
      lowerBetter: false,
      raw: (r) => r?.inference?.max_concurrent_sequences ?? null,
      display: (r) => (r?.inference ? `${r.inference.max_concurrent_sequences}` : "—"),
    },
  ];
}

const RADAR_KEYS = ["vram", "cost", "throughput", "ttft", "concurrency", "power"];
const RADAR_LABELS: Record<string, string> = {
  vram: "VRAM headroom",
  cost: "Cost efficiency",
  throughput: "Throughput",
  ttft: "Latency",
  concurrency: "Concurrency",
  power: "Power efficiency",
};

/** Min-max normalize raw values to a 20-100 range so the worst entry never collapses to the chart's center. */
function normalize(entries: { id: number; raw: number | null }[], lowerBetter: boolean): Record<number, number> {
  const valid = entries.filter((e): e is { id: number; raw: number } => e.raw !== null && !Number.isNaN(e.raw));
  if (valid.length === 0) return {};
  const vals = valid.map((v) => v.raw);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const out: Record<number, number> = {};
  for (const v of valid) {
    let t = min === max ? 1 : (v.raw - min) / (max - min);
    if (lowerBetter) t = 1 - t;
    out[v.id] = 20 + t * 80;
  }
  return out;
}

export function ComparePanel({
  gpu,
  gpuId,
  topoState,
  costInputs,
  inferenceInputs,
  workload,
  numGpus,
}: {
  gpu: GpuSpec | undefined;
  gpuId: string;
  topoState: TopologyState;
  costInputs: CostInputsState;
  inferenceInputs: InferenceInputsState;
  workload: ModelPanelState;
  numGpus: number;
}) {
  const [slots, setSlots] = useState<CompareSlot[]>(() => [makeSlot("llama2-7b"), makeSlot("qwen2.5-7b")]);
  const [results, setResults] = useState<Record<number, ColumnResult>>({});
  const [highlightBest, setHighlightBest] = useState(true);

  const addSlot = () => {
    if (slots.length >= MAX_SLOTS) return;
    const unused = MODEL_PRESETS.find((p) => !slots.some((s) => s.presetId === p.id));
    setSlots((prev) => [...prev, makeSlot(unused?.id ?? MODEL_PRESETS[0].id)]);
  };
  const removeSlot = (id: number) => setSlots((prev) => prev.filter((s) => s.id !== id));
  const updateSlot = (id: number, patch: Partial<CompareSlot>) =>
    setSlots((prev) => prev.map((s) => (s.id === id ? { ...s, ...patch } : s)));

  useEffect(() => {
    if (!gpu) return;
    let cancelled = false;
    const canCost = topoState.shape === "single_gpu" || !!gpu.nvlink_gbps;

    // eslint-disable-next-line react-hooks/set-state-in-effect -- flips a loading flag before the outbound fetches below, not derived state
    setResults((prev) => {
      const next = { ...prev };
      for (const slot of slots) {
        next[slot.id] = { ...(next[slot.id] ?? { vram: null, cost: null, inference: null, error: null }), loading: true };
      }
      return next;
    });

    for (const slot of slots) {
      Promise.all([
        calculateVram({
          model: slot.model,
          precision: slot.precision,
          batch_size: workload.batchSize,
          seq_len: workload.seqLen,
          optimizer: workload.optimizer,
          fp32_master_copy: workload.fp32MasterCopy,
          checkpointing: workload.checkpointing,
          training: workload.training,
        }),
        canCost
          ? calculateCost({
              model: slot.model,
              topology: {
                shape: topoState.shape,
                gpu_id: gpuId,
                gpus_per_node: topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode,
                num_nodes: topoState.shape === "multi_node" ? topoState.numNodes : 1,
                fabric_id: topoState.shape === "multi_node" ? topoState.fabricId : null,
              },
              tokens_per_step: costInputs.tokensPerStep,
              total_training_tokens: costInputs.totalTrainingTokensB * 1e9,
              precision: slot.precision,
              utilization: costInputs.utilization,
              tp_degree: costInputs.tpDegree,
              pp_degree: costInputs.ppDegree,
              batch_size: costInputs.batchSize,
              seq_len: costInputs.seqLen,
              num_microbatches: costInputs.numMicrobatches,
            })
          : Promise.resolve(null),
        calculateInference({
          model: slot.model,
          gpu_id: gpuId,
          precision: slot.precision,
          prompt_tokens: inferenceInputs.promptTokens,
          output_tokens: inferenceInputs.outputTokens,
          decode_batch_size: inferenceInputs.decodeBatchSize,
          requests_per_sec: inferenceInputs.requestsPerSec,
          cache_hit_fraction: inferenceInputs.cacheHitPct / 100,
          utilization: 0.35,
          paged_attention: inferenceInputs.pagedAttention,
          block_size: 16,
          gpu_memory_utilization: inferenceInputs.gpuMemoryUtilizationPct / 100,
        }),
      ])
        .then(([vram, cost, inference]) => {
          if (cancelled) return;
          setResults((prev) => ({ ...prev, [slot.id]: { vram, cost, inference, loading: false, error: null } }));
        })
        .catch((e) => {
          if (cancelled) return;
          setResults((prev) => ({
            ...prev,
            [slot.id]: { vram: null, cost: null, inference: null, loading: false, error: e instanceof Error ? e.message : String(e) },
          }));
        });
    }

    return () => {
      cancelled = true;
    };
  }, [
    slots,
    gpu,
    gpuId,
    topoState,
    costInputs,
    inferenceInputs,
    workload.batchSize,
    workload.seqLen,
    workload.optimizer,
    workload.fp32MasterCopy,
    workload.checkpointing,
    workload.training,
  ]);

  const metrics = useMemo(() => buildMetrics(gpu, numGpus), [gpu, numGpus]);
  const metricByKey = useMemo(() => Object.fromEntries(metrics.map((m) => [m.key, m])), [metrics]);

  const bestBySlotKey = useMemo(() => {
    const map: Record<string, number | null> = {};
    for (const m of metrics) {
      const entries = slots.map((s) => ({ id: s.id, raw: m.raw(results[s.id]) }));
      const valid = entries.filter((e) => e.raw !== null) as { id: number; raw: number }[];
      if (valid.length < 2) {
        map[m.key] = null;
        continue;
      }
      let best = valid[0];
      for (const e of valid) if (m.lowerBetter ? e.raw < best.raw : e.raw > best.raw) best = e;
      const isTie = valid.every((e) => e.raw === best.raw);
      map[m.key] = isTie ? null : best.id;
    }
    return map;
  }, [metrics, slots, results]);

  const radarAxes: RadarAxis[] = RADAR_KEYS.map((key) => ({ key, label: RADAR_LABELS[key] }));
  const radarSeries: RadarSeries[] = useMemo(() => {
    const normalizedPerKey: Record<string, Record<number, number>> = {};
    for (const key of RADAR_KEYS) {
      const m = metricByKey[key];
      normalizedPerKey[key] = normalize(
        slots.map((s) => ({ id: s.id, raw: m.raw(results[s.id]) })),
        m.lowerBetter,
      );
    }
    return slots.map((s, i) => ({
      id: s.id,
      label: slotLabel(s),
      color: SLOT_COLORS[i % SLOT_COLORS.length],
      values: Object.fromEntries(RADAR_KEYS.map((key) => [key, normalizedPerKey[key][s.id] ?? 0])),
    }));
  }, [slots, results, metricByKey]);

  const gridTemplate = `200px repeat(${slots.length}, minmax(160px, 1fr))`;
  const hasCustom = slots.some((s) => s.presetId === "custom");

  return (
    <Card title="Compare models" className="overflow-hidden">
      <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
        <p className="text-xs text-muted max-w-xl">
          Same GPU, topology, and workload assumptions as the Training/Inference tabs — only the model architecture
          and precision vary per column. Change GPU above, or topology/cost/inference inputs on those tabs, to see
          this view update.
        </p>
        <Toggle checked={highlightBest} onChange={setHighlightBest} label="Highlight best" />
      </div>

      <p className="text-xs text-muted mb-4">
        Comparing on <span className="text-body-strong">{gpu?.name ?? "…"}</span>
        {gpu && (
          <>
            {" "}
            · {numGpus} GPU{numGpus > 1 ? "s" : ""} ({topoState.shape.replace("_", " ")})
          </>
        )}
      </p>

      <div className="overflow-x-auto">
        <div style={{ minWidth: 200 + slots.length * 180 }}>
          {/* Header: color dot + preset select + remove, per column */}
          <div className="grid gap-3 items-center pb-3" style={{ gridTemplateColumns: gridTemplate }}>
            <div />
            {slots.map((slot, i) => (
              <div key={slot.id} className="flex items-center gap-2 min-w-0">
                <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: SLOT_COLORS[i % SLOT_COLORS.length] }} />
                <div className="flex-1 min-w-0">
                  <Select
                    value={slot.presetId}
                    onChange={(id) => {
                      if (id === "custom") {
                        updateSlot(slot.id, { presetId: "custom" });
                        return;
                      }
                      const preset = MODEL_PRESETS.find((p) => p.id === id);
                      if (preset) updateSlot(slot.id, { presetId: id, model: { ...preset } });
                    }}
                  >
                    {MODEL_PRESET_GROUPS.map((g) => (
                      <optgroup key={g.tier} label={g.label}>
                        {g.presets.map((p) => (
                          <option key={p.id} value={p.id}>
                            {p.label}
                          </option>
                        ))}
                      </optgroup>
                    ))}
                    <option value="custom">Custom</option>
                  </Select>
                </div>
                <button
                  onClick={() => removeSlot(slot.id)}
                  disabled={slots.length <= 1}
                  className="text-muted-soft hover:text-error disabled:opacity-30 px-1 shrink-0"
                  title="Remove from comparison"
                >
                  ✕
                </button>
              </div>
            ))}
            {slots.length < MAX_SLOTS && (
              <button
                onClick={addSlot}
                className="justify-self-start text-xs px-2.5 py-1 rounded-full border border-hairline-strong hover:bg-surface-strong transition-colors whitespace-nowrap"
              >
                + Add model
              </button>
            )}
          </div>

          {hasCustom && (
            <div className="grid gap-3 pb-4 border-b border-hairline" style={{ gridTemplateColumns: gridTemplate }}>
              <div className="text-xs text-muted self-start pt-2">Custom model shape</div>
              {slots.map((slot) => (
                <div key={slot.id}>
                  {slot.presetId === "custom" && (
                    <div className="flex flex-col gap-2">
                      <input
                        type="text"
                        className="w-full rounded-md border border-hairline-strong bg-surface-card px-2.5 py-1.5 text-sm text-ink outline-none focus:border-ink"
                        value={slot.customName}
                        onChange={(e) => updateSlot(slot.id, { customName: e.target.value })}
                        placeholder="Model name"
                      />
                      <div className="grid grid-cols-2 gap-2">
                        <Field label="Params (B)">
                          <NumberInput value={slot.model.params / 1e9} min={0.01} step={0.1} onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, params: v * 1e9 } })} />
                        </Field>
                        <Field label="Layers">
                          <NumberInput value={slot.model.num_layers} min={1} onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, num_layers: v } })} />
                        </Field>
                        <Field label="Hidden dim">
                          <NumberInput value={slot.model.hidden_dim} min={1} onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, hidden_dim: v } })} />
                        </Field>
                        <Field label="Heads">
                          <NumberInput value={slot.model.num_heads} min={1} onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, num_heads: v } })} />
                        </Field>
                        <Field label="Head dim">
                          <NumberInput value={slot.model.head_dim} min={1} onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, head_dim: v } })} />
                        </Field>
                        <Field label="KV heads">
                          <NumberInput
                            value={slot.model.num_kv_heads ?? slot.model.num_heads}
                            min={1}
                            onChange={(v) => updateSlot(slot.id, { model: { ...slot.model, num_kv_heads: v } })}
                          />
                        </Field>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* Radar chart */}
          <div className="flex flex-wrap items-center gap-8 justify-center py-6 border-b border-hairline">
            <div className="flex flex-col gap-1.5">
              {slots.map((s, i) => (
                <div key={s.id} className="flex items-center gap-2 text-xs">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ background: SLOT_COLORS[i % SLOT_COLORS.length] }} />
                  <span className="text-body-strong">{slotLabel(s)}</span>
                  <span className="text-muted">{archSummary(s.model)}</span>
                </div>
              ))}
            </div>
            <RadarChart axes={radarAxes} series={radarSeries} />
          </div>

          {/* Overview row (non-highlighted, descriptive) */}
          <div className="pt-4">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-2">Overview</p>
            {[
              { label: "Architecture", value: (s: CompareSlot) => archSummary(s.model) },
              { label: "Params", value: (s: CompareSlot) => `${(s.model.params / 1e9).toFixed(2)}B` },
              { label: "Precision", value: (s: CompareSlot) => s.precision },
            ].map((row) => (
              <div key={row.label} className="grid gap-3 py-2 border-b border-hairline-soft last:border-0" style={{ gridTemplateColumns: gridTemplate }}>
                <span className="text-sm text-body">{row.label}</span>
                {slots.map((s) => (
                  <span key={s.id} className="text-sm text-body-strong">
                    {row.value(s)}
                  </span>
                ))}
              </div>
            ))}
          </div>

          {(["VRAM", "Training cost", "Inference"] as const).map((section) => {
            const keys =
              section === "VRAM" ? ["vram"] : section === "Training cost" ? ["cost", "costPer1k", "wallClock", "power"] : ["ttft", "throughput", "concurrency"];
            return (
              <div key={section} className="pt-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-2">{section}</p>
                {keys.map((key) => {
                  const m = metricByKey[key];
                  const bestId = highlightBest ? bestBySlotKey[key] : null;
                  return (
                    <div key={key} className="grid gap-3 py-2 border-b border-hairline-soft last:border-0" style={{ gridTemplateColumns: gridTemplate }}>
                      <span className="text-sm text-body">{m.label}</span>
                      {slots.map((s) => {
                        const r = results[s.id];
                        const isBest = bestId === s.id;
                        return (
                          <span key={s.id} className={`text-sm tabular-nums ${isBest ? "text-success font-semibold" : "text-body-strong"}`}>
                            {m.display(r)}
                          </span>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            );
          })}

          {slots.some((s) => results[s.id]?.error) && (
            <div className="pt-4 grid gap-3" style={{ gridTemplateColumns: gridTemplate }}>
              <div />
              {slots.map((s) => (
                <span key={s.id} className="text-xs text-error">
                  {results[s.id]?.error ?? ""}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
