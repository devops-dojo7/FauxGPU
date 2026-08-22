"use client";

import { useEffect, useState } from "react";
import { calculateCost, calculateInference, calculateVram } from "@/lib/api";
import { CostResponse, GpuSpec, InferenceResponse, MODEL_PRESETS, ModelShape, VramResponse } from "@/lib/types";
import { ModelPanelState } from "./ModelPanel";
import { TopologyState } from "./TopologyPanel";
import { CostInputsState } from "./CostPanel";
import { InferenceInputsState } from "./InferencePanel";
import { Card } from "./ui";
import { CompareColumn } from "./CompareColumn";

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

/** Id of the slot with the best (min or max) value for a metric, ignoring nulls. Ties keep the first. */
function bestId<T extends { id: number }>(items: T[], value: (item: T) => number | null, dir: "min" | "max"): number | null {
  let best: { id: number; v: number } | null = null;
  for (const item of items) {
    const v = value(item);
    if (v === null || Number.isNaN(v)) continue;
    if (!best || (dir === "min" ? v < best.v : v > best.v)) best = { id: item.id, v };
  }
  return best?.id ?? null;
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

  const perGpuVram = (r: ColumnResult | undefined) => (r?.vram ? r.vram.total_gb / numGpus : null);
  const bestVramId = bestId(slots, (s) => perGpuVram(results[s.id]), "min");
  const bestCostId = bestId(slots, (s) => results[s.id]?.cost?.total_cost_usd ?? null, "min");
  const bestTpsId = bestId(slots, (s) => results[s.id]?.inference?.disaggregated_tokens_per_sec_per_gpu ?? null, "max");

  return (
    <Card title="Compare models">
      <p className="text-xs text-muted mb-4 max-w-2xl">
        Same GPU, topology, and workload assumptions as the Training/Inference tabs — only the model architecture and
        precision vary per column. Change GPU, topology, or cost/inference inputs on those tabs to see this view
        update.
      </p>

      <p className="text-xs text-muted mb-4">
        Comparing on <span className="text-body-strong">{gpu?.name ?? "…"}</span>
        {gpu && (
          <>
            {" "}
            · {numGpus} GPU{numGpus > 1 ? "s" : ""} ({topoState.shape.replace("_", " ")})
          </>
        )}
      </p>

      <div className="flex gap-4 overflow-x-auto pb-2">
        {slots.map((slot) => (
          <CompareColumn
            key={slot.id}
            slot={slot}
            result={results[slot.id]}
            gpu={gpu}
            numGpus={numGpus}
            canRemove={slots.length > 1}
            isBestVram={bestVramId === slot.id}
            isBestCost={bestCostId === slot.id}
            isBestThroughput={bestTpsId === slot.id}
            onChange={(patch) => updateSlot(slot.id, patch)}
            onRemove={() => removeSlot(slot.id)}
          />
        ))}

        {slots.length < MAX_SLOTS && (
          <button
            onClick={addSlot}
            className="w-72 shrink-0 rounded-xl border border-dashed border-hairline-strong text-muted hover:text-ink hover:border-ink transition-colors flex items-center justify-center min-h-[200px] text-sm font-medium"
          >
            + Add model
          </button>
        )}
      </div>
    </Card>
  );
}
