"use client";

import { GpuSpec, VramResponse } from "@/lib/types";
import { formatGb } from "@/lib/format";
import { Card } from "./ui";

const SEGMENTS: { key: keyof VramResponse; label: string; color: string }[] = [
  { key: "weights_gb", label: "Weights", color: "bg-blue-500" },
  { key: "gradients_gb", label: "Gradients", color: "bg-purple-500" },
  { key: "optimizer_states_gb", label: "Optimizer states", color: "bg-amber-500" },
  { key: "activations_gb", label: "Activations", color: "bg-teal-500" },
  { key: "kv_cache_gb", label: "KV cache", color: "bg-rose-500" },
];

export function VramPanel({
  vram,
  gpu,
  numGpus,
  loading,
  error,
  zeroStage,
  dpSize,
  peftMethod,
}: {
  vram: VramResponse | null;
  gpu: GpuSpec | undefined;
  numGpus: number;
  loading: boolean;
  error: string | null;
  zeroStage: number;
  dpSize: number;
  peftMethod: string;
}) {
  const capacityPerGpu = gpu?.vram_gb ?? 0;
  const totalCapacity = capacityPerGpu * numGpus;
  const total = vram?.total_gb ?? 0;
  const peftLabel = peftMethod === "qlora" ? "QLoRA" : peftMethod === "lora" ? "LoRA" : null;
  // When ZeRO/FSDP sharding is off, this is a naive even split across GPUs
  // (real non-ZeRO sharding strategies differ; this is for intuition only).
  // When it's on, the backend has already sharded weights/gradients/
  // optimizer-state across dpSize ranks (see engine.memory.
  // compute_vram_breakdown) — `total` is already one rank's per-GPU
  // figure, so it is used directly rather than divided again by numGpus.
  // LoRA/QLoRA are single-replica figures for the same reason: there's
  // nothing DP-specific being sharded, so dividing by numGpus again would
  // double-count whatever real parallelism the deployment actually uses.
  const perGpu = zeroStage > 0 || peftLabel ? total : numGpus > 0 ? total / numGpus : total;
  const overflow = capacityPerGpu > 0 && perGpu > capacityPerGpu;

  return (
    <Card title="VRAM breakdown">
      {error && <p className="text-sm text-error">{error}</p>}
      {!error && vram && (
        <>
          <div className="h-8 w-full flex rounded-md overflow-hidden border border-hairline">
            {SEGMENTS.map((s) => {
              const v = vram[s.key] as number;
              const pct = total > 0 ? (v / total) * 100 : 0;
              if (pct <= 0) return null;
              return <div key={s.key} className={s.color} style={{ width: `${pct}%` }} title={`${s.label}: ${formatGb(v)}`} />;
            })}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
            {SEGMENTS.map((s) => {
              const v = vram[s.key] as number;
              if (v <= 0) return null;
              return (
                <div key={s.key} className="flex items-center gap-2">
                  <span className={`h-2.5 w-2.5 rounded-sm ${s.color}`} />
                  <span className="text-body">{s.label}</span>
                  <span className="ml-auto tabular-nums">{formatGb(v)}</span>
                </div>
              );
            })}
          </div>

          <div className="mt-4 border-t border-hairline pt-3 flex items-baseline justify-between">
            <span className="text-sm text-body">
              {peftLabel
                ? `Per GPU (${peftLabel} — single replica)`
                : zeroStage > 0
                  ? `Per GPU (ZeRO stage ${zeroStage}, ${dpSize}-way sharded)`
                  : `Total (per GPU, ${numGpus}-way split)`}
            </span>
            <span className={`text-lg font-semibold tabular-nums ${overflow ? "text-error" : ""}`}>
              {formatGb(perGpu)} {capacityPerGpu > 0 && <span className="text-sm font-normal">/ {capacityPerGpu} GB</span>}
            </span>
          </div>
          {peftLabel ? (
            <p className="mt-1 text-xs text-muted-soft">
              {peftLabel === "QLoRA"
                ? "QLoRA (Dettmers et al. 2023) freezes the base model in 4-bit NormalFloat and trains only a tiny low-rank adapter — this is what lets a 65B model fit on a single 48GB GPU in the paper's own headline result."
                : "LoRA (Hu et al. 2021) freezes the base model and trains only a tiny low-rank adapter — no optimizer state is kept for the (vast majority) frozen parameters, unlike full fine-tuning."}
            </p>
          ) : zeroStage > 0 ? (
            <p className="mt-1 text-xs text-muted-soft">
              ZeRO-DP/FSDP shards weights/gradients/optimizer-state across {dpSize} data-parallel ranks (Rajbhandari
              et al. 2020) — activations and KV cache are unaffected. Independent of, and stackable with, the
              Tensor/Pipeline parallel degree set below.
            </p>
          ) : (
            <p className="mt-1 text-xs text-muted-soft">
              Assumes an ideal, perfectly even split across your {numGpus} selected GPU{numGpus > 1 ? "s" : ""} — it
              isn&apos;t tied to the Tensor/Pipeline parallel degree set below, which is what actually determines how the
              model is sharded in a real run. Turn on ZeRO/FSDP sharding above for a stage-aware per-GPU figure.
            </p>
          )}
          {overflow && (
            <p className="mt-2 text-xs text-error">
              Doesn&apos;t fit in one {gpu?.name}. You&apos;d need more GPUs, a smaller batch/seq length, precision
              reduction, activation checkpointing, or ZeRO/FSDP sharding{peftLabel ? "" : ", or LoRA/QLoRA fine-tuning"}. To try more/larger GPUs, head to the{" "}
              <span className="font-medium">Datacenter</span> tab — it lets you scale up the cluster size and GPU
              type and see how a model like this actually splits across a bigger fleet.
            </p>
          )}
          {totalCapacity > 0 && zeroStage === 0 && !peftLabel && (
            <p className="mt-2 text-xs text-muted">
              Total across {numGpus} GPU{numGpus > 1 ? "s" : ""}: {formatGb(total)} of {formatGb(totalCapacity)} available
            </p>
          )}
        </>
      )}
      {loading && <p className="text-sm text-muted">Calculating…</p>}
    </Card>
  );
}
