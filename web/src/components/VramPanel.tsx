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
}: {
  vram: VramResponse | null;
  gpu: GpuSpec | undefined;
  numGpus: number;
  loading: boolean;
  error: string | null;
}) {
  const capacityPerGpu = gpu?.vram_gb ?? 0;
  const totalCapacity = capacityPerGpu * numGpus;
  const total = vram?.total_gb ?? 0;
  // Naive even split across GPUs (real sharding strategies differ; this is for intuition).
  const perGpu = numGpus > 0 ? total / numGpus : total;
  const overflow = capacityPerGpu > 0 && perGpu > capacityPerGpu;

  return (
    <Card title="VRAM breakdown">
      {error && <p className="text-sm text-red-500">{error}</p>}
      {!error && vram && (
        <>
          <div className="h-8 w-full flex rounded-md overflow-hidden border border-black/10 dark:border-white/10">
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
                  <span className="text-black/60 dark:text-white/60">{s.label}</span>
                  <span className="ml-auto tabular-nums">{formatGb(v)}</span>
                </div>
              );
            })}
          </div>

          <div className="mt-4 border-t border-black/10 dark:border-white/10 pt-3 flex items-baseline justify-between">
            <span className="text-sm text-black/60 dark:text-white/60">Total (per GPU, {numGpus}-way split)</span>
            <span className={`text-lg font-semibold tabular-nums ${overflow ? "text-red-500" : ""}`}>
              {formatGb(perGpu)} {capacityPerGpu > 0 && <span className="text-sm font-normal">/ {capacityPerGpu} GB</span>}
            </span>
          </div>
          {overflow && (
            <p className="mt-1 text-xs text-red-500">
              Doesn&apos;t fit in one {gpu?.name}. You&apos;d need more GPUs, a smaller batch/seq length, precision
              reduction, or activation checkpointing.
            </p>
          )}
          {totalCapacity > 0 && (
            <p className="mt-2 text-xs text-black/45 dark:text-white/45">
              Total across {numGpus} GPU{numGpus > 1 ? "s" : ""}: {formatGb(total)} of {formatGb(totalCapacity)} available
            </p>
          )}
        </>
      )}
      {loading && <p className="text-sm text-black/45 dark:text-white/45">Calculating…</p>}
    </Card>
  );
}
