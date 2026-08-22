"use client";

import { GpuSpec, ModelShape, MODEL_PRESETS } from "@/lib/types";
import { formatGb, formatUsd } from "@/lib/format";
import { BadgePill, Field, ModelArchBadges, NumberInput, Select, Stat } from "./ui";
import type { CompareSlot, ColumnResult } from "./ComparePanel";

const CUSTOM_DEFAULT_MODEL: ModelShape = { params: 7e9, num_layers: 32, hidden_dim: 4096, num_heads: 32, head_dim: 128 };

export function CompareColumn({
  slot,
  result,
  gpu,
  numGpus,
  canRemove,
  isBestVram,
  isBestCost,
  isBestThroughput,
  onChange,
  onRemove,
}: {
  slot: CompareSlot;
  result: ColumnResult | undefined;
  gpu: GpuSpec | undefined;
  numGpus: number;
  canRemove: boolean;
  isBestVram: boolean;
  isBestCost: boolean;
  isBestThroughput: boolean;
  onChange: (patch: Partial<CompareSlot>) => void;
  onRemove: () => void;
}) {
  const set = (patch: Partial<CompareSlot>) => onChange(patch);

  const onPreset = (id: string) => {
    if (id === "custom") {
      set({ presetId: "custom", model: slot.model ?? CUSTOM_DEFAULT_MODEL });
      return;
    }
    const preset = MODEL_PRESETS.find((p) => p.id === id);
    if (preset) set({ presetId: id, model: { ...preset } });
  };

  const capacityPerGpu = gpu?.vram_gb ?? 0;
  const perGpuVram = result?.vram ? result.vram.total_gb / numGpus : null;
  const overflow = perGpuVram !== null && capacityPerGpu > 0 && perGpuVram > capacityPerGpu;

  return (
    <div className="w-72 shrink-0 rounded-xl border border-hairline bg-surface-card p-5 flex flex-col gap-4">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <Select value={slot.presetId} onChange={onPreset}>
            {MODEL_PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
            <option value="custom">Custom</option>
          </Select>
        </div>
        <button
          onClick={onRemove}
          disabled={!canRemove}
          className="text-muted-soft hover:text-error disabled:opacity-30 px-1 shrink-0"
          title="Remove from comparison"
        >
          ✕
        </button>
      </div>

      {slot.presetId === "custom" && (
        <div className="flex flex-col gap-2">
          <input
            type="text"
            className="rounded-md border border-hairline-strong bg-surface-card px-2.5 py-1.5 text-sm text-ink outline-none focus:border-ink"
            value={slot.customName}
            onChange={(e) => set({ customName: e.target.value })}
            placeholder="Model name"
          />
          <div className="grid grid-cols-2 gap-2">
            <Field label="Params (B)">
              <NumberInput value={slot.model.params / 1e9} min={0.01} step={0.1} onChange={(v) => set({ model: { ...slot.model, params: v * 1e9 } })} />
            </Field>
            <Field label="Layers">
              <NumberInput value={slot.model.num_layers} min={1} onChange={(v) => set({ model: { ...slot.model, num_layers: v } })} />
            </Field>
            <Field label="Hidden dim">
              <NumberInput value={slot.model.hidden_dim} min={1} onChange={(v) => set({ model: { ...slot.model, hidden_dim: v } })} />
            </Field>
            <Field label="Heads">
              <NumberInput value={slot.model.num_heads} min={1} onChange={(v) => set({ model: { ...slot.model, num_heads: v } })} />
            </Field>
            <Field label="Head dim">
              <NumberInput value={slot.model.head_dim} min={1} onChange={(v) => set({ model: { ...slot.model, head_dim: v } })} />
            </Field>
            <Field label="KV heads">
              <NumberInput
                value={slot.model.num_kv_heads ?? slot.model.num_heads}
                min={1}
                onChange={(v) => set({ model: { ...slot.model, num_kv_heads: v } })}
              />
            </Field>
          </div>
        </div>
      )}

      <ModelArchBadges model={slot.model} />

      <Field label="Precision">
        <Select value={slot.precision} onChange={(v) => set({ precision: v })}>
          <option value="fp32">fp32</option>
          <option value="bf16">bf16</option>
          <option value="fp16">fp16</option>
          <option value="fp8">fp8</option>
        </Select>
      </Field>

      {result?.error && <p className="text-xs text-error">{result.error}</p>}

      <div className="border-t border-hairline pt-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">VRAM / GPU</span>
          {isBestVram && <BadgePill className="bg-success/10 text-success normal-case tracking-normal">Smallest</BadgePill>}
        </div>
        <p className={`text-lg font-semibold tabular-nums ${overflow ? "text-error" : "text-ink"}`}>
          {perGpuVram !== null ? formatGb(perGpuVram) : "—"}
          {capacityPerGpu > 0 && <span className="text-sm font-normal text-muted"> / {capacityPerGpu} GB</span>}
        </p>
        {overflow && <p className="text-xs text-error mt-0.5">Doesn&apos;t fit</p>}
      </div>

      <div className="border-t border-hairline pt-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">Training cost</span>
          {isBestCost && <BadgePill className="bg-success/10 text-success normal-case tracking-normal">Cheapest</BadgePill>}
        </div>
        {result?.cost ? (
          <>
            <Stat label="Total cost" value={formatUsd(result.cost.total_cost_usd)} sub={`${result.cost.total_time_hours.toFixed(1)} hr wall-clock`} />
            <p className="text-xs text-muted mt-1">
              {(result.cost.total_s_per_step * 1000).toFixed(0)}ms/step · {formatUsd(result.cost.cost_per_1k_tokens_usd)}/1K tokens
            </p>
          </>
        ) : (
          <p className="text-sm text-muted">—</p>
        )}
      </div>

      <div className="border-t border-hairline pt-3">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">Inference</span>
          {isBestThroughput && <BadgePill className="bg-success/10 text-success normal-case tracking-normal">Fastest</BadgePill>}
        </div>
        {result?.inference ? (
          <>
            <Stat label="Tokens/sec per GPU" value={result.inference.disaggregated_tokens_per_sec_per_gpu.toFixed(0)} sub={`TTFT ${result.inference.ttft_ms.toFixed(0)} ms`} />
            <p className="text-xs text-muted mt-1">Max concurrent seqs: {result.inference.max_concurrent_sequences}</p>
          </>
        ) : (
          <p className="text-sm text-muted">—</p>
        )}
      </div>

      {result?.loading && <p className="text-xs text-muted">Calculating…</p>}
    </div>
  );
}
