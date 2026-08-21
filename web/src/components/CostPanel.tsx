"use client";

import { CostResponse } from "@/lib/types";
import { formatCompact, formatUsd } from "@/lib/format";
import { Card, Field, NumberInput, Stat } from "./ui";

export interface CostInputsState {
  tokensPerStep: number;
  totalTrainingTokensB: number; // billions, for a friendlier input
  utilization: number;
  tpDegree: number;
  ppDegree: number;
  batchSize: number;
  seqLen: number;
  numMicrobatches: number;
}

export function CostPanel({
  state,
  onChange,
  cost,
  loading,
  error,
  numGpus,
  pricePerHr,
}: {
  state: CostInputsState;
  onChange: (s: CostInputsState) => void;
  cost: CostResponse | null;
  loading: boolean;
  error: string | null;
  numGpus: number;
  pricePerHr: number;
}) {
  const set = (patch: Partial<CostInputsState>) => onChange({ ...state, ...patch });
  const effectiveGpus = cost?.total_gpus ?? numGpus;

  return (
    <Card title="Training cost">
      <div className="grid grid-cols-3 gap-3 mb-4">
        <Field label="Tokens / step">
          <NumberInput value={state.tokensPerStep} min={1} step={1024} onChange={(v) => set({ tokensPerStep: v })} />
        </Field>
        <Field label="Training tokens (B)">
          <NumberInput value={state.totalTrainingTokensB} min={0.1} step={10} onChange={(v) => set({ totalTrainingTokensB: v })} />
        </Field>
        <Field label="GPU utilization">
          <NumberInput value={state.utilization} min={0.05} max={1} step={0.05} onChange={(v) => set({ utilization: v })} />
        </Field>
      </div>

      <p className="text-xs uppercase tracking-wide text-black/40 dark:text-white/40 mb-2">
        Parallelism (adds on top of the {numGpus}-GPU topology above)
      </p>
      <div className="grid grid-cols-3 md:grid-cols-5 gap-3 mb-4">
        <Field label="Tensor parallel (TP)">
          <NumberInput value={state.tpDegree} min={1} max={16} onChange={(v) => set({ tpDegree: v })} />
        </Field>
        <Field label="Pipeline parallel (PP)">
          <NumberInput value={state.ppDegree} min={1} max={16} onChange={(v) => set({ ppDegree: v })} />
        </Field>
        <Field label="Microbatches">
          <NumberInput value={state.numMicrobatches} min={1} max={256} onChange={(v) => set({ numMicrobatches: v })} />
        </Field>
        <Field label="Batch size">
          <NumberInput value={state.batchSize} min={1} onChange={(v) => set({ batchSize: v })} />
        </Field>
        <Field label="Seq length">
          <NumberInput value={state.seqLen} min={1} step={128} onChange={(v) => set({ seqLen: v })} />
        </Field>
      </div>

      <p className="text-xs text-black/45 dark:text-white/45 mb-3">
        Cluster: {effectiveGpus} GPU{effectiveGpus > 1 ? "s" : ""} × ${pricePerHr}/hr = ${(effectiveGpus * pricePerHr).toFixed(2)}/hr
        {(state.tpDegree > 1 || state.ppDegree > 1) && ` (${numGpus} × TP${state.tpDegree} × PP${state.ppDegree})`}
      </p>

      {error && <p className="text-sm text-red-500">{error}</p>}
      {cost && !error && (
        <>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <Stat
              label="Step time"
              value={`${(cost.total_s_per_step * 1000).toFixed(0)} ms`}
              sub={`compute ${(cost.compute_s_per_step * 1000).toFixed(0)}ms + DP comm ${(cost.communication_s_per_step * 1000).toFixed(0)}ms${cost.tp_communication_s_per_step > 0 ? ` + TP comm ${(cost.tp_communication_s_per_step * 1000).toFixed(0)}ms` : ""}${cost.pipeline_bubble_s_per_step > 0 ? ` + PP bubble ${(cost.pipeline_bubble_s_per_step * 1000).toFixed(0)}ms` : ""}`}
            />
            <Stat label="Total steps" value={formatCompact(cost.total_steps)} />
            <Stat label="Wall-clock time" value={`${cost.total_time_hours.toFixed(1)} hr`} sub={`${(cost.total_time_hours / 24).toFixed(1)} days`} />
            <Stat label="Total cost" value={formatUsd(cost.total_cost_usd)} sub={`${formatUsd(cost.cost_per_1k_tokens_usd)} / 1K tokens`} />
          </div>
          <div className="border-t border-black/10 dark:border-white/10 pt-4 grid grid-cols-2 md:grid-cols-3 gap-4">
            <Stat label="Power per GPU" value={`${cost.power_watts_per_gpu.toFixed(0)} W`} sub="time-weighted avg" />
            <Stat label="Cluster power" value={`${cost.total_power_kw.toFixed(1)} kW`} sub={`${effectiveGpus} GPUs`} />
            <Stat label="Total energy" value={`${formatCompact(cost.total_energy_kwh)} kWh`} sub="for the full training run" />
          </div>
        </>
      )}
      {loading && <p className="text-sm text-black/45 dark:text-white/45">Calculating…</p>}
    </Card>
  );
}
