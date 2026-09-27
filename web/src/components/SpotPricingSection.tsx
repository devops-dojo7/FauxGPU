"use client";

import { useEffect, useState } from "react";
import { calculateSpotPricing } from "@/lib/api";
import { CostResponse, SpotPricingResponse } from "@/lib/types";
import { formatCompact, formatUsd } from "@/lib/format";
import { Field, NumberInput, Stat, Toggle } from "./ui";

export interface SpotInputsState {
  enabled: boolean;
  checkpointIntervalSteps: number;
  checkpointSizeGb: number;
  preemptionsPer1000GpuHours: number;
  discount: number;
}

export function SpotPricingSection({
  state,
  onChange,
  cost,
  pricePerHr,
}: {
  state: SpotInputsState;
  onChange: (s: SpotInputsState) => void;
  cost: CostResponse | null;
  pricePerHr: number;
}) {
  const [result, setResult] = useState<SpotPricingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const set = (patch: Partial<SpotInputsState>) => onChange({ ...state, ...patch });

  useEffect(() => {
    if (!state.enabled || !cost) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clears stale results when the toggle is off or there's no base cost yet
      setResult(null);
      return;
    }
    calculateSpotPricing({
      on_demand_price_per_hr_usd: pricePerHr,
      total_gpus: cost.total_gpus,
      base_time_hours: cost.total_time_hours,
      step_time_s: cost.total_s_per_step,
      checkpoint_interval_steps: state.checkpointIntervalSteps,
      checkpoint_size_gb: state.checkpointSizeGb,
      preemptions_per_1000_gpu_hours: state.preemptionsPer1000GpuHours,
      discount: state.discount,
    })
      .then((r) => {
        setResult(r);
        setError(null);
      })
      .catch((e) => setError(e.message));
  }, [state.enabled, state.checkpointIntervalSteps, state.checkpointSizeGb, state.preemptionsPer1000GpuHours, state.discount, cost, pricePerHr]);

  return (
    <div className="border-t border-hairline pt-4 mt-4">
      <div className="flex items-center justify-between mb-3">
        <p className="text-xs uppercase tracking-wide text-muted-soft">
          What if this ran on spot/preemptible capacity?
        </p>
        <Toggle checked={state.enabled} onChange={(v) => set({ enabled: v })} label="Enable" />
      </div>

      {state.enabled && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
            <Field label="Spot discount">
              <NumberInput value={state.discount} min={0} max={0.95} step={0.05} onChange={(v) => set({ discount: v })} />
            </Field>
            <Field label="Preemptions / 1000 GPU-hr">
              <NumberInput
                value={state.preemptionsPer1000GpuHours}
                min={0}
                step={0.5}
                onChange={(v) => set({ preemptionsPer1000GpuHours: v })}
              />
            </Field>
            <Field label="Checkpoint interval (steps)">
              <NumberInput
                value={state.checkpointIntervalSteps}
                min={1}
                step={10}
                onChange={(v) => set({ checkpointIntervalSteps: v })}
              />
            </Field>
            <Field label="Checkpoint size (GB)">
              <NumberInput value={state.checkpointSizeGb} min={0} step={5} onChange={(v) => set({ checkpointSizeGb: v })} />
            </Field>
          </div>

          {error && <p className="text-sm text-error">{error}</p>}
          {result && !error && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Stat label="Spot price" value={`$${result.spot_price_per_hr_usd.toFixed(2)}/hr/GPU`} sub={`vs $${pricePerHr}/hr on-demand`} />
              <Stat
                label="Expected preemptions"
                value={result.expected_preemptions.toFixed(1)}
                sub={`≈ ${formatCompact(result.expected_lost_steps)} steps redone`}
              />
              <Stat
                label="Expected wall-clock"
                value={`${result.expected_wall_clock_hours.toFixed(1)} hr`}
                sub={`+${result.expected_recovery_overhead_hours.toFixed(1)}hr recovery`}
              />
              <Stat
                label="Expected total cost"
                value={formatUsd(result.expected_total_cost_usd)}
                sub={`${result.savings_pct.toFixed(0)}% savings vs ${formatUsd(result.on_demand_cost_usd)} on-demand`}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
