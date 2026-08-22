"use client";

import { GpuSpec } from "@/lib/types";
import { Card, Stat } from "./ui";
import { formatUsd } from "@/lib/format";

/** Quick, client-side scale economics — no training/inference config needed,
 * just "how big and how expensive is this many of this GPU". */
export function DatacenterStats({ gpu, numGpus }: { gpu: GpuSpec | undefined; numGpus: number }) {
  if (!gpu) return null;

  const powerKw = (gpu.tdp_watts * numGpus) / 1000;
  const powerLabel = powerKw >= 1000 ? `${(powerKw / 1000).toFixed(1)} MW` : `${powerKw.toFixed(1)} kW`;
  const costPerHr = gpu.price_per_hr_usd * numGpus;

  return (
    <Card title="At this scale">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="Total GPUs" value={numGpus.toLocaleString()} sub={gpu.name} />
        <Stat label="Peak cluster power" value={powerLabel} sub={`${gpu.tdp_watts} W/GPU`} />
        <Stat label="Cluster cost" value={`${formatUsd(costPerHr)}/hr`} sub={`${formatUsd(gpu.price_per_hr_usd)}/GPU-hr`} />
        <Stat label="Extrapolated" value={`${formatUsd(costPerHr * 24)}/day`} sub={`${formatUsd(costPerHr * 24 * 30)}/mo`} />
      </div>
    </Card>
  );
}
