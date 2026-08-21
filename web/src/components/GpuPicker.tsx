"use client";

import { GpuSpec } from "@/lib/types";
import { Card, Field, Select, Stat } from "./ui";

export function GpuPicker({
  gpus,
  selectedId,
  onSelect,
}: {
  gpus: GpuSpec[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const gpu = gpus.find((g) => g.id === selectedId);
  const byVendor = gpus.reduce<Record<string, GpuSpec[]>>((acc, g) => {
    (acc[g.vendor] ??= []).push(g);
    return acc;
  }, {});

  return (
    <Card title="GPU & Vendor">
      <Field label="GPU">
        <Select value={selectedId} onChange={onSelect}>
          {Object.entries(byVendor).map(([vendor, list]) => (
            <optgroup key={vendor} label={vendor}>
              {list.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.name}
                </option>
              ))}
            </optgroup>
          ))}
        </Select>
      </Field>

      {gpu && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          <Stat label="VRAM" value={`${gpu.vram_gb} GB`} />
          <Stat label="Price" value={`$${gpu.price_per_hr_usd}/hr`} />
          <Stat label="BF16 TFLOPS" value={gpu.bf16_tflops.toLocaleString()} />
          <Stat label="Mem bandwidth" value={`${gpu.mem_bandwidth_gbps.toLocaleString()} GB/s`} />
          <Stat label="NVLink" value={gpu.nvlink_gbps ? `${gpu.nvlink_gbps} GB/s` : "None (PCIe only)"} />
        </div>
      )}
    </Card>
  );
}
