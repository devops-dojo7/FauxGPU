"use client";

import { GpuSpec } from "@/lib/types";
import { BadgePill, Card, Field, Select, Stat } from "./ui";

const na = (v: number | string | null | undefined, suffix = "") => (v === null || v === undefined ? "—" : `${v}${suffix}`);

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
        <>
          {gpu.architecture && (
            <div className="flex flex-wrap gap-2 mt-4">
              <BadgePill>{gpu.architecture}</BadgePill>
              {gpu.process_node && <BadgePill className="normal-case tracking-normal font-medium text-body-strong">{gpu.process_node}</BadgePill>}
              {gpu.launch_year && <BadgePill className="normal-case tracking-normal font-medium text-body-strong">{gpu.launch_year}</BadgePill>}
            </div>
          )}

          <div className="mt-4 grid grid-cols-2 gap-4">
            <Stat label="VRAM" value={`${gpu.vram_gb} GB`} />
            <Stat label="Price" value={`$${gpu.price_per_hr_usd}/hr`} sub={gpu.price_note ?? undefined} />
            <Stat label="BF16 TFLOPS" value={gpu.bf16_tflops.toLocaleString()} />
            <Stat label="FP8 TFLOPS" value={gpu.fp8_tflops ? gpu.fp8_tflops.toLocaleString() : "—"} />
            <Stat label="Mem bandwidth" value={`${gpu.mem_bandwidth_gbps.toLocaleString()} GB/s`} />
            <Stat label="NVLink" value={gpu.nvlink_gbps ? `${gpu.nvlink_gbps} GB/s` : "None (PCIe only)"} />
          </div>

          <div className="mt-5 pt-4 border-t border-hairline">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">Silicon</p>
            <div className="grid grid-cols-2 gap-4">
              <Stat label="Transistors" value={na(gpu.transistors_b, "B")} />
              <Stat label="SM / CU count" value={na(gpu.sm_count)} />
              <Stat label={gpu.core_label} value={na(gpu.core_count?.toLocaleString())} />
              <Stat label={gpu.matrix_core_label ?? "Matrix cores"} value={na(gpu.matrix_core_count?.toLocaleString())} />
              <Stat label="Boost clock" value={na(gpu.boost_clock_ghz, " GHz")} />
              <Stat label="L2 / on-die cache" value={na(gpu.l2_cache_mb, " MB")} />
            </div>
            {gpu.interconnect_name && (
              <p className="text-xs text-muted mt-3">
                Interconnect: <span className="text-body-strong">{gpu.interconnect_name}</span>
              </p>
            )}
          </div>
        </>
      )}
    </Card>
  );
}
