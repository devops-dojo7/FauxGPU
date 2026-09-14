"use client";

import { useMemo, useState } from "react";
import { GpuSpec } from "@/lib/types";
import { groupGpusByVendor } from "@/lib/format";
import { Card, Select, Toggle } from "./ui";
import { RadarChart, RadarAxis, RadarSeries } from "./RadarChart";

const MAX_SLOTS = 4;
const SLOT_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e"];
let nextSlotId = 1;

interface Slot {
  id: number;
  gpuId: string;
}

function makeSlot(gpuId: string): Slot {
  return { id: nextSlotId++, gpuId };
}

const na = (v: number | string | null | undefined, suffix = "") => (v === null || v === undefined || v === "" ? "—" : `${v}${suffix}`);

interface Row {
  key: string;
  label: string;
  lowerBetter?: boolean; // omit for informational rows with no "winner"
  raw: (g: GpuSpec) => number | null;
  display: (g: GpuSpec) => string;
}

const SECTIONS: { title: string; rows: Row[] }[] = [
  {
    title: "Vendor & architecture",
    rows: [
      { key: "vendor", label: "Vendor", raw: () => null, display: (g) => g.vendor },
      { key: "architecture", label: "Architecture", raw: () => null, display: (g) => na(g.architecture) },
      { key: "process_node", label: "Process node", raw: () => null, display: (g) => na(g.process_node) },
      { key: "launch_year", label: "Launch year", raw: () => null, display: (g) => na(g.launch_year) },
    ],
  },
  {
    title: "Compute",
    rows: [
      { key: "bf16", label: "BF16 TFLOPS (dense)", lowerBetter: false, raw: (g) => g.bf16_tflops, display: (g) => g.bf16_tflops.toLocaleString() },
      { key: "fp8", label: "FP8 TFLOPS (dense)", lowerBetter: false, raw: (g) => g.fp8_tflops, display: (g) => na(g.fp8_tflops?.toLocaleString()) },
      {
        key: "efficiency",
        label: "Efficiency (BF16 TFLOPS/W)",
        lowerBetter: false,
        raw: (g) => g.bf16_tflops / g.tdp_watts,
        display: (g) => (g.bf16_tflops / g.tdp_watts).toFixed(2),
      },
    ],
  },
  {
    title: "Memory",
    rows: [
      { key: "vram", label: "VRAM", lowerBetter: false, raw: (g) => g.vram_gb, display: (g) => `${g.vram_gb} GB` },
      { key: "bandwidth", label: "Memory bandwidth", lowerBetter: false, raw: (g) => g.mem_bandwidth_gbps, display: (g) => `${g.mem_bandwidth_gbps.toLocaleString()} GB/s` },
      { key: "nvlink", label: "NVLink / scale-up link", lowerBetter: false, raw: (g) => g.nvlink_gbps, display: (g) => (g.nvlink_gbps ? `${g.nvlink_gbps.toLocaleString()} GB/s` : "None (PCIe only)") },
    ],
  },
  {
    title: "Silicon",
    rows: [
      { key: "transistors", label: "Transistors", raw: () => null, display: (g) => na(g.transistors_b, "B") },
      { key: "sm_count", label: "SM / CU count", raw: () => null, display: (g) => na(g.sm_count) },
      { key: "core_count", label: "Core count", raw: () => null, display: (g) => (g.core_count ? `${g.core_count.toLocaleString()} (${g.core_label})` : "—") },
      { key: "matrix_core_count", label: "Matrix / tensor cores", raw: () => null, display: (g) => (g.matrix_core_count ? `${g.matrix_core_count.toLocaleString()} (${g.matrix_core_label})` : "—") },
      { key: "boost_clock", label: "Boost clock", raw: () => null, display: (g) => na(g.boost_clock_ghz, " GHz") },
      { key: "l2_cache", label: "L2 / on-die cache", raw: () => null, display: (g) => na(g.l2_cache_mb, " MB") },
    ],
  },
  {
    title: "Power & price",
    rows: [
      { key: "tdp", label: "TDP", lowerBetter: true, raw: (g) => g.tdp_watts, display: (g) => `${g.tdp_watts} W` },
      { key: "idle", label: "Idle power", lowerBetter: true, raw: (g) => g.idle_watts, display: (g) => `${g.idle_watts} W` },
      { key: "price", label: "Price", lowerBetter: true, raw: (g) => g.price_per_hr_usd, display: (g) => `$${g.price_per_hr_usd}/hr` },
    ],
  },
];

const RADAR_ROWS = ["vram", "bf16", "fp8", "bandwidth", "nvlink", "efficiency"];
const RADAR_LABELS: Record<string, string> = {
  vram: "VRAM",
  bf16: "BF16 throughput",
  fp8: "FP8 throughput",
  bandwidth: "Mem bandwidth",
  nvlink: "Scale-up link",
  efficiency: "Efficiency",
};

const FLOPS_ROWS: { key: "bf16" | "fp8"; label: string }[] = [
  { key: "bf16", label: "BF16" },
  { key: "fp8", label: "FP8" },
];

/** Grouped horizontal bars, one row per precision, one bar per selected GPU
 * within that row — skips a GPU entirely for a precision it has no figure
 * for (null), rather than drawing a misleading zero-length bar. */
function FlopsByPrecisionChart({ slots, slotGpus }: { slots: Slot[]; slotGpus: (GpuSpec | undefined)[] }) {
  const width = 480;
  const barH = 14;
  const barGap = 4;
  const rowGap = 20;
  const leftPad = 46;
  const rightPad = 56;
  const topPad = 8;

  const values = slotGpus.flatMap((g) => (g ? [g.bf16_tflops, g.fp8_tflops].filter((v): v is number => v !== null && v !== undefined) : []));
  const max = Math.max(1, ...values);
  const rowHeight = slots.length * barH + Math.max(0, slots.length - 1) * barGap;
  const height = topPad + FLOPS_ROWS.length * (rowHeight + rowGap);
  const plotW = width - leftPad - rightPad;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        {FLOPS_ROWS.map((row, ri) => {
          const y0 = topPad + ri * (rowHeight + rowGap);
          return (
            <g key={row.key}>
              <text x={leftPad - 8} y={y0 + rowHeight / 2} textAnchor="end" dominantBaseline="middle" className="fill-body text-[11px] font-medium">
                {row.label}
              </text>
              {slots.map((s, i) => {
                const g = slotGpus[i];
                const v = g ? (row.key === "bf16" ? g.bf16_tflops : g.fp8_tflops) : null;
                if (v === null || v === undefined) return null;
                const w = Math.max(1, (v / max) * plotW);
                const y = y0 + i * (barH + barGap);
                return (
                  <g key={s.id}>
                    <rect x={leftPad} y={y} width={w} height={barH} rx={3} fill={SLOT_COLORS[i % SLOT_COLORS.length]} />
                    <text x={leftPad + w + 6} y={y + barH / 2} dominantBaseline="middle" className="fill-muted-soft text-[10px]">
                      {v.toLocaleString()}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="text-xs text-muted-soft text-center mt-1">TFLOPS (dense)</p>
    </div>
  );
}

/** Scatter: TDP (x) vs BF16 throughput (y) — one bubble per GPU, so the
 * power/performance tradeoff reads at a glance rather than as two separate
 * numbers in a table. */
function EfficiencyChart({ slots, slotGpus }: { slots: Slot[]; slotGpus: (GpuSpec | undefined)[] }) {
  const width = 480;
  const height = 260;
  const padding = { top: 20, right: 16, bottom: 30, left: 40 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const points = slots.map((s, i) => ({ id: s.id, i, gpu: slotGpus[i] })).filter((p): p is { id: number; i: number; gpu: GpuSpec } => !!p.gpu);
  if (points.length === 0) return <div className="text-sm text-muted">No data.</div>;

  const xMax = Math.max(...points.map((p) => p.gpu.tdp_watts)) * 1.2;
  const yMax = Math.max(...points.map((p) => p.gpu.bf16_tflops)) * 1.2;
  const scaleX = (x: number) => (x / xMax) * plotW;
  const scaleY = (y: number) => plotH - (y / yMax) * plotH;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        <g transform={`translate(${padding.left},${padding.top})`}>
          {[0, 0.5, 1].map((t) => (
            <line key={t} x1={0} x2={plotW} y1={plotH * t} y2={plotH * t} className="stroke-hairline-strong" strokeWidth={1} />
          ))}
          {points.map((p) => {
            const cx = scaleX(p.gpu.tdp_watts);
            const cy = scaleY(p.gpu.bf16_tflops);
            return (
              <g key={p.id}>
                <circle cx={cx} cy={cy} r={9} fill={SLOT_COLORS[p.i % SLOT_COLORS.length]} fillOpacity={0.85} />
                <text x={cx} y={cy - 14} textAnchor="middle" className="fill-body text-[10px]">
                  {p.gpu.name}
                </text>
              </g>
            );
          })}
          <text x={-6} y={4} textAnchor="end" className="fill-muted-soft" fontSize={10}>
            {yMax.toFixed(0)}
          </text>
          <text x={-6} y={plotH} textAnchor="end" className="fill-muted-soft" fontSize={10}>
            0
          </text>
          <text x={0} y={plotH + 18} textAnchor="start" className="fill-muted-soft" fontSize={10}>
            0 W
          </text>
          <text x={plotW} y={plotH + 18} textAnchor="end" className="fill-muted-soft" fontSize={10}>
            {xMax.toFixed(0)} W
          </text>
        </g>
      </svg>
      <p className="text-xs text-muted-soft text-center mt-1">TDP (W) → BF16 TFLOPS</p>
    </div>
  );
}

/** Min-max normalize raw values to a 20-100 range so the worst entry never collapses to the chart's center. */
function normalize(entries: { id: number; raw: number | null }[]): Record<number, number> {
  const valid = entries.filter((e): e is { id: number; raw: number } => e.raw !== null && !Number.isNaN(e.raw));
  if (valid.length === 0) return {};
  const vals = valid.map((v) => v.raw);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const out: Record<number, number> = {};
  for (const v of valid) {
    const t = min === max ? 1 : (v.raw - min) / (max - min);
    out[v.id] = 20 + t * 80;
  }
  return out;
}

export function GpuComparePanel({ gpus }: { gpus: GpuSpec[] }) {
  const [slots, setSlots] = useState<Slot[]>(() => [makeSlot(gpus[0]?.id ?? ""), makeSlot(gpus[1]?.id ?? gpus[0]?.id ?? "")]);
  const [highlightBest, setHighlightBest] = useState(true);
  const [diffsOnly, setDiffsOnly] = useState(false);

  const byVendor = groupGpusByVendor(gpus);

  const addSlot = () => {
    if (slots.length >= MAX_SLOTS) return;
    const unused = gpus.find((g) => !slots.some((s) => s.gpuId === g.id));
    setSlots((prev) => [...prev, makeSlot(unused?.id ?? gpus[0]?.id ?? "")]);
  };
  const removeSlot = (id: number) => setSlots((prev) => prev.filter((s) => s.id !== id));
  const updateSlot = (id: number, gpuId: string) => setSlots((prev) => prev.map((s) => (s.id === id ? { ...s, gpuId } : s)));

  const slotGpus = slots.map((s) => gpus.find((g) => g.id === s.gpuId)).filter((g): g is GpuSpec => !!g);

  const bestByRowKey = useMemo(() => {
    const map: Record<string, number | null> = {};
    for (const section of SECTIONS) {
      for (const row of section.rows) {
        if (row.lowerBetter === undefined) continue;
        const entries = slots.map((s, i) => ({ id: s.id, raw: slotGpus[i] ? row.raw(slotGpus[i]) : null }));
        const valid = entries.filter((e) => e.raw !== null) as { id: number; raw: number }[];
        if (valid.length < 2) {
          map[row.key] = null;
          continue;
        }
        let best = valid[0];
        for (const e of valid) if (row.lowerBetter ? e.raw < best.raw : e.raw > best.raw) best = e;
        const isTie = valid.every((e) => e.raw === best.raw);
        map[row.key] = isTie ? null : best.id;
      }
    }
    return map;
  }, [slots, slotGpus]);

  const rowDiffers = (row: Row) => {
    const values = slots.map((s, i) => (slotGpus[i] ? row.display(slotGpus[i]) : "—"));
    return new Set(values).size > 1;
  };

  const radarAxes: RadarAxis[] = RADAR_ROWS.map((key) => ({ key, label: RADAR_LABELS[key] }));
  const radarSeries: RadarSeries[] = useMemo(() => {
    const normalizedPerKey: Record<string, Record<number, number>> = {};
    for (const key of RADAR_ROWS) {
      const row = SECTIONS.flatMap((s) => s.rows).find((r) => r.key === key)!;
      normalizedPerKey[key] = normalize(slots.map((s, i) => ({ id: s.id, raw: slotGpus[i] ? row.raw(slotGpus[i]) : null })));
    }
    return slots.map((s, i) => ({
      id: s.id,
      label: slotGpus[i]?.name ?? "…",
      color: SLOT_COLORS[i % SLOT_COLORS.length],
      values: Object.fromEntries(RADAR_ROWS.map((key) => [key, normalizedPerKey[key][s.id] ?? 0])),
    }));
  }, [slots, slotGpus]);

  const gridTemplate = `180px repeat(${slots.length}, minmax(160px, 1fr))`;

  const mostEfficient = useMemo(() => {
    const valid = slotGpus.filter((g): g is GpuSpec => !!g);
    if (valid.length < 2) return null;
    return valid.reduce((best, g) => (g.bf16_tflops / g.tdp_watts > best.bf16_tflops / best.tdp_watts ? g : best));
  }, [slotGpus]);

  return (
    <Card title="Compare GPUs" className="overflow-hidden">
      <div className="flex items-start justify-between gap-4 flex-wrap mb-4">
        <p className="text-xs text-muted max-w-xl">
          Raw silicon specs, side by side — no model or workload involved. Pick up to {MAX_SLOTS} GPUs.
        </p>
        <div className="flex items-center gap-4">
          <Toggle checked={diffsOnly} onChange={setDiffsOnly} label="Differences only" />
          <Toggle checked={highlightBest} onChange={setHighlightBest} label="Highlight best" />
        </div>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: 180 + slots.length * 180 }}>
          {/* Header: color dot + GPU select + remove, per column */}
          <div className="grid gap-3 items-center pb-3" style={{ gridTemplateColumns: gridTemplate }}>
            <div />
            {slots.map((slot, i) => (
              <div key={slot.id} className="flex items-center gap-2 min-w-0">
                <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: SLOT_COLORS[i % SLOT_COLORS.length] }} />
                <div className="flex-1 min-w-0">
                  <Select value={slot.gpuId} onChange={(id) => updateSlot(slot.id, id)}>
                    {byVendor.map(([vendor, list]) => (
                      <optgroup key={vendor} label={vendor}>
                        {list.map((g) => (
                          <option key={g.id} value={g.id}>
                            {g.name}
                          </option>
                        ))}
                      </optgroup>
                    ))}
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
                + Add GPU
              </button>
            )}
          </div>

          {/* Radar chart */}
          <div className="flex flex-wrap items-center gap-8 justify-center py-6 border-b border-hairline">
            <div className="flex flex-col gap-1.5">
              {slots.map((s, i) => (
                <div key={s.id} className="flex items-center gap-2 text-xs">
                  <span className="h-2 w-2 rounded-full shrink-0" style={{ background: SLOT_COLORS[i % SLOT_COLORS.length] }} />
                  <span className="text-body-strong">{slotGpus[i]?.name ?? "…"}</span>
                </div>
              ))}
            </div>
            <RadarChart axes={radarAxes} series={radarSeries} />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 py-6 border-b border-hairline">
            <div className="rounded-lg border border-hairline p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">FLOPS by precision</p>
              <FlopsByPrecisionChart slots={slots} slotGpus={slotGpus} />
            </div>
            <div className="rounded-lg border border-hairline p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">Efficiency — TDP vs BF16 performance</p>
              <EfficiencyChart slots={slots} slotGpus={slotGpus} />
            </div>
          </div>

          {mostEfficient && (
            <div className="py-6 border-b border-hairline">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">Analysis</p>
              <div className="rounded-lg border border-success/30 bg-success/10 p-4 inline-block">
                <p className="text-xs text-muted mb-1">Most power efficient</p>
                <p className="text-base font-semibold text-success">{mostEfficient.name}</p>
                <p className="text-xs text-muted">{(mostEfficient.bf16_tflops / mostEfficient.tdp_watts).toFixed(2)} TFLOPS/W</p>
              </div>
            </div>
          )}

          {SECTIONS.map((section) => {
            const rows = diffsOnly ? section.rows.filter(rowDiffers) : section.rows;
            if (rows.length === 0) return null;
            return (
              <div key={section.title} className="pt-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted mb-2">{section.title}</p>
                {rows.map((row) => {
                  const bestId = highlightBest ? bestByRowKey[row.key] : null;
                  return (
                    <div key={row.key} className="grid gap-3 py-2 border-b border-hairline-soft last:border-0" style={{ gridTemplateColumns: gridTemplate }}>
                      <span className="text-sm text-body">{row.label}</span>
                      {slots.map((s, i) => {
                        const g = slotGpus[i];
                        const isBest = bestId === s.id;
                        return (
                          <span key={s.id} className={`text-sm tabular-nums ${isBest ? "text-success font-semibold" : "text-body-strong"}`}>
                            {g ? row.display(g) : "—"}
                          </span>
                        );
                      })}
                    </div>
                  );
                })}
              </div>
            );
          })}

          {slotGpus.some((g) => g.price_note) && (
            <div className="pt-4 grid gap-3" style={{ gridTemplateColumns: gridTemplate }}>
              <div />
              {slots.map((s, i) => (
                <span key={s.id} className="text-xs text-muted">
                  {slotGpus[i]?.price_note ?? ""}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
