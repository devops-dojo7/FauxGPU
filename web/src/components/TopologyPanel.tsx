"use client";

import { Fabric, GpuSpec, TopologyShape } from "@/lib/types";
import { Card, Field, NumberInput, Select } from "./ui";

export interface TopologyState {
  shape: TopologyShape;
  gpusPerNode: number;
  numNodes: number;
  fabricId: string;
}

interface DcPreset {
  id: string;
  label: string;
  sub: string;
  shape: TopologyShape;
  gpusPerNode: number;
  numNodes: number;
}

// Round-number scale presets, from a laptop-friendly single GPU up to a
// frontier-lab-scale cluster (xAI Colossus/OpenAI-class training clusters are
// widely reported in the ~100K H100 range) — not meant to model any specific
// real deployment, just to give an intuitive sense of how cost/power/network
// topology change by orders of magnitude.
const DC_PRESETS: DcPreset[] = [
  { id: "single", label: "Single GPU", sub: "1 GPU", shape: "single_gpu", gpusPerNode: 1, numNodes: 1 },
  { id: "node", label: "Single node", sub: "8 GPUs", shape: "nvlink_node", gpusPerNode: 8, numNodes: 1 },
  { id: "small", label: "Small cluster", sub: "64 GPUs", shape: "multi_node", gpusPerNode: 8, numNodes: 8 },
  { id: "large", label: "Large cluster", sub: "1,024 GPUs", shape: "multi_node", gpusPerNode: 8, numNodes: 128 },
  { id: "frontier", label: "Frontier scale", sub: "100,000 GPUs", shape: "multi_node", gpusPerNode: 8, numNodes: 12500 },
];

export function TopologyPanel({
  state,
  onChange,
  gpu,
  fabrics,
}: {
  state: TopologyState;
  onChange: (s: TopologyState) => void;
  gpu: GpuSpec | undefined;
  fabrics: Fabric[];
}) {
  const set = (patch: Partial<TopologyState>) => onChange({ ...state, ...patch });
  const nvlinkAvailable = !!gpu?.nvlink_gbps;

  const activePresetId = DC_PRESETS.find(
    (p) => p.shape === state.shape && p.gpusPerNode === state.gpusPerNode && (p.shape !== "multi_node" || p.numNodes === state.numNodes),
  )?.id;

  return (
    <Card title="Cluster topology">
      <div className="mb-4">
        <p className="text-xs text-body mb-2">Datacenter scale</p>
        <div className="flex flex-wrap gap-2">
          {DC_PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => set({ shape: p.shape, gpusPerNode: p.gpusPerNode, numNodes: p.numNodes })}
              disabled={p.shape !== "single_gpu" && !nvlinkAvailable}
              title={p.sub}
              className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors disabled:opacity-30 ${
                activePresetId === p.id
                  ? "bg-primary text-on-primary border-primary"
                  : "border-hairline-strong text-body hover:bg-surface-strong"
              }`}
            >
              {p.label} <span className="opacity-70">· {p.sub}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="col-span-2">
          <Field label="Shape">
            <Select value={state.shape} onChange={(v) => set({ shape: v as TopologyShape })}>
              <option value="single_gpu">Single GPU</option>
              <option value="nvlink_node" disabled={!nvlinkAvailable}>
                Single node ({nvlinkAvailable ? "NVLink" : "no NVLink on this GPU"})
              </option>
              <option value="multi_node" disabled={!nvlinkAvailable}>
                Multi-node (NVLink + InfiniBand/Ethernet)
              </option>
            </Select>
          </Field>
        </div>

        {state.shape !== "single_gpu" && (
          <Field label="GPUs per node">
            <NumberInput value={state.gpusPerNode} min={1} max={16} onChange={(v) => set({ gpusPerNode: v })} />
          </Field>
        )}
        {state.shape === "multi_node" && (
          <>
            <Field label="Number of nodes">
              <NumberInput value={state.numNodes} min={1} onChange={(v) => set({ numNodes: v })} />
            </Field>
            <div className="col-span-2">
              <Field label="Inter-node fabric">
                <Select value={state.fabricId} onChange={(v) => set({ fabricId: v })}>
                  {fabrics.map((f) => (
                    <option key={f.id} value={f.id}>
                      {f.name} ({f.bandwidth_gbps} Gb/s)
                    </option>
                  ))}
                </Select>
              </Field>
            </div>
          </>
        )}
      </div>
    </Card>
  );
}
