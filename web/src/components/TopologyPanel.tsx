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

interface AiSystemPreset extends DcPreset {
  gpuId: string;
}

// Real, pre-wired multi-GPU appliances — each one is just an existing GPU
// spec at a specific node topology, not a distinct GPU. GB200 NVL72's 72
// GPUs form one flat non-blocking NVLink domain (NVSwitch fabric, not a
// multi-node network hop), so it's modeled as a single very-wide
// "nvlink_node" rather than "multi_node".
const AI_SYSTEM_PRESETS: AiSystemPreset[] = [
  { id: "dgx-h100", label: "DGX / HGX H100", sub: "8x H100, 640GB", gpuId: "h100-sxm", shape: "nvlink_node", gpusPerNode: 8, numNodes: 1 },
  { id: "dgx-a100", label: "DGX A100", sub: "8x A100 80GB, 640GB", gpuId: "a100-80gb-sxm", shape: "nvlink_node", gpusPerNode: 8, numNodes: 1 },
  { id: "dgx-h200", label: "DGX H200", sub: "8x H200, 1.13TB", gpuId: "h200-sxm", shape: "nvlink_node", gpusPerNode: 8, numNodes: 1 },
  { id: "dgx-b200", label: "DGX / HGX B200", sub: "8x B200, 1.44TB", gpuId: "b200-sxm", shape: "nvlink_node", gpusPerNode: 8, numNodes: 1 },
  // NVIDIA only publishes GB200/GB300 specs as whole-superchip (2-GPU)
  // aggregates, not a clean per-GPU split — so this preset reuses the
  // standalone b200-sxm entry (same Blackwell compute die) rather than
  // inventing a distinct "gb200" GPU from a halved, unsourced number.
  { id: "gb200-nvl72", label: "GB200 NVL72", sub: "72x B200-class GPUs, one NVLink domain", gpuId: "b200-sxm", shape: "nvlink_node", gpusPerNode: 72, numNodes: 1 },
];

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
  gpus,
  onSelectGpu,
  fabrics,
}: {
  state: TopologyState;
  onChange: (s: TopologyState) => void;
  gpu: GpuSpec | undefined;
  gpus: GpuSpec[];
  onSelectGpu: (id: string) => void;
  fabrics: Fabric[];
}) {
  const set = (patch: Partial<TopologyState>) => onChange({ ...state, ...patch });
  const nvlinkAvailable = !!gpu?.nvlink_gbps;

  const activePresetId = DC_PRESETS.find(
    (p) => p.shape === state.shape && p.gpusPerNode === state.gpusPerNode && (p.shape !== "multi_node" || p.numNodes === state.numNodes),
  )?.id;

  const activeAiSystemId = AI_SYSTEM_PRESETS.find(
    (p) => p.gpuId === gpu?.id && p.shape === state.shape && p.gpusPerNode === state.gpusPerNode && p.numNodes === state.numNodes,
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

      <div className="mb-4">
        <p className="text-xs text-body mb-2">AI systems — pre-wired multi-GPU appliances</p>
        <div className="flex flex-wrap gap-2">
          {AI_SYSTEM_PRESETS.map((p) => {
            const presetGpu = gpus.find((g) => g.id === p.gpuId);
            return (
              <button
                key={p.id}
                onClick={() => {
                  onSelectGpu(p.gpuId);
                  set({ shape: p.shape, gpusPerNode: p.gpusPerNode, numNodes: p.numNodes });
                }}
                disabled={!presetGpu}
                title={p.sub}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors disabled:opacity-30 ${
                  activeAiSystemId === p.id
                    ? "bg-primary text-on-primary border-primary"
                    : "border-hairline-strong text-body hover:bg-surface-strong"
                }`}
              >
                {p.label} <span className="opacity-70">· {p.sub}</span>
              </button>
            );
          })}
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
            <NumberInput value={state.gpusPerNode} min={1} max={128} onChange={(v) => set({ gpusPerNode: v })} />
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
