"use client";

import { Fabric, GpuSpec, TopologyShape } from "@/lib/types";
import { Card, Field, NumberInput, Select } from "./ui";

export interface TopologyState {
  shape: TopologyShape;
  gpusPerNode: number;
  numNodes: number;
  fabricId: string;
}

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

  return (
    <Card title="Cluster topology">
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
