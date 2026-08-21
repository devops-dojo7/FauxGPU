"use client";

import { useMemo } from "react";
import ReactFlow, { Background, Edge, Node, Position } from "reactflow";
import "reactflow/dist/style.css";
import { GpuSpec, TopologyShape } from "@/lib/types";

const GPU_W = 76;
const GPU_H = 44;
const GPU_GAP = 24;
const NODE_GAP = 64;
const ROW_Y_GPU = 190;
const ROW_Y_SWITCH = 100;
const ROW_Y_FABRIC = 10;

function gpuNode(id: string, x: number, y: number, label: string): Node {
  return {
    id,
    position: { x, y },
    data: { label },
    sourcePosition: Position.Top,
    targetPosition: Position.Top,
    style: {
      width: GPU_W,
      height: GPU_H,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 11,
      borderRadius: 8,
      border: "1px solid rgba(59,130,246,0.5)",
      background: "rgba(59,130,246,0.12)",
    },
  };
}

function hubNode(id: string, x: number, y: number, label: string, color: string): Node {
  return {
    id,
    position: { x, y },
    data: { label },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Bottom,
    style: {
      width: 150,
      height: 40,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 11,
      fontWeight: 600,
      borderRadius: 8,
      border: `1px solid ${color}`,
      background: `${color}22`,
    },
  };
}

export function TopologyDiagram({
  shape,
  gpu,
  gpusPerNode,
  numNodes,
  fabricName,
  fabricBandwidthGbps,
}: {
  shape: TopologyShape;
  gpu: GpuSpec | undefined;
  gpusPerNode: number;
  numNodes: number;
  fabricName: string;
  fabricBandwidthGbps: number;
}) {
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const nvlinkGbps = gpu?.nvlink_gbps ?? null;

    if (shape === "single_gpu") {
      nodes.push(gpuNode("gpu-0", 200, ROW_Y_GPU, gpu?.name ?? "GPU"));
      return { nodes, edges };
    }

    const nodeWidth = gpusPerNode * GPU_W + (gpusPerNode - 1) * GPU_GAP;
    const totalWidth = numNodes * nodeWidth + (numNodes - 1) * NODE_GAP;
    let cursorX = 0;

    const switchIds: string[] = [];

    for (let n = 0; n < numNodes; n++) {
      const switchId = `switch-${n}`;
      const switchX = cursorX + nodeWidth / 2 - 75;
      nodes.push(hubNode(switchId, switchX, ROW_Y_SWITCH, `Node ${n} · NVLink`, "#3b82f6"));
      switchIds.push(switchId);

      for (let g = 0; g < gpusPerNode; g++) {
        const gpuId = `gpu-${n}-${g}`;
        const gpuX = cursorX + g * (GPU_W + GPU_GAP);
        nodes.push(gpuNode(gpuId, gpuX, ROW_Y_GPU, `GPU ${g}`));
        edges.push({
          id: `e-${switchId}-${gpuId}`,
          source: switchId,
          target: gpuId,
          label: nvlinkGbps ? `${nvlinkGbps} GB/s` : "",
          style: { stroke: "#3b82f6" },
          labelStyle: { fontSize: 10 },
        });
      }
      cursorX += nodeWidth + NODE_GAP;
    }

    if (shape === "multi_node" && numNodes > 1) {
      const fabricX = totalWidth / 2 - 75;
      nodes.push(hubNode("fabric", fabricX, ROW_Y_FABRIC, fabricName, "#f59e0b"));
      for (const switchId of switchIds) {
        edges.push({
          id: `e-fabric-${switchId}`,
          source: "fabric",
          target: switchId,
          label: `${fabricBandwidthGbps} Gb/s`,
          style: { stroke: "#f59e0b" },
          labelStyle: { fontSize: 10 },
        });
      }
    }

    return { nodes, edges };
  }, [shape, gpu, gpusPerNode, numNodes, fabricName, fabricBandwidthGbps]);

  return (
    <div className="h-72 w-full rounded-lg border border-black/10 dark:border-white/10 bg-white dark:bg-black/20">
      <ReactFlow nodes={nodes} edges={edges} fitView fitViewOptions={{ padding: 0.3 }} nodesDraggable={false} nodesConnectable={false} panOnDrag zoomOnScroll={false} proOptions={{ hideAttribution: true }}>
        <Background gap={16} />
      </ReactFlow>
    </div>
  );
}
