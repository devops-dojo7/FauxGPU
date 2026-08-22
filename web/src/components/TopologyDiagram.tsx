"use client";

import { useEffect, useMemo, useRef } from "react";
import ReactFlow, { Background, Edge, Node, Position, ReactFlowProvider, useReactFlow } from "reactflow";
import "reactflow/dist/style.css";
import { GpuSpec, TopologyShape } from "@/lib/types";

const GPU_W = 76;
const GPU_H = 44;
const GPU_GAP = 24;
const NODE_GAP = 64;
const ROW_Y_GPU = 190;
const ROW_Y_SWITCH = 100;
const ROW_Y_FABRIC = 10;

// At frontier scale (thousands of nodes) drawing one box per node/GPU would
// freeze the browser. Draw a handful of real nodes for intuition, then
// collapse the rest into up to RACK_COUNT "rack" boxes — each one
// aggregating a slice of the remaining nodes and wired to the fabric with
// its own link, so it still reads as "a bunch of networked racks" rather
// than one opaque placeholder.
const MAX_DRAWN_NODES = 6;
const RACK_COUNT = 5;
const RACK_W = 92;
const RACK_H = 64;
const RACK_GAP = 20;
const RACK_COLOR = "#8b5cf6";

function gpuNode(id: string, x: number, y: number, label: string): Node {
  return {
    id,
    position: { x, y },
    data: { label },
    // Top-level width/height (distinct from style.width/height below) tell
    // fitView each node's real size immediately, instead of waiting on a
    // ResizeObserver measurement pass — with dozens of nodes appearing at
    // once (frontier-scale racks), some would otherwise still be
    // "unmeasured" when fitView's first bounding-box calc runs, and get
    // excluded from the fit entirely.
    width: GPU_W,
    height: GPU_H,
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
      color: "var(--color-ink)",
    },
  };
}

function hubNode(id: string, x: number, y: number, label: string, color: string): Node {
  return {
    id,
    position: { x, y },
    data: { label },
    width: 150,
    height: 40,
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
      color: "var(--color-ink)",
    },
  };
}

// A "rack" aggregates many real nodes into one box with a shelf-striped
// background (via repeating-linear-gradient) to visually read as a rack of
// servers rather than a single opaque GPU/node — still wired to the fabric
// like any other node.
function rackNode(id: string, x: number, y: number, label: string): Node {
  return {
    id,
    position: { x, y },
    data: { label },
    width: RACK_W,
    height: RACK_H,
    sourcePosition: Position.Top,
    targetPosition: Position.Bottom,
    style: {
      width: RACK_W,
      height: RACK_H,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: 10,
      fontWeight: 600,
      lineHeight: 1.35,
      textAlign: "center",
      whiteSpace: "pre-line",
      padding: 4,
      borderRadius: 6,
      border: `1px solid ${RACK_COLOR}`,
      background: `repeating-linear-gradient(0deg, ${RACK_COLOR}26 0px, ${RACK_COLOR}26 4px, transparent 4px, transparent 10px)`,
      color: "var(--color-ink)",
    },
  };
}

// The declarative `fitView` prop only fits once on ReactFlow's own initial
// mount, and the imperative `fitView()` call reads from ReactFlow's
// internal node store — which can still be mid-sync with a just-changed
// `nodes` prop (a one-frame race that fits a stale, smaller topology,
// observed in practice after several fast topology switches). Since we
// already know every node's exact position and size ourselves (we just
// generated them above), we compute the correct zoom/pan directly from
// that data instead of trusting ReactFlow's internal state to have caught
// up — fully deterministic, no timing dependency at all.
function FitViewOnChange({ nodes, containerRef }: { nodes: Node[]; containerRef: React.RefObject<HTMLDivElement | null> }) {
  const { setViewport } = useReactFlow();

  useEffect(() => {
    const el = containerRef.current;
    if (nodes.length === 0 || !el) return;
    const { width: containerWidth, height: containerHeight } = el.getBoundingClientRect();
    if (!containerWidth || !containerHeight) return;

    const PADDING = 0.15;
    const minX = Math.min(...nodes.map((n) => n.position.x));
    const minY = Math.min(...nodes.map((n) => n.position.y));
    const maxX = Math.max(...nodes.map((n) => n.position.x + (typeof n.width === "number" ? n.width : 150)));
    const maxY = Math.max(...nodes.map((n) => n.position.y + (typeof n.height === "number" ? n.height : 40)));
    const contentW = Math.max(1, maxX - minX);
    const contentH = Math.max(1, maxY - minY);

    const zoom = Math.min((containerWidth * (1 - PADDING)) / contentW, (containerHeight * (1 - PADDING)) / contentH, 1.5);
    const x = containerWidth / 2 - (minX + contentW / 2) * zoom;
    const y = containerHeight / 2 - (minY + contentH / 2) * zoom;

    setViewport({ x, y, zoom }, { duration: 0 });
  }, [nodes, containerRef, setViewport]);
  return null;
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
  const containerRef = useRef<HTMLDivElement>(null);
  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    const nvlinkGbps = gpu?.nvlink_gbps ?? null;

    if (shape === "single_gpu") {
      nodes.push(gpuNode("gpu-0", 200, ROW_Y_GPU, gpu?.name ?? "GPU"));
      return { nodes, edges };
    }

    const nodeWidth = gpusPerNode * GPU_W + (gpusPerNode - 1) * GPU_GAP;
    const drawnNodeCount = Math.min(numNodes, MAX_DRAWN_NODES);
    const remainingNodes = numNodes - drawnNodeCount;
    const rackCount = remainingNodes > 0 ? Math.min(RACK_COUNT, remainingNodes) : 0;
    const racksWidth = rackCount > 0 ? rackCount * RACK_W + (rackCount - 1) * RACK_GAP : 0;

    const drawnNodesWidth = drawnNodeCount * nodeWidth + Math.max(0, drawnNodeCount - 1) * NODE_GAP;
    const totalWidth = drawnNodesWidth + (rackCount > 0 ? NODE_GAP + racksWidth : 0);

    let cursorX = 0;
    // Each network-fabric link: which node/rack it goes to, and how many
    // real nodes' worth of bandwidth it represents (>1 for a rack).
    const fabricLinks: { id: string; nodeCount: number }[] = [];

    for (let n = 0; n < drawnNodeCount; n++) {
      const switchId = `switch-${n}`;
      const switchX = cursorX + nodeWidth / 2 - 75;
      nodes.push(hubNode(switchId, switchX, ROW_Y_SWITCH, `Node ${n} · NVLink`, "#3b82f6"));
      fabricLinks.push({ id: switchId, nodeCount: 1 });

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

    if (rackCount > 0) {
      const nodesPerRack = Math.ceil(remainingNodes / rackCount);
      let left = remainingNodes;
      let rackX = cursorX;
      for (let r = 0; r < rackCount; r++) {
        const rackNodes = r === rackCount - 1 ? left : Math.min(nodesPerRack, left);
        left -= rackNodes;
        const rackId = `rack-${r}`;
        const rackY = ROW_Y_SWITCH + (ROW_Y_GPU - ROW_Y_SWITCH - RACK_H) / 2;
        nodes.push(
          rackNode(
            rackId,
            rackX,
            rackY,
            `Rack ${r + 1}\n${rackNodes.toLocaleString()} node${rackNodes > 1 ? "s" : ""}\n${(rackNodes * gpusPerNode).toLocaleString()} GPUs`,
          ),
        );
        fabricLinks.push({ id: rackId, nodeCount: rackNodes });
        rackX += RACK_W + RACK_GAP;
      }
    }

    if (shape === "multi_node" && numNodes > 1) {
      const fabricX = totalWidth / 2 - 75;
      nodes.push(hubNode("fabric", fabricX, ROW_Y_FABRIC, fabricName, "#f59e0b"));
      for (const link of fabricLinks) {
        const bandwidth = fabricBandwidthGbps * link.nodeCount;
        edges.push({
          id: `e-fabric-${link.id}`,
          source: "fabric",
          target: link.id,
          label: link.nodeCount > 1 ? `${bandwidth.toLocaleString()} Gb/s aggregate` : `${bandwidth} Gb/s`,
          style: { stroke: "#f59e0b" },
          labelStyle: { fontSize: 10 },
        });
      }
    }

    return { nodes, edges };
  }, [shape, gpu, gpusPerNode, numNodes, fabricName, fabricBandwidthGbps]);

  return (
    <div ref={containerRef} className="h-72 w-full rounded-lg border border-hairline bg-surface-card">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          minZoom={0.05}
          nodesDraggable={false}
          nodesConnectable={false}
          panOnDrag
          zoomOnScroll={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} />
          <FitViewOnChange nodes={nodes} containerRef={containerRef} />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}
