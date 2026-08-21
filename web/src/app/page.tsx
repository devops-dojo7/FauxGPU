"use client";

import { useEffect, useState } from "react";
import { calculateCost, calculateVram, fetchFabrics, fetchGpus } from "@/lib/api";
import { CostResponse, Fabric, GpuSpec, MODEL_PRESETS, VramResponse } from "@/lib/types";
import { ModelPanel, ModelPanelState } from "@/components/ModelPanel";
import { GpuPicker } from "@/components/GpuPicker";
import { VramPanel } from "@/components/VramPanel";
import { TopologyPanel, TopologyState } from "@/components/TopologyPanel";
import { TopologyDiagram } from "@/components/TopologyDiagram";
import { CostPanel, CostInputsState } from "@/components/CostPanel";
import { LiveTrainingPanel } from "@/components/LiveTrainingPanel";
import { InferencePanel, InferenceInputsState } from "@/components/InferencePanel";
import { LiveInferencePlayground } from "@/components/LiveInferencePlayground";
import { MultiRequestPlayground } from "@/components/MultiRequestPlayground";
import { SpeculativeDecodingPanel } from "@/components/SpeculativeDecodingPanel";
import { ThroughputCurvePanel } from "@/components/ThroughputCurvePanel";
import { ParallelismCurvePanel } from "@/components/ParallelismCurvePanel";
import { decodeConfig, encodeConfig } from "@/lib/shareConfig";

type Tab = "training" | "inference";

interface SharedConfig {
  modelState: ModelPanelState;
  gpuId: string;
  topoState: TopologyState;
  costInputs: CostInputsState;
  inferenceInputs: InferenceInputsState;
}

export default function Home() {
  const [tab, setTab] = useState<Tab>("training");
  const [gpus, setGpus] = useState<GpuSpec[]>([]);
  const [fabrics, setFabrics] = useState<Fabric[]>([]);
  const [gpuId, setGpuId] = useState<string>("h100-sxm");
  const [apiError, setApiError] = useState<string | null>(null);

  const [modelState, setModelState] = useState<ModelPanelState>({
    presetId: MODEL_PRESETS[0].id,
    model: { ...MODEL_PRESETS[0] },
    precision: "bf16",
    batchSize: 4,
    seqLen: 2048,
    optimizer: "adam",
    fp32MasterCopy: true,
    checkpointing: false,
    training: true,
  });

  const [topoState, setTopoState] = useState<TopologyState>({
    shape: "nvlink_node",
    gpusPerNode: 8,
    numNodes: 1,
    fabricId: "infiniband-hdr",
  });

  const [costInputs, setCostInputs] = useState<CostInputsState>({
    tokensPerStep: 32768,
    totalTrainingTokensB: 100,
    utilization: 0.35,
    tpDegree: 1,
    ppDegree: 1,
    batchSize: 4,
    seqLen: 2048,
    numMicrobatches: 1,
  });

  const [inferenceInputs, setInferenceInputs] = useState<InferenceInputsState>({
    precision: "bf16",
    promptTokens: 2048,
    outputTokens: 256,
    decodeBatchSize: 8,
    requestsPerSec: 2,
    cacheHitPct: 30,
    decodeGpus: 1,
    pagedAttention: true,
    gpuMemoryUtilizationPct: 90,
  });

  const [vram, setVram] = useState<VramResponse | null>(null);
  const [vramLoading, setVramLoading] = useState(false);
  const [vramError, setVramError] = useState<string | null>(null);

  const [cost, setCost] = useState<CostResponse | null>(null);
  const [costLoading, setCostLoading] = useState(false);
  const [costError, setCostError] = useState<string | null>(null);

  const [shareCopied, setShareCopied] = useState(false);

  useEffect(() => {
    fetchGpus()
      .then(setGpus)
      .catch((e) => setApiError(e.message));
    fetchFabrics()
      .then(setFabrics)
      .catch((e) => setApiError(e.message));

    const params = new URLSearchParams(window.location.search);
    const encoded = params.get("config");
    if (!encoded) return;
    const shared = decodeConfig<SharedConfig>(encoded);
    if (!shared) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time hydration from a shared link's URL param
    if (shared.modelState) setModelState(shared.modelState);
    if (shared.gpuId) setGpuId(shared.gpuId);
    if (shared.topoState) setTopoState(shared.topoState);
    if (shared.costInputs) setCostInputs(shared.costInputs);
    if (shared.inferenceInputs) setInferenceInputs(shared.inferenceInputs);
  }, []);

  const shareConfig = () => {
    const config: SharedConfig = { modelState, gpuId, topoState, costInputs, inferenceInputs };
    const encoded = encodeConfig(config);
    const url = new URL(window.location.href);
    url.searchParams.set("config", encoded);
    window.history.replaceState(null, "", url.toString());
    navigator.clipboard?.writeText(url.toString()).catch(() => {});
    setShareCopied(true);
    setTimeout(() => setShareCopied(false), 2000);
  };

  const gpu = gpus.find((g) => g.id === gpuId);
  const numGpus = topoState.shape === "single_gpu" ? 1 : topoState.shape === "nvlink_node" ? topoState.gpusPerNode : topoState.gpusPerNode * topoState.numNodes;

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for an outbound fetch, not derived state
    setVramLoading(true);
    setVramError(null);
    calculateVram({
      model: modelState.model,
      precision: modelState.precision,
      batch_size: modelState.batchSize,
      seq_len: modelState.seqLen,
      optimizer: modelState.optimizer,
      fp32_master_copy: modelState.fp32MasterCopy,
      checkpointing: modelState.checkpointing,
      training: modelState.training,
    })
      .then(setVram)
      .catch((e) => setVramError(e.message))
      .finally(() => setVramLoading(false));
  }, [modelState]);

  useEffect(() => {
    if (!gpu) return;
    if (topoState.shape !== "single_gpu" && !gpu.nvlink_gbps) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for an outbound fetch, not derived state
    setCostLoading(true);
    setCostError(null);
    calculateCost({
      model: modelState.model,
      topology: {
        shape: topoState.shape,
        gpu_id: gpuId,
        gpus_per_node: topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode,
        num_nodes: topoState.shape === "multi_node" ? topoState.numNodes : 1,
        fabric_id: topoState.shape === "multi_node" ? topoState.fabricId : null,
      },
      tokens_per_step: costInputs.tokensPerStep,
      total_training_tokens: costInputs.totalTrainingTokensB * 1e9,
      precision: modelState.precision,
      utilization: costInputs.utilization,
      tp_degree: costInputs.tpDegree,
      pp_degree: costInputs.ppDegree,
      batch_size: costInputs.batchSize,
      seq_len: costInputs.seqLen,
      num_microbatches: costInputs.numMicrobatches,
    })
      .then(setCost)
      .catch((e) => setCostError(e.message))
      .finally(() => setCostLoading(false));
  }, [modelState, topoState, costInputs, gpuId, gpu]);

  const fabric = fabrics.find((f) => f.id === topoState.fabricId);

  return (
    <div className="min-h-screen">
      <div className="pointer-events-none fixed inset-x-0 top-0 h-80 bg-gradient-to-b from-blue-500/10 via-transparent to-transparent" />

      <div className="relative max-w-7xl mx-auto p-6 md:p-10">
        <header className="mb-8 flex flex-col gap-5">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-500 to-emerald-500 bg-clip-text text-transparent inline-block">
              GPU Cluster Simulator
            </h1>
            <p className="text-sm text-black/55 dark:text-white/55 mt-1.5 max-w-2xl">
              Learn how GPU training and inference actually work — VRAM, KV cache, NVLink/InfiniBand topology,
              disaggregated serving, and real cost tradeoffs — with no GPU required.
            </p>
            {apiError && (
              <p className="mt-3 text-sm text-red-500">
                Couldn&apos;t reach the API ({apiError}). Is it running at{" "}
                <code>{process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}</code>?
              </p>
            )}
          </div>

          <div className="flex items-center justify-between gap-3 flex-wrap">
            <nav className="inline-flex w-fit rounded-lg border border-black/10 dark:border-white/10 bg-black/[0.02] dark:bg-white/[0.03] p-1 gap-1">
              {(
                [
                  ["training", "Training"],
                  ["inference", "Inference (llm-d)"],
                ] as [Tab, string][]
              ).map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => setTab(id)}
                  className={`px-3.5 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    tab === id
                      ? "bg-blue-500 text-white shadow-sm"
                      : "text-black/60 dark:text-white/60 hover:bg-black/5 dark:hover:bg-white/5"
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>

            <button
              onClick={shareConfig}
              className="px-3.5 py-1.5 rounded-md text-sm font-medium border border-black/15 dark:border-white/15 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            >
              {shareCopied ? "Link copied!" : "Share this config"}
            </button>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <ModelPanel state={modelState} onChange={setModelState} />
          <GpuPicker gpus={gpus} selectedId={gpuId} onSelect={setGpuId} />
        </div>

        {tab === "training" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="flex flex-col gap-6">
              <VramPanel vram={vram} gpu={gpu} numGpus={topoState.shape === "single_gpu" ? 1 : numGpus} loading={vramLoading} error={vramError} />
            </div>

            <div className="flex flex-col gap-6">
              <CostPanel
                state={costInputs}
                onChange={setCostInputs}
                cost={cost}
                loading={costLoading}
                error={costError}
                numGpus={numGpus}
                pricePerHr={gpu?.price_per_hr_usd ?? 0}
              />
            </div>

            <div className="lg:col-span-2 flex flex-col gap-6">
              <TopologyPanel state={topoState} onChange={setTopoState} gpu={gpu} fabrics={fabrics} />
              <TopologyDiagram
                shape={topoState.shape}
                gpu={gpu}
                gpusPerNode={topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode}
                numNodes={topoState.shape === "multi_node" ? topoState.numNodes : 1}
                fabricName={fabric?.name ?? "Fabric"}
                fabricBandwidthGbps={fabric?.bandwidth_gbps ?? 0}
              />
              <LiveTrainingPanel
                model={modelState.model}
                modelLabel={modelState.presetId}
                topology={{
                  shape: topoState.shape,
                  gpu_id: gpuId,
                  gpus_per_node: topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode,
                  num_nodes: topoState.shape === "multi_node" ? topoState.numNodes : 1,
                  fabric_id: topoState.shape === "multi_node" ? topoState.fabricId : null,
                }}
                precision={modelState.precision}
                tokensPerStep={costInputs.tokensPerStep}
                gpu={gpu}
              />
              <ParallelismCurvePanel
                model={modelState.model}
                topology={{
                  shape: topoState.shape,
                  gpu_id: gpuId,
                  gpus_per_node: topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode,
                  num_nodes: topoState.shape === "multi_node" ? topoState.numNodes : 1,
                  fabric_id: topoState.shape === "multi_node" ? topoState.fabricId : null,
                }}
                precision={modelState.precision}
                tokensPerStep={costInputs.tokensPerStep}
                utilization={costInputs.utilization}
                batchSize={costInputs.batchSize}
                seqLen={costInputs.seqLen}
                numMicrobatches={costInputs.numMicrobatches}
                ppDegree={costInputs.ppDegree}
              />
            </div>
          </div>
        )}

        {tab === "inference" && (
          <div className="flex flex-col gap-6">
            <InferencePanel model={modelState.model} gpu={gpu} state={inferenceInputs} onChange={setInferenceInputs} />
            <ThroughputCurvePanel
              model={modelState.model}
              gpu={gpu}
              precision={inferenceInputs.precision}
              promptTokens={inferenceInputs.promptTokens}
              outputTokens={inferenceInputs.outputTokens}
              decodeBatchSize={inferenceInputs.decodeBatchSize}
              cacheHitFraction={inferenceInputs.cacheHitPct / 100}
            />
            <LiveInferencePlayground
              model={modelState.model}
              modelLabel={modelState.presetId}
              gpu={gpu}
              precision={inferenceInputs.precision}
            />
            <MultiRequestPlayground
              model={modelState.model}
              modelLabel={modelState.presetId}
              gpu={gpu}
              precision={inferenceInputs.precision}
            />
            <SpeculativeDecodingPanel targetModel={modelState.model} gpu={gpu} precision={inferenceInputs.precision} />
          </div>
        )}
      </div>
    </div>
  );
}
