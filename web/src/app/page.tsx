"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { calculateCost, calculateVram, fetchFabrics, fetchGpus } from "@/lib/api";
import { CostResponse, Fabric, GpuSpec, MODEL_PRESETS, VramResponse } from "@/lib/types";
import { ModelPanel, ModelPanelState, getModelLabel } from "@/components/ModelPanel";
import { ComparePanel } from "@/components/ComparePanel";
import { DatacenterStats } from "@/components/DatacenterStats";
import { PlaygroundPanel } from "@/components/PlaygroundPanel";
import { GpuComparePanel } from "@/components/GpuComparePanel";
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
import { RecommenderPanel } from "@/components/RecommenderPanel";
import { CostDashboardPanel } from "@/components/CostDashboardPanel";
import { SchedulerPanel } from "@/components/SchedulerPanel";
import { MigPlannerPanel } from "@/components/MigPlannerPanel";
import { AutoscalingPanel } from "@/components/AutoscalingPanel";
import { NetworkContentionPanel } from "@/components/NetworkContentionPanel";
import { TraceReplayPanel } from "@/components/TraceReplayPanel";
import { decodeConfig, encodeConfig } from "@/lib/shareConfig";
import { ThemeToggle } from "@/components/ThemeToggle";
import { AiSettingsPanel } from "@/components/AiSettingsPanel";
import { TabNav } from "@/components/TabNav";

type Tab = "training" | "inference" | "datacenter" | "gpu-compare" | "compare" | "playground" | "recommender" | "cost-dashboard" | "scheduler" | "mig-planner" | "autoscaling" | "network-contention" | "trace-replay";

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
  const [aiSettingsOpen, setAiSettingsOpen] = useState(false);

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
    customName: "",
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
    <div className="min-h-screen relative overflow-x-hidden">
      <div className="gradient-orb gradient-orb-sky w-[420px] h-[420px] -top-40 -left-32" />
      <div className="gradient-orb gradient-orb-lavender w-[360px] h-[360px] -top-24 right-0" />

      <div className="relative max-w-7xl mx-auto p-6 md:p-10">
        <header className="mb-8 flex flex-col gap-6">
          <div>
            <div className="flex items-center gap-3">
              <Image src="/logo.png" alt="FauxGPU logo" width={48} height={48} className="h-10 w-10 md:h-12 md:w-12 rounded-xl" priority />
              <h1 className="font-wordmark text-3xl md:text-4xl inline-block tracking-tight uppercase">
                <span className="text-ink">FAUX</span>
                <span style={{ color: "#38e29b", textShadow: "0 0 18px rgba(56, 226, 155, 0.55)" }}>GPU</span>
              </h1>
            </div>
            <p className="text-sm text-body mt-2 max-w-2xl">
              Learn how GPU training and inference actually work — VRAM, KV cache, NVLink/InfiniBand topology,
              disaggregated serving, and real cost tradeoffs — with no GPU required.
            </p>
            {apiError && (
              <p className="mt-3 text-sm text-error">
                Couldn&apos;t reach the API ({apiError}). Is it running at{" "}
                <code>{process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}</code>?
              </p>
            )}
          </div>

          <div className="flex items-center justify-between gap-3 flex-wrap">
            <TabNav
              tabs={
                [
                  ["training", "Training"],
                  ["inference", "Inference (llm-d)"],
                  ["datacenter", "Datacenter"],
                  ["gpu-compare", "Compare GPUs"],
                  ["compare", "Compare Models"],
                  ["playground", "Playground"],
                  ["recommender", "Recommender"],
                  ["cost-dashboard", "Cost Dashboard"],
                  ["scheduler", "Scheduler"],
                  ["mig-planner", "MIG Planner"],
                  ["autoscaling", "Autoscaling"],
                  ["network-contention", "Network Contention"],
                  ["trace-replay", "Trace Replay"],
                ] as [Tab, string][]
              }
              active={tab}
              onSelect={setTab}
            />

            <div className="flex items-center gap-2">
              <button
                onClick={shareConfig}
                className="px-4 py-1.5 rounded-full text-sm font-medium border border-hairline-strong text-ink hover:bg-surface-strong transition-colors"
              >
                {shareCopied ? "Link copied!" : "Share this config"}
              </button>
              <button
                onClick={() => setAiSettingsOpen(true)}
                className="px-4 py-1.5 rounded-full text-sm font-medium border border-hairline-strong text-ink hover:bg-surface-strong transition-colors"
              >
                AI settings
              </button>
              <ThemeToggle />
            </div>
          </div>
        </header>

        {aiSettingsOpen && <AiSettingsPanel onClose={() => setAiSettingsOpen(false)} />}

        {tab === "playground" || tab === "gpu-compare" || tab === "recommender" || tab === "cost-dashboard" || tab === "scheduler" || tab === "mig-planner" || tab === "autoscaling" || tab === "network-contention" || tab === "trace-replay" ? null : tab === "compare" || tab === "datacenter" ? (
          <div className="mb-6 max-w-md">
            <GpuPicker gpus={gpus} selectedId={gpuId} onSelect={setGpuId} />
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <ModelPanel state={modelState} onChange={setModelState} />
            <GpuPicker gpus={gpus} selectedId={gpuId} onSelect={setGpuId} />
          </div>
        )}

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
              <LiveTrainingPanel
                model={modelState.model}
                modelLabel={getModelLabel(modelState)}
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
              modelLabel={getModelLabel(modelState)}
              gpu={gpu}
              precision={inferenceInputs.precision}
            />
            <MultiRequestPlayground
              model={modelState.model}
              modelLabel={getModelLabel(modelState)}
              gpu={gpu}
              precision={inferenceInputs.precision}
            />
            <SpeculativeDecodingPanel targetModel={modelState.model} gpu={gpu} precision={inferenceInputs.precision} />
          </div>
        )}

        {tab === "datacenter" && (
          <div className="flex flex-col gap-6">
            <TopologyPanel state={topoState} onChange={setTopoState} gpu={gpu} gpus={gpus} onSelectGpu={setGpuId} fabrics={fabrics} />
            <TopologyDiagram
              shape={topoState.shape}
              gpu={gpu}
              gpusPerNode={topoState.shape === "single_gpu" ? 1 : topoState.gpusPerNode}
              numNodes={topoState.shape === "multi_node" ? topoState.numNodes : 1}
              fabricName={fabric?.name ?? "Fabric"}
              fabricBandwidthGbps={fabric?.bandwidth_gbps ?? 0}
            />
            <DatacenterStats gpu={gpu} numGpus={numGpus} />
          </div>
        )}

        {tab === "compare" && (
          <ComparePanel
            gpu={gpu}
            gpuId={gpuId}
            topoState={topoState}
            costInputs={costInputs}
            inferenceInputs={inferenceInputs}
            workload={modelState}
            numGpus={numGpus}
          />
        )}

        {tab === "gpu-compare" && <GpuComparePanel gpus={gpus} />}

        {tab === "playground" && <PlaygroundPanel />}

        {tab === "recommender" && <RecommenderPanel />}

        {tab === "cost-dashboard" && <CostDashboardPanel />}

        {tab === "scheduler" && <SchedulerPanel gpus={gpus} />}

        {tab === "mig-planner" && <MigPlannerPanel />}

        {tab === "autoscaling" && <AutoscalingPanel gpus={gpus} />}

        {tab === "network-contention" && <NetworkContentionPanel />}

        {tab === "trace-replay" && <TraceReplayPanel gpus={gpus} />}
      </div>
    </div>
  );
}
