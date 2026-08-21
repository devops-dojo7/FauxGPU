import type {
  CostRequest,
  CostResponse,
  Fabric,
  GpuSpec,
  InferenceRequest,
  InferenceResponse,
  K8sAvailability,
  LaunchK8sJobResponse,
  RunDetail,
  RunSummary,
  SimulateRunRequest,
  SpeculativeDecodingRequest,
  SpeculativeDecodingResponse,
  TopologyRequest,
  TopologyResponse,
  VramRequest,
  VramResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function post<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request to ${path} failed (${res.status})`);
  }
  return res.json();
}

async function get<TRes>(path: string): Promise<TRes> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`Request to ${path} failed (${res.status})`);
  return res.json();
}

export const fetchGpus = () => get<GpuSpec[]>("/gpus");
export const fetchFabrics = () => get<Fabric[]>("/gpus/fabrics");
export const calculateVram = (req: VramRequest) => post<VramRequest, VramResponse>("/calculate/vram", req);
export const calculateCost = (req: CostRequest) => post<CostRequest, CostResponse>("/calculate/cost", req);
export const resolveTopology = (req: TopologyRequest) =>
  post<TopologyRequest, TopologyResponse>("/topology", req);
export const calculateInference = (req: InferenceRequest) =>
  post<InferenceRequest, InferenceResponse>("/calculate/inference", req);
export const fetchRuns = () => get<RunSummary[]>("/runs");
export const fetchRun = (runId: string) => get<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
export const simulateRun = (req: SimulateRunRequest) => post<SimulateRunRequest, RunSummary>("/runs/simulate", req);
export const fetchK8sAvailable = () => get<K8sAvailability>("/runs/k8s-available");
export const launchK8sJob = (req: SimulateRunRequest) =>
  post<SimulateRunRequest, LaunchK8sJobResponse>("/runs/launch-k8s-job", req);
export const calculateSpeculativeDecoding = (req: SpeculativeDecodingRequest) =>
  post<SpeculativeDecodingRequest, SpeculativeDecodingResponse>("/calculate/speculative-decoding", req);
