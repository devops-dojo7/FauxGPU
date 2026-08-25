import type {
  AiProviderStatus,
  AutoscalingRequest,
  AutoscalingResponse,
  ChatRequest,
  ChaosInjectRequest,
  CostRequest,
  CostResponse,
  EconomicsResponse,
  MigGpu,
  MigPackRequest,
  MigPackResponse,
  MigProfile,
  NetworkContentionRequest,
  NetworkContentionResponse,
  Fabric,
  GpuSpec,
  InferenceRequest,
  InferenceResponse,
  InferenceStreamRequest,
  K8sAvailability,
  LaunchK8sJobResponse,
  RecommendNlRequest,
  RecommendRequest,
  RecommendResponse,
  RunDetail,
  RunSummary,
  SchedulerRequest,
  SchedulerResponse,
  SimulateRunRequest,
  TraceGenerateNlRequest,
  TraceGenerateNlResponse,
  TraceReplayRequest,
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

async function getText(path: string): Promise<string> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`Request to ${path} failed (${res.status})`);
  return res.text();
}

async function put<TReq>(path: string, body: TReq): Promise<void> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail ?? `Request to ${path} failed (${res.status})`);
  }
}

async function del(path: string): Promise<void> {
  const res = await fetch(`${API_URL}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Request to ${path} failed (${res.status})`);
}

export interface SSEEvent {
  event: string;
  data: unknown;
}

/** POST /inference/stream is Server-Sent Events, but native EventSource is
 * GET-only — reads the response body's ReadableStream by hand instead,
 * yielding one parsed { event, data } per "event:\ndata:\n\n" block. Stops
 * when the stream ends or `signal` aborts. */
export async function* streamInference(req: InferenceStreamRequest, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${API_URL}/inference/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Request to /inference/stream failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // sse-starlette (and the SSE spec generally) terminates lines with
    // \r\n and blocks with a blank line — normalize to \n on the
    // accumulated buffer (not per-chunk, in case a \r\n pair straddles
    // a chunk boundary) so both \n\n and \r\n\r\n separators are found.
    buffer = buffer.replace(/\r\n/g, "\n");

    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      let eventName = "message";
      let dataLine = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (dataLine) yield { event: eventName, data: JSON.parse(dataLine) };
    }
  }
}

/** POST /ai/chat/stream is SSE, same "read the body by hand" reasoning and
 * event-block-parsing loop as streamInference above. */
export async function* streamAiChat(req: ChatRequest, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  const res = await fetch(`${API_URL}/ai/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Request to /ai/chat/stream failed (${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = buffer.replace(/\r\n/g, "\n");

    let sepIndex: number;
    while ((sepIndex = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, sepIndex);
      buffer = buffer.slice(sepIndex + 2);

      let eventName = "message";
      let dataLine = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) eventName = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLine += line.slice(5).trim();
      }
      if (dataLine) yield { event: eventName, data: JSON.parse(dataLine) };
    }
  }
}

export const fetchAiProviders = () => get<AiProviderStatus[]>("/ai/providers");
export const setAiProviderKey = (provider: string, apiKey: string) =>
  put<{ api_key: string }>(`/ai/providers/${encodeURIComponent(provider)}/key`, { api_key: apiKey });
export const deleteAiProviderKey = (provider: string) => del(`/ai/providers/${encodeURIComponent(provider)}/key`);
export const recommendNl = (req: RecommendNlRequest) => post<RecommendNlRequest, RecommendResponse>("/ai/recommend/nl", req);
export const generateTraceNl = (req: TraceGenerateNlRequest) =>
  post<TraceGenerateNlRequest, TraceGenerateNlResponse>("/ai/trace/generate", req);

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
export const stopRun = (runId: string) => post<Record<string, never>, RunSummary>(`/runs/${encodeURIComponent(runId)}/stop`, {});
export const injectFailure = (runId: string, req: ChaosInjectRequest) =>
  post<ChaosInjectRequest, RunSummary>(`/runs/${encodeURIComponent(runId)}/inject`, req);
export const calculateSpeculativeDecoding = (req: SpeculativeDecodingRequest) =>
  post<SpeculativeDecodingRequest, SpeculativeDecodingResponse>("/calculate/speculative-decoding", req);
export const calculateRecommend = (req: RecommendRequest) =>
  post<RecommendRequest, RecommendResponse>("/calculate/recommend", req);
export const fetchRunEconomics = (runId: string) => get<EconomicsResponse>(`/runs/${encodeURIComponent(runId)}/economics`);
export const simulateScheduler = (req: SchedulerRequest) =>
  post<SchedulerRequest, SchedulerResponse>("/scheduler/simulate", req);
export const replayTrace = (req: TraceReplayRequest) => post<TraceReplayRequest, SchedulerResponse>("/scheduler/replay-trace", req);
export const fetchSampleTrace = () => getText("/scheduler/sample-trace");
export const fetchMigGpus = () => get<MigGpu[]>("/mig/gpus");
export const fetchMigProfiles = (gpuId: string) => get<MigProfile[]>(`/mig/profiles/${encodeURIComponent(gpuId)}`);
export const packMigRequests = (req: MigPackRequest) => post<MigPackRequest, MigPackResponse>("/mig/pack", req);
export const simulateAutoscaling = (req: AutoscalingRequest) =>
  post<AutoscalingRequest, AutoscalingResponse>("/autoscaling/simulate", req);
export const simulateNetworkContention = (req: NetworkContentionRequest) =>
  post<NetworkContentionRequest, NetworkContentionResponse>("/network/contention", req);
