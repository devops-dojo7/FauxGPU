"use client";

import { useEffect, useState } from "react";
import { completeOnK8sServer, fetchInferenceK8sAvailable, launchInferenceK8sServer, stopInferenceK8sServer } from "@/lib/api";
import { K8sCompletionResult } from "@/lib/types";
import { Card, Field, NumberInput, Select, Stat, TerminalBlock } from "./ui";

// k3s/inference-server/server.py only understands these three hardcoded
// presets (its own MODEL_PRESETS dict) — a much smaller list than the
// site's full model catalog, so this gets its own selector rather than
// reusing the shared preset dropdown.
const SERVER_MODEL_PRESETS = ["llama2-7b", "llama2-13b", "llama2-70b"];

export function RealInferenceServerPanel({ gpuId }: { gpuId: string }) {
  const [k8sAvailable, setK8sAvailable] = useState(false);
  const [modelPreset, setModelPreset] = useState(SERVER_MODEL_PRESETS[0]);
  const [precision, setPrecision] = useState("bf16");

  const [serverName, setServerName] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);

  const [prompt, setPrompt] = useState("Explain how KV cache works in transformer inference.");
  const [maxTokens, setMaxTokens] = useState(80);
  const [sending, setSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);
  const [result, setResult] = useState<K8sCompletionResult | null>(null);

  useEffect(() => {
    fetchInferenceK8sAvailable()
      .then((r) => setK8sAvailable(r.available))
      .catch(() => setK8sAvailable(false));
  }, []);

  if (!k8sAvailable) return null;

  const launch = () => {
    setLaunching(true);
    setLaunchError(null);
    launchInferenceK8sServer({ model_preset: modelPreset, gpu_id: gpuId, precision })
      .then((res) => setServerName(res.name))
      .catch((e) => setLaunchError(e.message))
      .finally(() => setLaunching(false));
  };

  const stop = () => {
    if (!serverName) return;
    setStopping(true);
    stopInferenceK8sServer(serverName)
      .then(() => {
        setServerName(null);
        setResult(null);
        setSendError(null);
      })
      .catch((e) => setLaunchError(e.message))
      .finally(() => setStopping(false));
  };

  const sendTestRequest = () => {
    if (!serverName) return;
    setSending(true);
    setSendError(null);
    completeOnK8sServer(serverName, { prompt, max_tokens: maxTokens })
      .then(setResult)
      .catch((e) => setSendError(e.message))
      .finally(() => setSending(false));
  };

  return (
    <Card title="Real inference server (k8s)">
      <p className="text-sm text-muted mb-4">
        Unlike the playgrounds above (simulated timing over an in-process API call), this launches a real,
        long-running Deployment — <code>k3s/inference-server</code>, an OpenAI-Completions-shaped{" "}
        <code>/v1/completions</code> server — against this cluster&apos;s simulated GPU resources. Only 3 hardcoded
        model presets are supported (the real server&apos;s own <code>MODEL_PRESETS</code>), not the full catalog.
      </p>

      {!serverName ? (
        <div className="flex flex-wrap items-end gap-3">
          <Field label="Model preset">
            <Select value={modelPreset} onChange={setModelPreset}>
              {SERVER_MODEL_PRESETS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </Select>
          </Field>
          <Field label="Precision">
            <Select value={precision} onChange={setPrecision}>
              <option value="fp32">fp32</option>
              <option value="bf16">bf16</option>
              <option value="fp16">fp16</option>
              <option value="fp8">fp8</option>
            </Select>
          </Field>
          <button
            onClick={launch}
            disabled={launching}
            className="px-4 py-1.5 rounded-full text-sm font-medium bg-primary text-on-primary disabled:opacity-40 hover:bg-primary-active transition-colors"
          >
            {launching ? "Launching…" : "Launch real inference server"}
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-sm text-body">
            <code>{serverName}</code> deployed
          </span>
          <button
            onClick={stop}
            disabled={stopping}
            className="ml-auto px-3 py-1 rounded-full text-xs font-medium border border-error/40 text-error disabled:opacity-40 hover:bg-error/10 transition-colors"
          >
            {stopping ? "Stopping…" : "Stop server"}
          </button>
        </div>
      )}
      {launchError && <p className="text-sm text-error mt-3">{launchError}</p>}

      {serverName && (
        <>
          <div className="mt-4 mb-4">
            <TerminalBlock
              title="reach it yourself"
              lines={[
                `kubectl port-forward svc/${serverName} 9000:9000`,
                "# in another terminal:",
                `curl localhost:9000/v1/completions -H "Content-Type: application/json" -d '{"prompt": "${prompt.slice(0, 40)}", "max_tokens": ${maxTokens}}'`,
              ]}
            />
          </div>

          <div className="border-t border-hairline pt-4">
            <div className="flex flex-wrap items-end gap-3 mb-3">
              <Field label="Prompt">
                <input
                  className="rounded-md border border-hairline-strong bg-surface-card text-ink px-2.5 py-1.5 text-sm outline-none focus:border-ink w-72"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                />
              </Field>
              <Field label="Max tokens">
                <NumberInput value={maxTokens} min={1} max={500} onChange={setMaxTokens} />
              </Field>
              <button
                onClick={sendTestRequest}
                disabled={sending}
                className="px-4 py-1.5 rounded-full text-sm font-medium border border-hairline-strong disabled:opacity-40 hover:bg-surface-strong transition-colors"
                title="Proxied server-side through simgpu-api, inside the cluster — a real round trip to the pod above"
              >
                {sending ? "Sending…" : "Send test request"}
              </button>
            </div>
            {sendError && <p className="text-sm text-error mb-3">{sendError}</p>}

            {result && (
              <>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                  <Stat label="TTFT" value={`${(result.ttft_s * 1000).toFixed(0)} ms`} />
                  <Stat label="Tokens/sec" value={result.tokens_per_sec.toFixed(0)} />
                  <Stat label="Cost" value={`$${result.cost_usd.toFixed(6)}`} />
                  <Stat label="Round trip" value={`${result.round_trip_ms.toFixed(0)} ms`} sub="proxy → pod → proxy" />
                </div>
                <p className="text-sm text-body bg-surface-strong rounded-md p-3">{result.text}</p>
              </>
            )}
          </div>
        </>
      )}
    </Card>
  );
}
