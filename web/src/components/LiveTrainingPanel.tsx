"use client";

import { useEffect, useRef, useState } from "react";
import { fetchK8sAvailable, fetchRun, fetchRuns, launchK8sJob, simulateRun } from "@/lib/api";
import { GpuSpec, ModelShape, RunDetail, RunSummary, TopologyRequest } from "@/lib/types";
import { formatCompact } from "@/lib/format";
import { trainingStepPowerWatts } from "@/lib/simEngine";
import { Card, Field, NumberInput, Select, Stat } from "./ui";
import { Sparkline } from "./Sparkline";

const LIST_POLL_MS = 3000;
const DETAIL_POLL_MS = 1000;
const UTILIZATION = 0.35;

export function LiveTrainingPanel({
  model,
  modelLabel,
  topology,
  precision,
  tokensPerStep,
  gpu,
}: {
  model: ModelShape;
  modelLabel: string;
  topology: TopologyRequest;
  precision: string;
  tokensPerStep: number;
  gpu: GpuSpec | undefined;
}) {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const userPicked = useRef(false);

  const [totalSteps, setTotalSteps] = useState(50);
  const [speedup, setSpeedup] = useState(20);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const [k8sAvailable, setK8sAvailable] = useState(false);
  const [launchingK8s, setLaunchingK8s] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  useEffect(() => {
    fetchK8sAvailable()
      .then((r) => setK8sAvailable(r.available))
      .catch(() => setK8sAvailable(false));
  }, []);

  // Poll the run list; auto-select the newest run until the user picks one manually.
  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      fetchRuns()
        .then((list) => {
          if (cancelled) return;
          setRuns(list);
          setListError(null);
          if (!userPicked.current && list.length > 0) {
            setSelectedId((prev) => (prev === list[0].run_id ? prev : list[0].run_id));
          }
        })
        .catch((e) => !cancelled && setListError(e.message));
    };
    tick();
    const id = setInterval(tick, LIST_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Poll the selected run's detail while it's still running.
  useEffect(() => {
    if (!selectedId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clearing stale detail when selection is cleared
      setDetail(null);
      return;
    }
    let cancelled = false;
    let id: ReturnType<typeof setInterval> | null = null;

    const tick = () => {
      fetchRun(selectedId)
        .then((d) => {
          if (cancelled) return;
          setDetail(d);
          if (d.status === "done" && id) {
            clearInterval(id);
            id = null;
          }
        })
        .catch(() => {
          /* run may have been evicted by the store's cap; leave last-known detail visible */
        });
    };
    tick();
    id = setInterval(tick, DETAIL_POLL_MS);
    return () => {
      cancelled = true;
      if (id) clearInterval(id);
    };
  }, [selectedId]);

  const startNewRun = () => {
    setStarting(true);
    setStartError(null);
    simulateRun({
      model,
      model_label: modelLabel,
      topology,
      precision,
      tokens_per_step: tokensPerStep,
      total_steps: totalSteps,
      speedup,
      utilization: 0.35,
    })
      .then((run) => {
        userPicked.current = true;
        setSelectedId(run.run_id);
      })
      .catch((e) => setStartError(e.message))
      .finally(() => setStarting(false));
  };

  const launchRealJob = () => {
    setLaunchingK8s(true);
    setLaunchError(null);
    launchK8sJob({
      model,
      model_label: modelLabel,
      topology,
      precision,
      tokens_per_step: tokensPerStep,
      total_steps: totalSteps,
      speedup,
      utilization: 0.35,
    })
      .then((res) => {
        userPicked.current = true;
        setSelectedId(res.run_id);
      })
      .catch((e) => setLaunchError(e.message))
      .finally(() => setLaunchingK8s(false));
  };

  const meta = detail?.meta;
  const progressPct = meta && detail ? Math.min(100, ((detail.latest_step?.step ?? 0) / meta.total_steps) * 100) : 0;
  const recentSteps = detail ? detail.steps.slice(-8).reverse() : [];

  const last = detail?.steps[detail.steps.length - 1];
  const prev = detail && detail.steps.length > 1 ? detail.steps[detail.steps.length - 2] : undefined;
  const liveTokensPerSec =
    last && prev && last.elapsed_s > prev.elapsed_s
      ? (last.tokens_seen - prev.tokens_seen) / (last.elapsed_s - prev.elapsed_s)
      : null;

  // Power/utilization are constant per run (fixed step-time meta) — a bit of
  // deterministic wiggle per sample keeps the chart from looking like a dead
  // flat line, the way real monitoring graphs never sit perfectly still.
  const wiggle = (i: number) => 1 + 0.04 * Math.sin(i * 1.7);
  const monitorGpu = gpu ?? (meta ? { tdp_watts: 700, idle_watts: 90 } : undefined);
  const powerSamples =
    detail && meta && monitorGpu
      ? detail.steps.map((_, i) =>
          trainingStepPowerWatts(monitorGpu as GpuSpec, meta.compute_s_per_step, meta.communication_s_per_step, UTILIZATION) * wiggle(i),
        )
      : [];
  const utilSamples = detail ? detail.steps.map((_, i) => UTILIZATION * 100 * wiggle(i)) : [];
  const networkBusySamples =
    detail && meta
      ? detail.steps.map((_, i) => (meta.communication_s_per_step / meta.total_s_per_step) * 100 * wiggle(i))
      : [];

  return (
    <Card title="Live training run">
      <div className="mb-4 flex flex-wrap items-end gap-3 border-b border-black/10 dark:border-white/10 pb-4">
        <Field label="Steps">
          <NumberInput value={totalSteps} min={1} max={2000} onChange={setTotalSteps} />
        </Field>
        <Field label="Speedup (x realtime)">
          <NumberInput value={speedup} min={1} max={1000} onChange={setSpeedup} />
        </Field>
        <button
          onClick={startNewRun}
          disabled={starting}
          className="px-4 py-1.5 rounded-md text-sm font-medium bg-blue-500 text-white disabled:opacity-40 hover:bg-blue-600 transition-colors"
        >
          {starting ? "Starting…" : "Start new training run"}
        </button>
        {k8sAvailable && (
          <button
            onClick={launchRealJob}
            disabled={launchingK8s}
            className="px-4 py-1.5 rounded-md text-sm font-medium border border-emerald-500/40 text-emerald-600 dark:text-emerald-400 disabled:opacity-40 hover:bg-emerald-500/10 transition-colors"
            title="Creates a real Kubernetes Job scheduled against the simulated GPU resources — not the fast in-process simulation"
          >
            {launchingK8s ? "Launching…" : "Launch real k8s Job"}
          </button>
        )}
        <span className="text-xs text-black/45 dark:text-white/45">uses the model/GPU/topology selected above</span>
      </div>
      {startError && <p className="text-sm text-red-500 mb-3">{startError}</p>}
      {launchError && <p className="text-sm text-red-500 mb-3">{launchError}</p>}

      {listError && <p className="text-sm text-red-500">{listError}</p>}
      {!listError && runs.length === 0 && (
        <p className="text-sm text-black/45 dark:text-white/45">
          No runs yet — click &quot;Start new training run&quot; above, or deploy the K3s layer and run the sample
          trainer Job (see <code>k3s/helm/simgpu</code>).
        </p>
      )}

      {runs.length > 0 && (
        <div className="mb-4">
          <Select
            value={selectedId}
            onChange={(v) => {
              userPicked.current = true;
              setSelectedId(v);
            }}
          >
            {runs.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.run_id} — {r.meta?.model ?? "?"} on {r.meta?.gpu ?? "?"} ({r.status})
              </option>
            ))}
          </Select>
        </div>
      )}

      {detail && meta && (
        <>
          <div className="flex items-center gap-2 mb-1">
            <span
              className={`h-2 w-2 rounded-full ${detail.status === "running" ? "bg-emerald-500 animate-pulse" : "bg-black/30 dark:bg-white/30"}`}
            />
            <span className="text-sm text-black/60 dark:text-white/60">
              {detail.status === "running" ? "Running" : "Done"} · step {detail.latest_step?.step ?? 0} / {meta.total_steps}
            </span>
          </div>
          <div className="h-2 w-full rounded-full bg-black/10 dark:bg-white/10 overflow-hidden mb-4">
            <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progressPct}%` }} />
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <Stat label="Model / GPU" value={`${meta.model}`} sub={`${meta.gpu} × ${meta.total_gpus}`} />
            <Stat label="Topology" value={meta.topology} />
            <Stat label="Tokens seen" value={formatCompact(detail.latest_step?.tokens_seen ?? 0)} />
            <Stat
              label="Live tokens/sec (sim)"
              value={liveTokensPerSec !== null ? formatCompact(liveTokensPerSec) : "…"}
              sub={`sim step ${(meta.total_s_per_step * 1000).toFixed(0)}ms`}
            />
          </div>

          <div className="text-xs font-mono bg-black/[0.03] dark:bg-white/[0.05] rounded-md p-3 space-y-0.5 max-h-40 overflow-y-auto mb-4">
            {recentSteps.map((s) => (
              <div key={s.step} className="flex justify-between text-black/60 dark:text-white/60">
                <span>step {s.step}</span>
                <span>{formatCompact(s.tokens_seen)} tokens</span>
                <span>{s.elapsed_s.toFixed(2)}s elapsed</span>
              </div>
            ))}
          </div>

          <p className="text-xs uppercase tracking-wide text-black/40 dark:text-white/40 mb-2">GPU monitoring (simulated)</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Sparkline label="Power draw" unit="W" data={powerSamples} color="#f59e0b" max={monitorGpu?.tdp_watts} formatValue={(v) => v.toFixed(0)} />
            <Sparkline label="SM utilization" unit="%" data={utilSamples} color="#3b82f6" max={100} formatValue={(v) => v.toFixed(0)} />
            <Sparkline label="Network busy (NCCL)" unit="%" data={networkBusySamples} color="#ec4899" max={100} formatValue={(v) => v.toFixed(0)} />
          </div>
        </>
      )}
    </Card>
  );
}
