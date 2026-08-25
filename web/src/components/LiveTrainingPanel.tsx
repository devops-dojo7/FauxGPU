"use client";

import { useEffect, useRef, useState } from "react";
import { fetchK8sAvailable, fetchRun, fetchRuns, injectFailure, launchK8sJob, simulateRun, stopRun } from "@/lib/api";
import { ChaosKind, GpuSpec, ModelShape, RunDetail, RunSummary, TopologyRequest } from "@/lib/types";
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
  const [checkpointIntervalSteps, setCheckpointIntervalSteps] = useState(0);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);

  const [k8sAvailable, setK8sAvailable] = useState(false);
  const [launchingK8s, setLaunchingK8s] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState<string | null>(null);

  const [chaosKind, setChaosKind] = useState<ChaosKind>("xid_error");
  const [chaosSeverity, setChaosSeverity] = useState(5);
  const [chaosDuration, setChaosDuration] = useState(5);
  const [injecting, setInjecting] = useState(false);
  const [injectError, setInjectError] = useState<string | null>(null);

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
          if (d.status !== "running" && id) {
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
      checkpoint_interval_steps: checkpointIntervalSteps > 0 ? checkpointIntervalSteps : null,
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
      checkpoint_interval_steps: null, // not modeled for real k8s Jobs, in-process simulation only
    })
      .then((res) => {
        userPicked.current = true;
        setSelectedId(res.run_id);
      })
      .catch((e) => setLaunchError(e.message))
      .finally(() => setLaunchingK8s(false));
  };

  const stopSelectedRun = () => {
    if (!selectedId) return;
    setStopping(true);
    setStopError(null);
    stopRun(selectedId)
      .then((run) => setDetail((d) => (d ? { ...d, status: run.status } : d)))
      .catch((e) => setStopError(e.message))
      .finally(() => setStopping(false));
  };

  const injectSelectedFailure = () => {
    if (!selectedId) return;
    setInjecting(true);
    setInjectError(null);
    injectFailure(selectedId, {
      kind: chaosKind,
      severity: chaosSeverity,
      duration_steps: chaosKind === "node_drain" ? null : chaosDuration,
    })
      .catch((e) => setInjectError(e.message))
      .finally(() => setInjecting(false));
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
      <div className="mb-4 flex flex-wrap items-end gap-3 border-b border-hairline pb-4">
        <Field label="Steps">
          <NumberInput value={totalSteps} min={1} max={2000} onChange={setTotalSteps} />
        </Field>
        <Field label="Speedup (x realtime)">
          <NumberInput value={speedup} min={1} max={1000} onChange={setSpeedup} />
        </Field>
        <Field label="Checkpoint every N steps (0 = off)">
          <NumberInput value={checkpointIntervalSteps} min={0} onChange={setCheckpointIntervalSteps} />
        </Field>
        <button
          onClick={startNewRun}
          disabled={starting}
          className="px-4 py-1.5 rounded-full text-sm font-medium bg-primary text-on-primary disabled:opacity-40 hover:bg-primary-active transition-colors"
        >
          {starting ? "Starting…" : "Start new training run"}
        </button>
        {k8sAvailable && (
          <button
            onClick={launchRealJob}
            disabled={launchingK8s}
            className="px-4 py-1.5 rounded-full text-sm font-medium border border-emerald-500/40 text-emerald-600 disabled:opacity-40 hover:bg-emerald-500/10 transition-colors"
            title="Creates a real Kubernetes Job scheduled against the simulated GPU resources — not the fast in-process simulation"
          >
            {launchingK8s ? "Launching…" : "Launch real k8s Job"}
          </button>
        )}
        <span className="text-xs text-muted">uses the model/GPU/topology selected above</span>
      </div>
      {startError && <p className="text-sm text-error mb-3">{startError}</p>}
      {launchError && <p className="text-sm text-error mb-3">{launchError}</p>}

      {listError && <p className="text-sm text-error">{listError}</p>}
      {!listError && runs.length === 0 && (
        <p className="text-sm text-muted">
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
              className={`h-2 w-2 rounded-full ${detail.status === "running" ? "bg-emerald-500 animate-pulse" : "bg-muted-soft"}`}
            />
            <span className="text-sm text-body">
              {detail.status === "running" ? "Running" : detail.status === "stopped" ? "Stopped" : "Done"} · step{" "}
              {detail.latest_step?.step ?? 0} / {meta.total_steps}
            </span>
            {detail.latest_step?.active_gpus != null && detail.latest_step.active_gpus < meta.total_gpus && (
              <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-error/10 text-error border border-error/30">
                {detail.latest_step.active_gpus}/{meta.total_gpus} GPUs active
              </span>
            )}
            {detail.status === "running" && (
              <button
                onClick={stopSelectedRun}
                disabled={stopping}
                className="ml-auto px-3 py-1 rounded-full text-xs font-medium border border-error/40 text-error disabled:opacity-40 hover:bg-error/10 transition-colors"
              >
                {stopping ? "Stopping…" : "Stop run"}
              </button>
            )}
          </div>
          {stopError && <p className="text-sm text-error mb-2">{stopError}</p>}

          {detail.status === "running" && (
            <div className="flex flex-wrap items-end gap-2 mb-3 p-3 rounded-lg border border-hairline bg-surface-strong">
              <Field label="Inject failure">
                <Select
                  value={chaosKind}
                  onChange={(v) => {
                    const kind = v as ChaosKind;
                    setChaosKind(kind);
                    setChaosSeverity(kind === "xid_error" ? 5 : kind === "nvlink_degradation" ? 0.2 : 1);
                    setChaosDuration(kind === "nvlink_degradation" ? 10 : 5);
                  }}
                >
                  <option value="xid_error">GPU Xid error (transient stall)</option>
                  <option value="nvlink_degradation">NVLink degradation</option>
                  <option value="node_drain">Node drain (permanent)</option>
                </Select>
              </Field>
              {chaosKind === "xid_error" && (
                <Field label="Stall multiplier">
                  <NumberInput value={chaosSeverity} min={1.1} step={0.5} onChange={setChaosSeverity} />
                </Field>
              )}
              {chaosKind === "nvlink_degradation" && (
                <Field label="Bandwidth remaining (0-1)">
                  <NumberInput value={chaosSeverity} min={0.01} max={0.99} step={0.05} onChange={setChaosSeverity} />
                </Field>
              )}
              {chaosKind === "node_drain" && (
                <Field label="Nodes to drain">
                  <NumberInput value={chaosSeverity} min={1} step={1} onChange={setChaosSeverity} />
                </Field>
              )}
              {chaosKind !== "node_drain" && (
                <Field label="Duration (steps)">
                  <NumberInput value={chaosDuration} min={1} onChange={setChaosDuration} />
                </Field>
              )}
              <button
                onClick={injectSelectedFailure}
                disabled={injecting}
                className="px-3 py-1.5 rounded-full text-xs font-medium border border-hairline-strong disabled:opacity-40 hover:bg-surface-card transition-colors"
              >
                {injecting ? "Injecting…" : "Inject"}
              </button>
              {injectError && <p className="text-xs text-error w-full">{injectError}</p>}
            </div>
          )}

          <div className="h-2 w-full rounded-full bg-hairline overflow-hidden mb-1 relative">
            <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progressPct}%` }} />
          </div>
          {detail.events.length > 0 && (
            <div className="flex items-center gap-1.5 mb-4">
              {detail.events.map((e) => (
                <span
                  key={e.event_id}
                  className={`h-2.5 w-2.5 rounded-full ${
                    e.kind === "xid_error" ? "bg-error" : e.kind === "nvlink_degradation" ? "bg-amber-500" : "bg-rose-600"
                  }`}
                  title={`${e.kind} — severity ${e.severity} — injected at step ${e.injected_at_step}${
                    e.duration_steps ? `, lasts ${e.duration_steps} steps` : " (permanent)"
                  }`}
                />
              ))}
              <span className="text-xs text-muted-soft ml-1">chaos events (hover for details)</span>
            </div>
          )}
          {detail.events.length === 0 && <div className="mb-1" />}
          {detail.checkpoint_events.length > 0 && (
            <div className="flex items-center gap-1.5 mb-4">
              {detail.checkpoint_events.map((e) => (
                <span
                  key={e.event_id}
                  className={`h-2.5 w-2.5 rounded-sm ${e.kind === "save" ? "bg-sky-500" : "bg-purple-500"}`}
                  title={
                    e.kind === "save"
                      ? `checkpoint saved at step ${e.step} — ${e.size_gb.toFixed(1)} GB, ${e.overhead_s.toFixed(1)}s write`
                      : `restored from checkpoint at step ${e.step} — ${e.overhead_s.toFixed(1)}s recovery, ${e.steps_lost} step(s) lost`
                  }
                />
              ))}
              <span className="text-xs text-muted-soft ml-1">checkpoint save/restore (hover for details)</span>
            </div>
          )}
          {detail.checkpoint_events.length === 0 && <div className="mb-4" />}

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

          <div className="text-xs font-mono bg-surface-strong rounded-md p-3 space-y-0.5 max-h-40 overflow-y-auto mb-4">
            {recentSteps.map((s) => (
              <div key={s.step} className="flex justify-between text-body">
                <span>step {s.step}</span>
                <span>{formatCompact(s.tokens_seen)} tokens</span>
                <span>{s.elapsed_s.toFixed(2)}s elapsed</span>
              </div>
            ))}
          </div>

          <p className="text-xs uppercase tracking-wide text-muted-soft mb-2">GPU monitoring (simulated)</p>
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
