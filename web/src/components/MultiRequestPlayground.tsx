"use client";

import { useEffect, useReducer, useRef, useState } from "react";
import { GpuSpec, ModelShape } from "@/lib/types";
import { checkVramFit, decodeStepMs, estimateTokens, prefillMs } from "@/lib/simEngine";
import { Card, NumberInput, Stat } from "./ui";

interface Lane {
  id: number;
  prompt: string;
  maxOutputTokens: number;
  promptTokens: number;
  status: "idle" | "queued" | "prefilling" | "decoding" | "done";
  generated: number;
  ttftMs: number | null;
}

const MAX_LANES = 6;
const DEFAULT_PROMPTS = [
  "Summarize this support ticket in one sentence.",
  "Write a SQL query to find the top 10 customers by revenue.",
  "Translate this paragraph into French.",
  "What's the time complexity of quicksort?",
];

let nextLaneId = 1;

function makeLane(prompt: string): Lane {
  return { id: nextLaneId++, prompt, maxOutputTokens: 60, promptTokens: 0, status: "idle", generated: 0, ttftMs: null };
}

export function MultiRequestPlayground({ model, gpu, precision }: { model: ModelShape; gpu: GpuSpec | undefined; precision: string }) {
  const [lanes, setLanes] = useState<Lane[]>(() => DEFAULT_PROMPTS.slice(0, 2).map(makeLane));
  const [cacheHitPct, setCacheHitPct] = useState(20);
  const [running, setRunning] = useState(false);
  const [oomError, setOomError] = useState<string | null>(null);
  const [, forceRerender] = useReducer((x) => x + 1, 0);

  const lanesRef = useRef<Lane[]>(lanes);
  const runIdRef = useRef(0);
  const decodeLoopActiveRef = useRef(false);
  const overallStartRef = useRef(0);
  const totalGeneratedAtFirstTokenRef = useRef(0);
  const firstTokenAtRef = useRef(0);
  const [measuredTokensPerSec, setMeasuredTokensPerSec] = useState(0);

  useEffect(() => {
    lanesRef.current = lanes;
  }, [lanes]);

  useEffect(
    () => () => {
      runIdRef.current += 1;
    },
    [],
  ); // cleanup on unmount

  const syncLanes = () => setLanes([...lanesRef.current]);

  const addLane = () => {
    if (lanes.length >= MAX_LANES) return;
    setLanes((prev) => [...prev, makeLane(DEFAULT_PROMPTS[prev.length % DEFAULT_PROMPTS.length])]);
  };
  const removeLane = (id: number) => setLanes((prev) => prev.filter((l) => l.id !== id));
  const updateLane = (id: number, patch: Partial<Lane>) =>
    setLanes((prev) => prev.map((l) => (l.id === id ? { ...l, ...patch } : l)));

  const stop = () => {
    runIdRef.current += 1;
    decodeLoopActiveRef.current = false;
    setRunning(false);
  };

  const runAll = () => {
    if (!gpu || lanes.length === 0) return;

    const withTokens = lanes.map((l) => ({ ...l, promptTokens: estimateTokens(l.prompt) }));
    const totalPeakKvTokens = withTokens.reduce((s, l) => s + l.promptTokens + l.maxOutputTokens, 0);
    const fit = checkVramFit(model, gpu, precision, totalPeakKvTokens);
    if (!fit.fits) {
      setOomError(
        `Out of memory: if all ${lanes.length} requests were decoding at once, KV cache + weights would need ~${fit.requiredGb.toFixed(1)} GB, but ${gpu.name} only has ${fit.capacityGb} GB. Reduce concurrent requests, output length, or pick a bigger GPU.`,
      );
      return;
    }
    setOomError(null);

    runIdRef.current += 1;
    const myRun = runIdRef.current;
    overallStartRef.current = performance.now();
    firstTokenAtRef.current = 0;
    totalGeneratedAtFirstTokenRef.current = 0;
    setMeasuredTokensPerSec(0);

    const reset = withTokens.map((l) => ({ ...l, status: "queued" as const, generated: 0, ttftMs: null }));
    lanesRef.current = reset;
    setLanes(reset);
    setRunning(true);

    const scheduleNextPrefill = (index: number) => {
      if (runIdRef.current !== myRun) return;
      if (index >= lanesRef.current.length) return;
      const lane = lanesRef.current[index];
      lane.status = "prefilling";
      syncLanes();
      const dur = prefillMs(model, gpu, precision, lane.promptTokens, 0.35, cacheHitPct / 100);
      setTimeout(() => {
        if (runIdRef.current !== myRun) return;
        lane.status = "decoding";
        lane.ttftMs = performance.now() - overallStartRef.current;
        syncLanes();
        ensureDecodeLoop();
        scheduleNextPrefill(index + 1);
      }, dur);
    };

    const ensureDecodeLoop = () => {
      if (decodeLoopActiveRef.current) return;
      decodeLoopActiveRef.current = true;
      tick();
    };

    const tick = () => {
      if (runIdRef.current !== myRun) {
        decodeLoopActiveRef.current = false;
        return;
      }
      const active = lanesRef.current.filter((l) => l.status === "decoding");
      const stillPending = lanesRef.current.some((l) => l.status === "queued" || l.status === "prefilling");

      if (active.length === 0) {
        decodeLoopActiveRef.current = false;
        if (!stillPending) {
          setRunning(false);
          forceRerender();
        }
        return;
      }

      if (firstTokenAtRef.current === 0) firstTokenAtRef.current = performance.now();

      const avgKv = active.reduce((s, l) => s + l.promptTokens + l.generated, 0) / active.length;
      const stepMs = decodeStepMs(model, gpu, precision, avgKv, active.length);

      setTimeout(() => {
        if (runIdRef.current !== myRun) {
          decodeLoopActiveRef.current = false;
          return;
        }
        for (const lane of active) {
          if (lane.status !== "decoding") continue;
          lane.generated += 1;
          if (lane.generated >= lane.maxOutputTokens) lane.status = "done";
        }
        syncLanes();
        const totalGenerated = lanesRef.current.reduce((s, l) => s + l.generated, 0);
        const elapsed = (performance.now() - firstTokenAtRef.current) / 1000;
        if (elapsed > 0) setMeasuredTokensPerSec(totalGenerated / elapsed);
        tick();
      }, stepMs);
    };

    scheduleNextPrefill(0);
  };

  const totalGenerated = lanes.reduce((s, l) => s + l.generated, 0);
  const activeDecodingCount = lanes.filter((l) => l.status === "decoding").length;

  return (
    <Card title="Multi-request playground">
      <p className="text-xs text-black/45 dark:text-white/45 mb-4">
        Run several prompts concurrently against one GPU to see continuous batching and prefill queueing in action:
        requests queue for prefill one at a time (later ones wait longer for their first token), then decode together
        in a shared batch — batch size shrinks as requests finish, so survivors speed up.
      </p>

      <div className="space-y-2 mb-3">
        {lanes.map((lane, i) => (
          <div key={lane.id} className="flex items-center gap-2">
            <span className="text-xs text-black/40 dark:text-white/40 w-4">{i + 1}</span>
            <input
              className="flex-1 rounded-md border border-black/15 dark:border-white/15 bg-white dark:bg-black/40 px-2.5 py-1.5 text-sm outline-none focus:border-blue-500"
              value={lane.prompt}
              onChange={(e) => updateLane(lane.id, { prompt: e.target.value })}
              disabled={running}
              placeholder={`Request ${i + 1} prompt…`}
            />
            <div className="w-20">
              <NumberInput value={lane.maxOutputTokens} min={1} max={500} onChange={(v) => updateLane(lane.id, { maxOutputTokens: v })} />
            </div>
            <button
              onClick={() => removeLane(lane.id)}
              disabled={running || lanes.length <= 1}
              className="text-black/40 dark:text-white/40 hover:text-red-500 disabled:opacity-30 px-1"
              title="Remove request"
            >
              ✕
            </button>
            {lane.status !== "idle" && (
              <span
                className={`text-xs w-20 shrink-0 ${
                  lane.status === "decoding"
                    ? "text-emerald-500"
                    : lane.status === "done"
                      ? "text-black/40 dark:text-white/40"
                      : "text-amber-500"
                }`}
              >
                {lane.status}
                {lane.status === "done" ? ` (${lane.generated})` : ""}
              </span>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={addLane}
          disabled={running || lanes.length >= MAX_LANES}
          className="text-xs px-2.5 py-1 rounded-md border border-black/15 dark:border-white/15 hover:bg-black/5 dark:hover:bg-white/5 disabled:opacity-40"
        >
          + Add request
        </button>
        <label className="flex items-center gap-1.5 text-xs text-black/60 dark:text-white/60">
          Cache hit %
          <input
            type="number"
            className="w-14 rounded-md border border-black/15 dark:border-white/15 bg-white dark:bg-black/40 px-1.5 py-1 text-xs"
            value={cacheHitPct}
            min={0}
            max={100}
            onChange={(e) => setCacheHitPct(Number(e.target.value))}
            disabled={running}
          />
        </label>
      </div>

      {oomError && (
        <p className="text-sm text-red-500 mb-4 rounded-lg border border-red-500/25 bg-red-500/[0.06] p-3">⚠ {oomError}</p>
      )}

      <div className="flex gap-2 mb-5">
        <button
          onClick={runAll}
          disabled={!gpu || running}
          className="px-4 py-1.5 rounded-md text-sm font-medium bg-blue-500 text-white disabled:opacity-40 hover:bg-blue-600 transition-colors"
        >
          {running ? "Running…" : `Run ${lanes.length} requests`}
        </button>
        {running && (
          <button
            onClick={stop}
            className="px-4 py-1.5 rounded-md text-sm font-medium border border-black/15 dark:border-white/15 hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
          >
            Stop
          </button>
        )}
      </div>

      {!gpu && <p className="text-sm text-black/45 dark:text-white/45">Pick a GPU above to run the playground.</p>}

      {gpu && lanes.some((l) => l.status !== "idle") && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Stat label="Active batch size" value={`${activeDecodingCount}`} sub="decoding together right now" />
          <Stat label="Tokens generated" value={`${totalGenerated}`} sub="across all requests" />
          <Stat label="Aggregate tokens/sec" value={measuredTokensPerSec > 0 ? measuredTokensPerSec.toFixed(1) : "—"} sub="measured, whole batch" />
          <Stat
            label="TTFT range"
            value={
              lanes.some((l) => l.ttftMs !== null)
                ? `${Math.min(...lanes.filter((l) => l.ttftMs !== null).map((l) => l.ttftMs!)).toFixed(0)}–${Math.max(...lanes.filter((l) => l.ttftMs !== null).map((l) => l.ttftMs!)).toFixed(0)} ms`
                : "…"
            }
            sub="fastest → slowest to first token"
          />
        </div>
      )}
    </Card>
  );
}
