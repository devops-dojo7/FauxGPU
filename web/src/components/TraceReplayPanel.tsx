"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchSampleTrace, replayTrace } from "@/lib/api";
import { SchedulerResponse } from "@/lib/types";
import { Card, Field, NumberInput, Select, Stat, Toggle } from "./ui";
import { GanttChart, GanttRow } from "./GanttChart";

const TEAM_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#14b8a6"];

export function TraceReplayPanel({ gpus }: { gpus: { id: string; name: string }[] }) {
  const [traceCsv, setTraceCsv] = useState("");
  const [gpuId, setGpuId] = useState(gpus[0]?.id ?? "");
  const [totalGpus, setTotalGpus] = useState(64);
  const [preemptionEnabled, setPreemptionEnabled] = useState(true);

  const [result, setResult] = useState<SchedulerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSampleTrace()
      .then(setTraceCsv)
      .catch((e) => setError(e.message));
  }, []);

  const runReplay = () => {
    setLoading(true);
    setError(null);
    replayTrace({ trace_csv: traceCsv, gpu_id: gpuId, total_gpus: totalGpus, preemption_enabled: preemptionEnabled, horizon: null })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const teamColor = useMemo(() => {
    if (!result) return new Map<string, string>();
    const teams = Array.from(new Set(result.jobs.map((j) => j.team)));
    const map = new Map<string, string>();
    teams.forEach((t, i) => map.set(t, TEAM_COLORS[i % TEAM_COLORS.length]));
    return map;
  }, [result]);

  const ganttRows: GanttRow[] = useMemo(() => {
    if (!result) return [];
    return result.jobs
      .slice()
      .sort((a, b) => a.team.localeCompare(b.team) || (a.segments[0]?.start ?? 0) - (b.segments[0]?.start ?? 0))
      .map((j) => ({
        jobId: j.job_id,
        team: j.team,
        color: teamColor.get(j.team) ?? TEAM_COLORS[0],
        segments: j.segments,
        finalStatus: j.final_status,
      }));
  }, [result, teamColor]);

  const neverStartedCount = result?.jobs.filter((j) => j.final_status === "never_started").length ?? 0;
  const preemptionEvents = result?.jobs.reduce((sum, j) => sum + j.preempted_count, 0) ?? 0;

  return (
    <div className="flex flex-col gap-6">
      <Card title="GPU pool">
        <p className="text-xs text-muted max-w-2xl mb-4">
          Replays a job-queue trace through the same multi-tenant scheduler as the Scheduler tab, instead of
          hand-entered jobs. The textarea below is pre-filled with a bundled, illustrative sample trace — paste
          your own in the same <code>job_id,team,priority,gpu_count,submit_time,duration</code> CSV format to try it.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="col-span-2">
            <Field label="GPU type">
              <Select value={gpuId} onChange={setGpuId}>
                {gpus.map((g) => (
                  <option key={g.id} value={g.id}>
                    {g.name}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Total GPUs in pool">
            <NumberInput value={totalGpus} min={1} onChange={setTotalGpus} />
          </Field>
          <div className="flex items-end">
            <Toggle checked={preemptionEnabled} onChange={setPreemptionEnabled} label="Preemption enabled" />
          </div>
        </div>
      </Card>

      <Card title="Trace (CSV)">
        <textarea
          value={traceCsv}
          onChange={(e) => setTraceCsv(e.target.value)}
          rows={10}
          spellCheck={false}
          className="w-full rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-xs font-mono text-ink outline-none focus:border-ink focus:border-2"
        />
        <div className="flex justify-end mt-4">
          <button
            onClick={runReplay}
            disabled={loading || !gpuId || !traceCsv.trim()}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            {loading ? "Replaying…" : "Replay trace"}
          </button>
        </div>
      </Card>

      {error && <p className="text-sm text-error">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <Stat label="Makespan" value={result.makespan.toFixed(1)} />
            </Card>
            <Card>
              <Stat label="GPU utilization" value={`${result.gpu_utilization_pct.toFixed(1)}%`} />
            </Card>
            <Card>
              <Stat label="Preemption events" value={String(preemptionEvents)} />
            </Card>
            <Card>
              <Stat label="Jobs never started" value={String(neverStartedCount)} />
            </Card>
          </div>

          <Card title="Replayed timeline">
            <GanttChart rows={ganttRows} makespan={result.makespan} />
            <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-hairline">
              {Array.from(teamColor.entries()).map(([team, color]) => (
                <span key={team} className="flex items-center gap-1.5 text-xs text-body">
                  <span className="h-2 w-2 rounded-sm" style={{ background: color }} />
                  {team}
                </span>
              ))}
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
