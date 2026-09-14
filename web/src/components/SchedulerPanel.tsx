"use client";

import { useMemo, useState } from "react";
import { simulateScheduler } from "@/lib/api";
import { GpuSpec, SchedJob, SchedulerResponse } from "@/lib/types";
import { groupGpusByVendor } from "@/lib/format";
import { Card, Field, NumberInput, Select, Stat, Toggle } from "./ui";
import { GanttChart, GanttRow } from "./GanttChart";

const TEAM_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#14b8a6"];
let nextRowId = 1;

interface JobRow extends SchedJob {
  id: number;
}

function makeJobRow(overrides: Partial<SchedJob> = {}): JobRow {
  return {
    id: nextRowId++,
    job_id: `job-${nextRowId}`,
    team: "team-a",
    priority: 1,
    gpu_count: 4,
    submit_time: 0,
    duration: 10,
    ...overrides,
  };
}

export function SchedulerPanel({ gpus }: { gpus: GpuSpec[] }) {
  const [rows, setRows] = useState<JobRow[]>(() => [
    makeJobRow({ job_id: "team-a-train", team: "team-a", priority: 5, gpu_count: 8, submit_time: 0, duration: 10 }),
    makeJobRow({ job_id: "team-b-finetune", team: "team-b", priority: 1, gpu_count: 8, submit_time: 2, duration: 5 }),
  ]);
  const [gpuId, setGpuId] = useState(gpus[0]?.id ?? "");
  const [totalGpus, setTotalGpus] = useState(8);
  const [preemptionEnabled, setPreemptionEnabled] = useState(true);

  const [result, setResult] = useState<SchedulerResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateRow = (id: number, patch: Partial<SchedJob>) => setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const removeRow = (id: number) => setRows((prev) => prev.filter((r) => r.id !== id));
  const addRow = () => setRows((prev) => [...prev, makeJobRow({ submit_time: 0 })]);

  const teamColor = useMemo(() => {
    const teams = Array.from(new Set(rows.map((r) => r.team)));
    const map = new Map<string, string>();
    teams.forEach((t, i) => map.set(t, TEAM_COLORS[i % TEAM_COLORS.length]));
    return map;
  }, [rows]);

  const runSimulation = () => {
    setLoading(true);
    setError(null);
    simulateScheduler({
      jobs: rows.map((row): SchedJob => ({
        job_id: row.job_id,
        team: row.team,
        priority: row.priority,
        gpu_count: row.gpu_count,
        submit_time: row.submit_time,
        duration: row.duration,
      })),
      gpu_id: gpuId,
      total_gpus: totalGpus,
      preemption_enabled: preemptionEnabled,
      horizon: null,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="col-span-2">
            <Field label="GPU type">
              <Select value={gpuId} onChange={setGpuId}>
                {groupGpusByVendor(gpus).map(([vendor, list]) => (
                  <optgroup key={vendor} label={vendor}>
                    {list.map((g) => (
                      <option key={g.id} value={g.id}>
                        {g.name}
                      </option>
                    ))}
                  </optgroup>
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

      <Card title="Jobs">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                <th className="py-2 pr-3">Job ID</th>
                <th className="py-2 pr-3">Team</th>
                <th className="py-2 pr-3">Priority</th>
                <th className="py-2 pr-3">GPUs</th>
                <th className="py-2 pr-3">Submit time</th>
                <th className="py-2 pr-3">Duration</th>
                <th className="py-2 pr-3" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-b border-hairline-soft last:border-0">
                  <td className="py-2 pr-3">
                    <input
                      className="w-28 rounded-md border border-hairline-strong bg-surface-card px-2 py-1 text-sm text-ink outline-none"
                      value={row.job_id}
                      onChange={(e) => updateRow(row.id, { job_id: e.target.value })}
                    />
                  </td>
                  <td className="py-2 pr-3">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ background: teamColor.get(row.team) }} />
                      <input
                        className="w-24 rounded-md border border-hairline-strong bg-surface-card px-2 py-1 text-sm text-ink outline-none"
                        value={row.team}
                        onChange={(e) => updateRow(row.id, { team: e.target.value })}
                      />
                    </span>
                  </td>
                  <td className="py-2 pr-3 w-20">
                    <NumberInput value={row.priority} min={0} onChange={(v) => updateRow(row.id, { priority: v })} />
                  </td>
                  <td className="py-2 pr-3 w-20">
                    <NumberInput value={row.gpu_count} min={1} onChange={(v) => updateRow(row.id, { gpu_count: v })} />
                  </td>
                  <td className="py-2 pr-3 w-24">
                    <NumberInput value={row.submit_time} min={0} onChange={(v) => updateRow(row.id, { submit_time: v })} />
                  </td>
                  <td className="py-2 pr-3 w-24">
                    <NumberInput value={row.duration} min={0.1} step={0.5} onChange={(v) => updateRow(row.id, { duration: v })} />
                  </td>
                  <td className="py-2 pr-3">
                    <button onClick={() => removeRow(row.id)} disabled={rows.length <= 1} className="text-muted-soft hover:text-error disabled:opacity-30 px-1">
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between mt-4">
          <button onClick={addRow} className="text-xs px-2.5 py-1 rounded-full border border-hairline-strong hover:bg-surface-strong transition-colors">
            + Add job
          </button>
          <button
            onClick={runSimulation}
            disabled={loading || !gpuId}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            {loading ? "Simulating…" : "Run simulation"}
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

          <Card title="Scheduling timeline">
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
