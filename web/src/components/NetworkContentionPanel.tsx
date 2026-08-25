"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchFabrics, simulateNetworkContention } from "@/lib/api";
import { Fabric, NetworkContentionResponse } from "@/lib/types";
import { Card, Field, NumberInput, Select, Stat } from "./ui";

const TEAM_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#14b8a6"];
const ISOLATED_COLOR = "#10b981";
const CONTENDED_COLOR = "#f43f5e";
let nextRowId = 1;

interface JobRow {
  id: number;
  jobId: string;
  team: string;
  numGpus: number;
  payloadGb: number;
}

function makeRow(overrides: Partial<JobRow> = {}): JobRow {
  return { id: nextRowId++, jobId: `job-${nextRowId}`, team: "team-a", numGpus: 8, payloadGb: 1.0, ...overrides };
}

/** Grouped horizontal bars: one row per job, isolated vs. contended
 * communication time — same inline-SVG-bar-chart convention as
 * GpuComparePanel.tsx's local FlopsByPrecisionChart. */
function ContentionBarChart({ jobs, teamColor }: { jobs: NetworkContentionResponse["jobs"]; teamColor: Map<string, string> }) {
  const width = 560;
  const barH = 12;
  const barGap = 3;
  const rowGap = 18;
  const leftPad = 70;
  const rightPad = 70;
  const topPad = 8;

  const maxVal = Math.max(1e-9, ...jobs.flatMap((j) => [j.isolated_comm_s, j.contended_comm_s]));
  const rowHeight = 2 * barH + barGap;
  const height = topPad + jobs.length * (rowHeight + rowGap);
  const plotW = width - leftPad - rightPad;

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        {jobs.map((job, ri) => {
          const y0 = topPad + ri * (rowHeight + rowGap);
          const bars = [
            { label: "isolated", v: job.isolated_comm_s, color: ISOLATED_COLOR },
            { label: "contended", v: job.contended_comm_s, color: CONTENDED_COLOR },
          ];
          return (
            <g key={job.job_id}>
              <text x={leftPad - 8} y={y0 + rowHeight / 2} textAnchor="end" dominantBaseline="middle" className="fill-body text-[11px] font-medium">
                {job.job_id}
              </text>
              <circle cx={12} cy={y0 + rowHeight / 2} r={3} fill={teamColor.get(job.team) ?? TEAM_COLORS[0]} />
              {bars.map((bar, bi) => {
                const w = Math.max(1, (bar.v / maxVal) * plotW);
                const y = y0 + bi * (barH + barGap);
                return (
                  <g key={bar.label}>
                    <rect x={leftPad} y={y} width={w} height={barH} rx={2} fill={bar.color}>
                      <title>
                        {job.job_id} {bar.label}: {(bar.v * 1000).toFixed(1)}ms
                      </title>
                    </rect>
                    <text x={leftPad + w + 6} y={y + barH / 2} dominantBaseline="middle" className="fill-muted-soft text-[9px]">
                      {(bar.v * 1000).toFixed(1)}ms
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}
      </svg>
      <div className="flex items-center justify-center gap-4 mt-1">
        <span className="flex items-center gap-1.5 text-xs text-body">
          <span className="h-2 w-2 rounded-sm" style={{ background: ISOLATED_COLOR }} />
          Isolated (today&apos;s default: full fabric bandwidth)
        </span>
        <span className="flex items-center gap-1.5 text-xs text-body">
          <span className="h-2 w-2 rounded-sm" style={{ background: CONTENDED_COLOR }} />
          Contended (shared fairly by GPU count)
        </span>
      </div>
    </div>
  );
}

export function NetworkContentionPanel() {
  const [fabrics, setFabrics] = useState<Fabric[]>([]);
  const [fabricId, setFabricId] = useState("");
  const [rows, setRows] = useState<JobRow[]>([
    makeRow({ jobId: "team-a-train", team: "team-a", numGpus: 24, payloadGb: 2.0 }),
    makeRow({ jobId: "team-b-finetune", team: "team-b", numGpus: 8, payloadGb: 0.5 }),
  ]);

  const [result, setResult] = useState<NetworkContentionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFabrics()
      .then((list) => {
        setFabrics(list);
        setFabricId((prev) => prev || list[0]?.id || "");
      })
      .catch((e) => setError(e.message));
  }, []);

  const updateRow = (id: number, patch: Partial<JobRow>) => setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const removeRow = (id: number) => setRows((prev) => prev.filter((r) => r.id !== id));
  const addRow = () => setRows((prev) => [...prev, makeRow()]);

  const teamColor = useMemo(() => {
    const teams = Array.from(new Set(rows.map((r) => r.team)));
    const map = new Map<string, string>();
    teams.forEach((t, i) => map.set(t, TEAM_COLORS[i % TEAM_COLORS.length]));
    return map;
  }, [rows]);

  const runSimulation = () => {
    setLoading(true);
    setError(null);
    simulateNetworkContention({
      fabric_id: fabricId,
      jobs: rows.map((r) => ({ job_id: r.jobId, team: r.team, num_gpus: r.numGpus, payload_gb: r.payloadGb })),
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  return (
    <div className="flex flex-col gap-6">
      <Card title="Shared fabric">
        <p className="text-xs text-muted max-w-2xl mb-4">
          Models what happens when multiple concurrent training jobs share one physical interconnect fabric —
          each job&apos;s gradient all-reduce bandwidth gets a fair share proportional to its GPU count, instead of
          the full fabric bandwidth every job gets today when running alone.
        </p>
        <div className="max-w-sm">
          <Field label="Fabric">
            <Select value={fabricId} onChange={setFabricId}>
              {fabrics.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.name} ({f.bandwidth_gbps} GB/s)
                </option>
              ))}
            </Select>
          </Field>
        </div>
      </Card>

      <Card title="Concurrent jobs sharing this fabric">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                <th className="py-2 pr-3">Job ID</th>
                <th className="py-2 pr-3">Team</th>
                <th className="py-2 pr-3">GPUs</th>
                <th className="py-2 pr-3">Payload (GB)</th>
                <th className="py-2 pr-3" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-b border-hairline-soft last:border-0">
                  <td className="py-2 pr-3">
                    <input
                      className="w-28 rounded-md border border-hairline-strong bg-surface-card px-2 py-1 text-sm text-ink outline-none"
                      value={row.jobId}
                      onChange={(e) => updateRow(row.id, { jobId: e.target.value })}
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
                    <NumberInput value={row.numGpus} min={1} onChange={(v) => updateRow(row.id, { numGpus: v })} />
                  </td>
                  <td className="py-2 pr-3 w-24">
                    <NumberInput value={row.payloadGb} min={0.01} step={0.1} onChange={(v) => updateRow(row.id, { payloadGb: v })} />
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
            disabled={loading || !fabricId}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            {loading ? "Simulating…" : "Simulate"}
          </button>
        </div>
      </Card>

      {error && <p className="text-sm text-error">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <Card>
              <Stat label="Fabric bandwidth" value={`${result.fabric_bandwidth_gbps} GB/s`} />
            </Card>
            <Card>
              <Stat label="GPUs sharing fabric" value={String(result.total_gpus_sharing_fabric)} />
            </Card>
            <Card>
              <Stat label="Worst slowdown" value={`${Math.max(...result.jobs.map((j) => j.slowdown_factor)).toFixed(2)}x`} />
            </Card>
          </div>

          <Card title="Isolated vs. contended communication time">
            <ContentionBarChart jobs={result.jobs} teamColor={teamColor} />
          </Card>

          <Card title="Per-job detail">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                    <th className="py-2 pr-3">Job</th>
                    <th className="py-2 pr-3">Team</th>
                    <th className="py-2 pr-3">GPUs</th>
                    <th className="py-2 pr-3">Bandwidth share</th>
                    <th className="py-2 pr-3">Slowdown</th>
                  </tr>
                </thead>
                <tbody>
                  {result.jobs.map((j) => (
                    <tr key={j.job_id} className="border-b border-hairline-soft last:border-0">
                      <td className="py-2 pr-3">{j.job_id}</td>
                      <td className="py-2 pr-3">{j.team}</td>
                      <td className="py-2 pr-3 tabular-nums">{j.num_gpus}</td>
                      <td className="py-2 pr-3 tabular-nums">{j.bandwidth_share_gbps.toFixed(1)} GB/s</td>
                      <td className="py-2 pr-3 tabular-nums">{j.slowdown_factor.toFixed(2)}x</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
