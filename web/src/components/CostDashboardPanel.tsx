"use client";

import { useEffect, useState } from "react";
import { fetchRunEconomics, fetchRuns } from "@/lib/api";
import { formatUsd } from "@/lib/format";
import { EconomicsResponse, RunSummary } from "@/lib/types";
import { Card, Select, Stat } from "./ui";
import { LineChart } from "./LineChart";
import { Sparkline } from "./Sparkline";

const POLL_MS = 2000;

export function CostDashboardPanel() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [econ, setEcon] = useState<EconomicsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      fetchRuns()
        .then((rs) => {
          if (cancelled) return;
          setRuns(rs);
          setRunId((prev) => (prev && rs.some((r) => r.run_id === prev) ? prev : (rs[0]?.run_id ?? "")));
        })
        .catch(() => {});
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!runId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- clear stale data when the run selection is cleared (e.g. no runs exist)
      setEcon(null);
      return;
    }
    let cancelled = false;
    const poll = () => {
      fetchRunEconomics(runId)
        .then((r) => {
          if (!cancelled) setEcon(r);
        })
        .catch((e) => {
          if (!cancelled) setError(e.message);
        });
    };
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [runId]);

  const selectedRun = runs.find((r) => r.run_id === runId);

  return (
    <div className="flex flex-col gap-6">
      <Card title="Cost & utilization dashboard">
        <p className="text-xs text-muted max-w-2xl mb-4">
          Cost accumulated and GPU utilization over a training run&apos;s simulated timeline — start a run from the
          Training tab&apos;s live panel, then track it here.
        </p>
        {runs.length === 0 ? (
          <p className="text-sm text-muted">No runs yet. Start one from the Training tab.</p>
        ) : (
          <div className="max-w-sm">
            <Select value={runId} onChange={setRunId}>
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_id} — {r.meta?.model ?? "?"} on {r.meta?.gpu ?? "?"} x {r.meta?.total_gpus ?? "?"} ({r.status})
                </option>
              ))}
            </Select>
          </div>
        )}
      </Card>

      {error && <p className="text-sm text-error">{error}</p>}

      {selectedRun && econ && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <Stat label="Total spend" value={formatUsd(econ.summary.total_cost_usd)} />
            </Card>
            <Card>
              <Stat label="GPU-hours" value={econ.summary.total_gpu_hours.toFixed(3)} />
            </Card>
            <Card>
              <Stat label="Avg utilization" value={`${econ.summary.avg_utilization_pct.toFixed(1)}%`} />
            </Card>
            <Card>
              <Stat label="Idle cost" value={formatUsd(econ.summary.idle_cost_usd)} sub={`${econ.summary.idle_pct.toFixed(0)}% idle`} />
            </Card>
          </div>

          <Card title="Cumulative cost over the run">
            <LineChart
              series={[
                {
                  label: "Cumulative cost",
                  color: "#3b82f6",
                  points: econ.points.map((p) => ({ x: p.step, y: p.cumulative_cost_usd })),
                },
              ]}
              xLabel="step"
              yLabel="cost"
              formatX={(v) => `step ${v.toFixed(0)}`}
              formatY={(v) => formatUsd(v)}
            />
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Sparkline
              label="Current $/hr (cluster)"
              data={econ.points.map((p) => p.cost_usd)}
              color="#3b82f6"
              formatValue={(v) => formatUsd(v)}
            />
            <Sparkline
              label="Utilization"
              unit="%"
              data={econ.points.map((p) => p.utilization_pct)}
              color="#10b981"
              max={100}
            />
          </div>
        </>
      )}
    </div>
  );
}
