"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchMigGpus, fetchMigProfiles, packMigRequests } from "@/lib/api";
import { MigGpu, MigPackResponse, MigProfile } from "@/lib/types";
import { Card, Field, NumberInput, Select, Stat } from "./ui";
import { MigGrid, MigGpuRow, MigSlice } from "./MigGrid";

const TEAM_COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#f43f5e", "#8b5cf6", "#14b8a6"];
let nextRowId = 1;

interface RequestRow {
  id: number;
  tenant: string;
  profileId: string;
  quantity: number;
}

function makeRow(overrides: Partial<RequestRow> = {}): RequestRow {
  return { id: nextRowId++, tenant: "team-a", profileId: "", quantity: 1, ...overrides };
}

export function MigPlannerPanel() {
  const [gpus, setGpus] = useState<MigGpu[]>([]);
  const [gpuId, setGpuId] = useState("");
  const [poolSize, setPoolSize] = useState(2);
  const [profiles, setProfiles] = useState<MigProfile[]>([]);
  const [rows, setRows] = useState<RequestRow[]>([makeRow({ tenant: "team-a" }), makeRow({ tenant: "team-b" })]);

  const [result, setResult] = useState<MigPackResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMigGpus()
      .then((list) => {
        setGpus(list);
        setGpuId((prev) => prev || list[0]?.gpu_id || "");
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!gpuId) return;
    fetchMigProfiles(gpuId)
      .then((list) => {
        setProfiles(list);
        setRows((prev) => prev.map((r) => (r.profileId ? r : { ...r, profileId: list[0]?.id ?? "" })));
      })
      .catch((e) => setError(e.message));
  }, [gpuId]);

  const updateRow = (id: number, patch: Partial<RequestRow>) => setRows((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const removeRow = (id: number) => setRows((prev) => prev.filter((r) => r.id !== id));
  const addRow = () => setRows((prev) => [...prev, makeRow({ profileId: profiles[0]?.id ?? "" })]);

  const teamColor = useMemo(() => {
    const teams = Array.from(new Set(rows.map((r) => r.tenant)));
    const map = new Map<string, string>();
    teams.forEach((t, i) => map.set(t, TEAM_COLORS[i % TEAM_COLORS.length]));
    return map;
  }, [rows]);

  const runPack = () => {
    setLoading(true);
    setError(null);
    const requests = rows.flatMap((row) =>
      Array.from({ length: row.quantity }, (_, i) => ({
        request_id: `${row.id}-${i}`,
        tenant: row.tenant,
        profile_id: row.profileId,
      })),
    );
    packMigRequests({ gpu_id: gpuId, pool_size: poolSize, requests })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const gridRows: MigGpuRow[] = useMemo(() => {
    if (!result) return [];
    const byGpu = new Map<number, MigSlice[]>();
    for (const p of result.placements) {
      const slice: MigSlice = { tenant: p.tenant, profileId: p.profile_id, color: teamColor.get(p.tenant) ?? TEAM_COLORS[0], memorySlots: p.memory_slots };
      const existing = byGpu.get(p.gpu_index) ?? [];
      existing.push(slice);
      byGpu.set(p.gpu_index, existing);
    }
    return Array.from(byGpu.entries())
      .sort(([a], [b]) => a - b)
      .map(([gpuIndex, slices]) => ({ gpuIndex, slices }));
  }, [result, teamColor]);

  const idleGpus = result ? result.pool_size - result.gpus_used : 0;

  return (
    <div className="flex flex-col gap-6">
      <Card title="MIG-capable GPU pool">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="col-span-2">
            <Field label="GPU type">
              <Select value={gpuId} onChange={setGpuId}>
                {gpus.map((g) => (
                  <option key={g.gpu_id} value={g.gpu_id}>
                    {g.gpu_name}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Pool size (physical GPUs)">
            <NumberInput value={poolSize} min={1} onChange={setPoolSize} />
          </Field>
        </div>
      </Card>

      <Card title="MIG slice requests">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                <th className="py-2 pr-3">Tenant</th>
                <th className="py-2 pr-3">Profile</th>
                <th className="py-2 pr-3">Quantity</th>
                <th className="py-2 pr-3" />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-b border-hairline-soft last:border-0">
                  <td className="py-2 pr-3">
                    <span className="inline-flex items-center gap-1.5">
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ background: teamColor.get(row.tenant) }} />
                      <input
                        className="w-24 rounded-md border border-hairline-strong bg-surface-card px-2 py-1 text-sm text-ink outline-none"
                        value={row.tenant}
                        onChange={(e) => updateRow(row.id, { tenant: e.target.value })}
                      />
                    </span>
                  </td>
                  <td className="py-2 pr-3 w-40">
                    <Select value={row.profileId} onChange={(v) => updateRow(row.id, { profileId: v })}>
                      {profiles.map((p) => (
                        <option key={p.id} value={p.id}>
                          {p.id}
                        </option>
                      ))}
                    </Select>
                  </td>
                  <td className="py-2 pr-3 w-20">
                    <NumberInput value={row.quantity} min={1} onChange={(v) => updateRow(row.id, { quantity: v })} />
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
            + Add request
          </button>
          <button
            onClick={runPack}
            disabled={loading || !gpuId}
            className="inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            {loading ? "Packing…" : "Pack"}
          </button>
        </div>
      </Card>

      {error && <p className="text-sm text-error">{error}</p>}

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <Stat label="GPUs used" value={`${result.gpus_used} / ${result.pool_size}`} />
            </Card>
            <Card>
              <Stat label="Compute utilization" value={`${result.compute_utilization_pct.toFixed(1)}%`} />
            </Card>
            <Card>
              <Stat label="Memory utilization" value={`${result.memory_utilization_pct.toFixed(1)}%`} />
            </Card>
            <Card>
              <Stat label="Unplaced requests" value={String(result.unplaced.length)} />
            </Card>
          </div>

          <Card title="Slice placement">
            <MigGrid rows={gridRows} />
            {idleGpus > 0 && <p className="text-xs text-muted-soft mt-2">+{idleGpus} idle GPU{idleGpus > 1 ? "s" : ""} in the pool, unused</p>}
            <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-hairline">
              {Array.from(teamColor.entries()).map(([team, color]) => (
                <span key={team} className="flex items-center gap-1.5 text-xs text-body">
                  <span className="h-2 w-2 rounded-sm" style={{ background: color }} />
                  {team}
                </span>
              ))}
            </div>
          </Card>

          {result.unplaced.length > 0 && (
            <Card title="Unplaced requests">
              <p className="text-xs text-error mb-2">These didn&apos;t fit anywhere in the pool — add more GPUs or a bigger pool size.</p>
              <ul className="text-sm text-body space-y-1">
                {result.unplaced.map((r) => (
                  <li key={r.request_id}>
                    {r.tenant} — {r.profile_id}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
