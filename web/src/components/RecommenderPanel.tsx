"use client";

import { useEffect, useState } from "react";
import { calculateRecommend } from "@/lib/api";
import { formatCompact, formatUsd } from "@/lib/format";
import { MODEL_PRESETS, RecommendationCandidate } from "@/lib/types";
import { Card, Field, NumberInput, Select, Toggle } from "./ui";

export function RecommenderPanel() {
  const [presetId, setPresetId] = useState(MODEL_PRESETS.find((p) => p.id === "llama2-7b")?.id ?? MODEL_PRESETS[0].id);
  const [precision, setPrecision] = useState("bf16");
  const [totalTrainingTokensB, setTotalTrainingTokensB] = useState(100);
  const [objective, setObjective] = useState<"cost" | "time">("cost");
  const [budgetEnabled, setBudgetEnabled] = useState(false);
  const [budgetUsd, setBudgetUsd] = useState(50000);
  const [maxGpus, setMaxGpus] = useState(64);

  const [candidates, setCandidates] = useState<RecommendationCandidate[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const preset = MODEL_PRESETS.find((p) => p.id === presetId) ?? MODEL_PRESETS[0];

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for an outbound fetch, not derived state
    setLoading(true);
    setError(null);
    calculateRecommend({
      model: preset,
      precision,
      tokens_per_step: 32768,
      total_training_tokens: totalTrainingTokensB * 1e9,
      objective,
      batch_size: 4,
      seq_len: 2048,
      utilization: 0.35,
      num_microbatches: 1,
      max_gpus: maxGpus,
      budget_usd: budgetEnabled ? budgetUsd : null,
      max_time_hours: null,
      candidate_gpu_ids: null,
    })
      .then((res) => setCandidates(res.candidates))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [preset, precision, totalTrainingTokensB, objective, budgetEnabled, budgetUsd, maxGpus]);

  return (
    <div className="flex flex-col gap-6">
      <Card title="What-if recommender">
        <p className="text-xs text-muted max-w-2xl mb-4">
          Pick a model and a target training run, and this searches GPU type x GPU count x tensor/pipeline-parallel
          degree for feasible configurations, ranked by {objective === "cost" ? "lowest total cost" : "fastest total time"}.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="col-span-2">
            <Field label="Model">
              <Select value={presetId} onChange={setPresetId}>
                {MODEL_PRESETS.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </Select>
            </Field>
          </div>
          <Field label="Precision">
            <Select value={precision} onChange={setPrecision}>
              <option value="fp32">fp32</option>
              <option value="bf16">bf16</option>
              <option value="fp16">fp16</option>
              <option value="fp8">fp8</option>
            </Select>
          </Field>
          <Field label="Objective">
            <Select value={objective} onChange={(v) => setObjective(v as "cost" | "time")}>
              <option value="cost">Minimize cost</option>
              <option value="time">Minimize time</option>
            </Select>
          </Field>
          <Field label="Total training tokens (B)">
            <NumberInput value={totalTrainingTokensB} min={1} onChange={setTotalTrainingTokensB} />
          </Field>
          <Field label="Max GPUs to consider">
            <NumberInput value={maxGpus} min={1} max={1024} onChange={setMaxGpus} />
          </Field>
          <div className="flex items-end">
            <Toggle checked={budgetEnabled} onChange={setBudgetEnabled} label="Cap by budget" />
          </div>
          {budgetEnabled && (
            <Field label="Budget (USD)">
              <NumberInput value={budgetUsd} min={0} step={1000} onChange={setBudgetUsd} />
            </Field>
          )}
        </div>
      </Card>

      <Card title={`Ranked candidates${candidates ? ` (${candidates.length})` : ""}`}>
        {loading && <p className="text-sm text-muted">Searching…</p>}
        {error && <p className="text-sm text-error">{error}</p>}
        {!loading && !error && candidates && candidates.length === 0 && (
          <p className="text-sm text-muted">No feasible configuration found under these constraints — try relaxing the budget or max GPU count.</p>
        )}
        {!loading && !error && candidates && candidates.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wider text-muted border-b border-hairline">
                  <th className="py-2 pr-3">GPU</th>
                  <th className="py-2 pr-3">GPUs</th>
                  <th className="py-2 pr-3">TP x PP</th>
                  <th className="py-2 pr-3">Total cost</th>
                  <th className="py-2 pr-3">Total time</th>
                  <th className="py-2 pr-3">$/1k tokens</th>
                  <th className="py-2 pr-3">VRAM headroom</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c, i) => (
                  <tr key={`${c.gpu_id}-${c.num_gpus}-${c.tp_degree}-${c.pp_degree}`} className={`border-b border-hairline-soft last:border-0 ${i === 0 ? "text-success font-semibold" : "text-body-strong"}`}>
                    <td className="py-2 pr-3">{c.gpu_name}</td>
                    <td className="py-2 pr-3 tabular-nums">{c.num_gpus}</td>
                    <td className="py-2 pr-3 tabular-nums">{c.tp_degree} x {c.pp_degree}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatUsd(c.total_cost_usd)}</td>
                    <td className="py-2 pr-3 tabular-nums">{formatCompact(c.total_time_hours)} hr</td>
                    <td className="py-2 pr-3 tabular-nums">{formatUsd(c.cost_per_1k_tokens_usd)}</td>
                    <td className="py-2 pr-3 tabular-nums">{c.vram_headroom_gb.toFixed(1)} GB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
