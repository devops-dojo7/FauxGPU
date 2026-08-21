"use client";

import { useEffect, useState } from "react";
import { calculateSpeculativeDecoding } from "@/lib/api";
import { GpuSpec, MODEL_PRESETS, ModelShape, SpeculativeDecodingResponse } from "@/lib/types";
import { Card, Field, NumberInput, Select, Stat } from "./ui";

const DRAFT_CANDIDATES = MODEL_PRESETS.filter((p) => p.params <= 2e9);

export function SpeculativeDecodingPanel({
  targetModel,
  gpu,
  precision,
}: {
  targetModel: ModelShape;
  gpu: GpuSpec | undefined;
  precision: string;
}) {
  const [draftId, setDraftId] = useState(DRAFT_CANDIDATES[1]?.id ?? DRAFT_CANDIDATES[0].id);
  const [gamma, setGamma] = useState(4);
  const [acceptanceRate, setAcceptanceRate] = useState(70);
  const [avgKvTokens, setAvgKvTokens] = useState(1024);

  const [result, setResult] = useState<SpeculativeDecodingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const draftModel = DRAFT_CANDIDATES.find((p) => p.id === draftId) ?? DRAFT_CANDIDATES[0];

  useEffect(() => {
    if (!gpu) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for an outbound fetch, not derived state
    setLoading(true);
    setError(null);
    calculateSpeculativeDecoding({
      draft_model: draftModel,
      target_model: targetModel,
      gpu_id: gpu.id,
      precision,
      gamma,
      acceptance_rate: acceptanceRate / 100,
      avg_kv_tokens: avgKvTokens,
      batch_size: 1,
    })
      .then(setResult)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [draftModel, targetModel, gpu, precision, gamma, acceptanceRate, avgKvTokens]);

  const maxTps = result ? Math.max(result.speculative_tokens_per_sec, result.baseline_tokens_per_sec) : 0;

  return (
    <Card title="Speculative decoding">
      <p className="text-xs text-black/45 dark:text-white/45 mb-4">
        A small <strong>draft</strong> model proposes <code>gamma</code> tokens cheaply and quickly; the big{" "}
        <strong>target</strong> model verifies all of them in one batched pass — about the same cost as decoding a
        single token normally. Accepted tokens are free; a low acceptance rate (draft and target disagree a lot)
        means you paid for draft steps that got thrown away, which can make this <em>slower</em> than plain decoding.
      </p>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <div className="col-span-2">
          <Field label="Draft model">
            <Select value={draftId} onChange={setDraftId}>
              {DRAFT_CANDIDATES.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                </option>
              ))}
            </Select>
          </Field>
        </div>
        <Field label="Gamma (tokens/round)">
          <NumberInput value={gamma} min={1} max={16} onChange={setGamma} />
        </Field>
        <Field label="Acceptance rate %">
          <NumberInput value={acceptanceRate} min={0} max={100} step={5} onChange={setAcceptanceRate} />
        </Field>
        <Field label="Avg KV length (tokens)">
          <NumberInput value={avgKvTokens} min={1} step={128} onChange={setAvgKvTokens} />
        </Field>
      </div>

      <p className="text-xs text-black/45 dark:text-white/45 mb-4">
        Target: {targetModel.params >= 1e9 ? `${(targetModel.params / 1e9).toFixed(1)}B` : `${(targetModel.params / 1e6).toFixed(0)}M`}{" "}
        params · Draft: {(draftModel.params / 1e9).toFixed(2)}B params (
        {(targetModel.params / draftModel.params).toFixed(0)}× smaller)
      </p>

      {error && <p className="text-sm text-red-500">{error}</p>}
      {result && !error && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-5">
            <Stat label="Draft step" value={`${result.draft_step_ms.toFixed(2)} ms`} sub={`× ${gamma} per round`} />
            <Stat label="Verify step" value={`${result.verify_step_ms.toFixed(1)} ms`} sub="target model, batched" />
            <Stat label="Round time" value={`${result.round_ms.toFixed(1)} ms`} sub={`~${result.expected_tokens_per_round.toFixed(2)} tokens/round`} />
            <Stat
              label="Speedup"
              value={`${result.speedup.toFixed(2)}×`}
              sub={result.speedup >= 1 ? "faster than plain decoding" : "slower — acceptance too low"}
            />
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex justify-between text-xs text-black/60 dark:text-white/60 mb-1">
                <span>Baseline (plain autoregressive decoding)</span>
                <span className="tabular-nums">{result.baseline_tokens_per_sec.toFixed(1)} tok/s</span>
              </div>
              <div className="h-5 w-full rounded-md bg-black/[0.06] dark:bg-white/[0.06] overflow-hidden">
                <div
                  className="h-full bg-black/30 dark:bg-white/30 transition-all"
                  style={{ width: `${maxTps > 0 ? Math.max(2, (result.baseline_tokens_per_sec / maxTps) * 100) : 0}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs text-black/60 dark:text-white/60 mb-1">
                <span>Speculative decoding</span>
                <span className="tabular-nums">{result.speculative_tokens_per_sec.toFixed(1)} tok/s</span>
              </div>
              <div className="h-5 w-full rounded-md bg-black/[0.06] dark:bg-white/[0.06] overflow-hidden">
                <div
                  className={`h-full transition-all ${result.speedup >= 1 ? "bg-emerald-500" : "bg-red-500"}`}
                  style={{ width: `${maxTps > 0 ? Math.max(2, (result.speculative_tokens_per_sec / maxTps) * 100) : 0}%` }}
                />
              </div>
            </div>
          </div>
        </>
      )}
      {loading && <p className="text-sm text-black/45 dark:text-white/45 mt-2">Calculating…</p>}
    </Card>
  );
}
