"use client";

import { useEffect, useRef, useState } from "react";
import { GpuSpec, ModelShape } from "@/lib/types";
import {
  checkVramFit,
  decodePowerWatts,
  estimateTokens,
  kvCacheBytesPerToken,
  prefillPowerWatts,
  DECODE_POWER_FRACTION,
} from "@/lib/simEngine";
import { streamInference } from "@/lib/api";
import { formatGb } from "@/lib/format";
import { Card, Field, NumberInput, Stat } from "./ui";
import { Sparkline } from "./Sparkline";

const FILLER_WORDS =
  "the model considers each token in context, attending to prior state before predicting the next most likely continuation of the sequence based on patterns learned during training across a broad corpus of text data and code"
    .split(" ");

type Status = "idle" | "prefill" | "decoding" | "done";

export function LiveInferencePlayground({
  model,
  modelLabel,
  gpu,
  precision,
  tpDegree,
}: {
  model: ModelShape;
  modelLabel?: string;
  gpu: GpuSpec | undefined;
  precision: string;
  tpDegree: number;
}) {
  const [prompt, setPrompt] = useState("Explain how KV cache works in transformer inference.");
  const [maxOutputTokens, setMaxOutputTokens] = useState(80);
  const [cacheHitPct, setCacheHitPct] = useState(0);

  const [status, setStatus] = useState<Status>("idle");
  const [ttftMs, setTtftMs] = useState<number | null>(null);
  const [generatedTokens, setGeneratedTokens] = useState(0);
  const [promptTokens, setPromptTokens] = useState(0);
  const [measuredTokensPerSec, setMeasuredTokensPerSec] = useState(0);
  const [outputWords, setOutputWords] = useState<string[]>([]);
  const [oomError, setOomError] = useState<string | null>(null);
  const [powerSamples, setPowerSamples] = useState<number[]>([]);
  const [memBusySamples, setMemBusySamples] = useState<number[]>([]);

  const abortRef = useRef<AbortController | null>(null);
  const firstTokenAtRef = useRef(0);

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
  };

  useEffect(() => () => stop(), []); // cleanup on unmount

  const run = async () => {
    if (!gpu) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const pTokens = estimateTokens(prompt);

    const fit = checkVramFit(model, gpu, precision, pTokens + maxOutputTokens, tpDegree);
    if (!fit.fits) {
      setOomError(
        `Out of memory: weights + KV cache for this prompt/output length need ~${fit.requiredGb.toFixed(1)} GB${tpDegree > 1 ? ` per GPU (${tpDegree}-way split)` : ""}, but ${gpu.name} only has ${fit.capacityGb} GB. Reduce output tokens, prompt length, pick a bigger GPU, or increase tensor-parallel GPUs.`,
      );
      return;
    }
    setOomError(null);

    setPromptTokens(pTokens);
    setGeneratedTokens(0);
    setOutputWords([]);
    setMeasuredTokensPerSec(0);
    setTtftMs(null);
    setStatus("prefill");
    setPowerSamples([prefillPowerWatts(gpu, 0.35)]);
    setMemBusySamples([30]);

    try {
      for await (const evt of streamInference(
        {
          model,
          model_label: modelLabel ?? "custom",
          gpu_id: gpu.id,
          precision,
          prompt,
          prompt_tokens: pTokens,
          max_output_tokens: maxOutputTokens,
          cache_hit_fraction: cacheHitPct / 100,
          utilization: 0.35,
          tp_degree: tpDegree,
        },
        controller.signal,
      )) {
        if (controller.signal.aborted) return;
        if (evt.event === "ttft") {
          setTtftMs((evt.data as { ttft_s: number }).ttft_s * 1000);
          setStatus("decoding");
          firstTokenAtRef.current = performance.now();
        } else if (evt.event === "token") {
          const next = (evt.data as { index: number }).index;
          setGeneratedTokens(next);
          setOutputWords((prev) => [...prev, FILLER_WORDS[(prev.length + pTokens) % FILLER_WORDS.length]]);
          const elapsedSinceFirst = (performance.now() - firstTokenAtRef.current) / 1000;
          if (elapsedSinceFirst > 0) setMeasuredTokensPerSec(next / elapsedSinceFirst);
          setPowerSamples((prev) => [...prev.slice(-59), decodePowerWatts(gpu)]);
          setMemBusySamples((prev) => [...prev.slice(-59), DECODE_POWER_FRACTION * 100 + 30]);
        } else if (evt.event === "done") {
          setStatus("done");
        } else if (evt.event === "error") {
          setOomError((evt.data as { detail: string }).detail);
          setStatus("idle");
        }
      }
    } catch (err) {
      if (!controller.signal.aborted) setOomError(err instanceof Error ? err.message : String(err));
    }
  };

  const kvBytesPerToken = gpu ? kvCacheBytesPerToken(model, precision) : 0;
  const currentKvTokens = promptTokens + generatedTokens;
  const kvCacheGb = (kvBytesPerToken * currentKvTokens) / 1e9;
  const kvCapacityGb = gpu?.vram_gb ?? 0;
  const kvPct = kvCapacityGb > 0 ? Math.min(100, (kvCacheGb / kvCapacityGb) * 100) : 0;

  return (
    <Card title="Live prompt playground">
      <p className="text-xs text-muted mb-4">
        Type a prompt and watch simulated generation — timing comes from the same prefill/decode formulas as the
        panel above, paced token-by-token in real time. Output text is placeholder filler (there&apos;s no real
        model here), but TTFT, tokens/sec, and KV cache growth are the real simulated numbers.
      </p>

      <Field label="Prompt">
        <textarea
          className="rounded-md border border-hairline-strong bg-surface-card text-ink px-2.5 py-1.5 text-sm outline-none focus:border-ink min-h-20 resize-y"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={status === "prefill" || status === "decoding"}
        />
      </Field>

      <div className="grid grid-cols-2 gap-3 mt-3 mb-4">
        <Field label="Max output tokens">
          <NumberInput value={maxOutputTokens} min={1} max={1000} onChange={setMaxOutputTokens} />
        </Field>
        <Field label="Prefix cache hit %">
          <NumberInput value={cacheHitPct} min={0} max={100} step={5} onChange={setCacheHitPct} />
        </Field>
      </div>

      <div className="flex gap-2 mb-5">
        <button
          onClick={run}
          disabled={!gpu || status === "prefill" || status === "decoding"}
          className="px-4 py-1.5 rounded-full text-sm font-medium bg-primary text-on-primary disabled:opacity-40 hover:bg-primary-active transition-colors"
        >
          {status === "prefill" || status === "decoding" ? "Running…" : "Run inference"}
        </button>
        {(status === "prefill" || status === "decoding") && (
          <button
            onClick={stop}
            className="px-4 py-1.5 rounded-full text-sm font-medium border border-hairline-strong hover:bg-surface-strong transition-colors"
          >
            Stop
          </button>
        )}
      </div>

      {!gpu && <p className="text-sm text-muted">Pick a GPU above to run the playground.</p>}
      {oomError && (
        <p className="text-sm text-error mb-4 rounded-lg border border-red-500/25 bg-red-500/[0.06] p-3">
          ⚠ {oomError}
        </p>
      )}

      {gpu && status !== "idle" && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <Stat label="TTFT" value={ttftMs !== null ? `${ttftMs.toFixed(0)} ms` : "…"} sub={status === "prefill" ? "prefilling…" : "time to first token"} />
            <Stat label="Tokens generated" value={`${generatedTokens} / ${maxOutputTokens}`} />
            <Stat label="Live tokens/sec" value={measuredTokensPerSec > 0 ? measuredTokensPerSec.toFixed(1) : "—"} sub="measured, this request" />
            <Stat label="Live KV cache" value={formatGb(kvCacheGb)} sub={`${currentKvTokens} tokens`} />
          </div>

          <div className="mb-4">
            <div className="flex justify-between text-xs text-body mb-1">
              <span>KV cache vs. GPU VRAM</span>
              <span className="tabular-nums">
                {formatGb(kvCacheGb)} / {kvCapacityGb} GB
              </span>
            </div>
            <div className="h-2.5 w-full rounded-full bg-surface-strong overflow-hidden">
              <div className="h-full bg-rose-500 transition-all" style={{ width: `${kvPct}%` }} />
            </div>
          </div>

          <div className="text-sm font-mono bg-surface-strong rounded-md p-3 min-h-16 leading-relaxed text-body mb-4">
            {status === "prefill" && <span className="text-muted-soft">prefilling…</span>}
            {outputWords.join(" ")}
            {status === "decoding" && <span className="animate-pulse">▊</span>}
          </div>

          <p className="text-xs uppercase tracking-wide text-muted-soft mb-2">GPU monitoring (simulated)</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <Sparkline label="Power draw" unit="W" data={powerSamples} color="#f59e0b" max={gpu.tdp_watts} formatValue={(v) => v.toFixed(0)} />
            <Sparkline label="Memory bus busy (HBM)" unit="%" data={memBusySamples} color="#10b981" max={100} formatValue={(v) => v.toFixed(0)} />
          </div>
        </>
      )}
    </Card>
  );
}
