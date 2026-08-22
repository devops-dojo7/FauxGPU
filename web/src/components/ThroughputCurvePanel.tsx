"use client";

import { useEffect, useState } from "react";
import { calculateInference } from "@/lib/api";
import { GpuSpec, ModelShape } from "@/lib/types";
import { Card } from "./ui";
import { LineChart, LineSeries } from "./LineChart";

const REQUEST_RATES = [0.5, 1, 2, 4, 6, 8, 12, 16, 20, 25];

export function ThroughputCurvePanel({
  model,
  gpu,
  precision,
  promptTokens,
  outputTokens,
  decodeBatchSize,
  cacheHitFraction,
}: {
  model: ModelShape;
  gpu: GpuSpec | undefined;
  precision: string;
  promptTokens: number;
  outputTokens: number;
  decodeBatchSize: number;
  cacheHitFraction: number;
}) {
  const [series, setSeries] = useState<LineSeries[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!gpu) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for outbound sweep fetches, not derived state
    setLoading(true);
    setError(null);
    Promise.all(
      REQUEST_RATES.map((rps) =>
        calculateInference({
          model,
          gpu_id: gpu.id,
          precision,
          prompt_tokens: promptTokens,
          output_tokens: outputTokens,
          decode_batch_size: decodeBatchSize,
          requests_per_sec: rps,
          cache_hit_fraction: cacheHitFraction,
          utilization: 0.35,
          paged_attention: true,
          block_size: 16,
          gpu_memory_utilization: 0.9,
        }).then((r) => ({ rps, r })),
      ),
    )
      .then((results) => {
        setSeries([
          {
            label: "Colocated",
            color: "#f59e0b",
            points: results.map(({ rps, r }) => ({ x: rps, y: r.colocated_tokens_per_sec_per_gpu })),
          },
          {
            label: "Disaggregated (llm-d)",
            color: "#10b981",
            points: results.map(({ rps, r }) => ({ x: rps, y: r.disaggregated_tokens_per_sec_per_gpu })),
          },
        ]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [model, gpu, precision, promptTokens, outputTokens, decodeBatchSize, cacheHitFraction]);

  return (
    <Card title="Throughput vs. request rate">
      <p className="text-xs text-muted mb-4">
        Sweeping incoming requests/sec at the current prompt/output length and batch size — the colocated curve
        bends down as prefill bursts eat more of the GPU&apos;s time; disaggregated stays flat because prefill runs
        on separate GPUs entirely.
      </p>
      {error && <p className="text-sm text-error">{error}</p>}
      {!gpu && <p className="text-sm text-muted">Pick a GPU above.</p>}
      {series && !error && (
        <LineChart
          series={series}
          xLabel="requests/sec"
          yLabel="tokens/sec/GPU"
          formatX={(v) => v.toFixed(0)}
          formatY={(v) => v.toFixed(0)}
        />
      )}
      {loading && <p className="text-sm text-muted mt-2">Sweeping…</p>}
    </Card>
  );
}
