"use client";

import { useEffect, useState } from "react";
import { calculateCost } from "@/lib/api";
import { ModelShape, TopologyRequest } from "@/lib/types";
import { Card } from "./ui";
import { LineChart, LineSeries } from "./LineChart";

const TP_DEGREES = [1, 2, 4, 8];

export function ParallelismCurvePanel({
  model,
  topology,
  precision,
  tokensPerStep,
  utilization,
  batchSize,
  seqLen,
  numMicrobatches,
  ppDegree,
}: {
  model: ModelShape;
  topology: TopologyRequest;
  precision: string;
  tokensPerStep: number;
  utilization: number;
  batchSize: number;
  seqLen: number;
  numMicrobatches: number;
  ppDegree: number;
}) {
  const [series, setSeries] = useState<LineSeries[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- loading flag for outbound sweep fetches, not derived state
    setLoading(true);
    setError(null);
    Promise.all(
      TP_DEGREES.map((tp) =>
        calculateCost({
          model,
          topology,
          tokens_per_step: tokensPerStep,
          total_training_tokens: 1e11,
          precision,
          utilization,
          tp_degree: tp,
          pp_degree: ppDegree,
          batch_size: batchSize,
          seq_len: seqLen,
          num_microbatches: numMicrobatches,
        })
          .then((r) => ({ tp, r, ok: true as const }))
          .catch(() => ({ tp, r: null, ok: false as const })),
      ),
    )
      .then((results) => {
        const ok = results.filter((x) => x.ok && x.r);
        if (ok.length === 0) {
          setError("Tensor parallelism needs NVLink — this GPU/topology can't run any TP degree above 1.");
          setSeries(null);
          return;
        }
        setSeries([
          { label: "Compute", color: "#3b82f6", points: ok.map(({ tp, r }) => ({ x: tp, y: r!.compute_s_per_step * 1000 })) },
          { label: "TP comm (NVLink)", color: "#ec4899", points: ok.map(({ tp, r }) => ({ x: tp, y: r!.tp_communication_s_per_step * 1000 })) },
          { label: "Total step time", color: "#10b981", points: ok.map(({ tp, r }) => ({ x: tp, y: r!.total_s_per_step * 1000 })) },
        ]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [model, topology, precision, tokensPerStep, utilization, batchSize, seqLen, numMicrobatches, ppDegree]);

  return (
    <Card title="Step time vs. tensor parallel degree">
      <p className="text-xs text-black/45 dark:text-white/45 mb-4">
        Sweeping TP degree at the current batch/sequence length: compute time drops ~linearly as matmuls split
        across more GPUs, but the per-layer NVLink activation all-reduce grows — the crossover is where adding more
        TP stops paying off.
      </p>
      {error && <p className="text-sm text-red-500">{error}</p>}
      {series && !error && (
        <LineChart
          series={series}
          xLabel="TP degree"
          yLabel="ms/step"
          formatX={(v) => `${v}`}
          formatY={(v) => v.toFixed(0)}
        />
      )}
      {loading && <p className="text-sm text-black/45 dark:text-white/45 mt-2">Sweeping…</p>}
    </Card>
  );
}
