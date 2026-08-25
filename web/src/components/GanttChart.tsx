"use client";

export interface GanttRow {
  jobId: string;
  team: string;
  color: string;
  segments: { start: number; end: number }[];
  finalStatus: "completed" | "incomplete" | "never_started";
}

/** Scheduling timeline: one horizontal row per job, bars colored by team.
 * Multiple segments in a row (from preemption) are drawn as separate bars
 * with a dashed connector across the gap. Jobs that never got GPU time
 * render as a greyed-out label with no bars. */
export function GanttChart({ rows, makespan }: { rows: GanttRow[]; makespan: number }) {
  const width = 640;
  const rowH = 20;
  const rowGap = 6;
  const leftPad = 140;
  const rightPad = 12;
  const topPad = 8;
  const bottomPad = 22;

  const plotW = width - leftPad - rightPad;
  const chartH = rows.length * (rowH + rowGap) - rowGap;
  const height = topPad + chartH + bottomPad;

  const maxT = makespan > 0 ? makespan * 1.05 : 1;
  const scaleX = (t: number) => (t / maxT) * plotW;

  if (rows.length === 0) {
    return <div className="text-sm text-muted">No jobs to schedule.</div>;
  }

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        <g transform={`translate(${leftPad},${topPad})`}>
          {[0, 0.5, 1].map((t) => (
            <line key={t} x1={plotW * t} x2={plotW * t} y1={0} y2={chartH} className="stroke-hairline-strong" strokeWidth={1} />
          ))}
          {rows.map((row, i) => {
            const y = i * (rowH + rowGap);
            if (row.finalStatus === "never_started") {
              return (
                <g key={row.jobId}>
                  <text x={-8} y={y + rowH / 2} textAnchor="end" dominantBaseline="middle" className="fill-muted-soft text-[10px]">
                    {row.jobId}
                  </text>
                  <text x={4} y={y + rowH / 2} dominantBaseline="middle" className="fill-muted-soft text-[10px] italic">
                    queued — never started
                  </text>
                </g>
              );
            }
            return (
              <g key={row.jobId}>
                <text x={-8} y={y + rowH / 2} textAnchor="end" dominantBaseline="middle" className="fill-body text-[10px] font-medium">
                  {row.jobId}
                </text>
                {row.segments.map((seg, i2) => {
                  const prev = row.segments[i2 - 1];
                  const x0 = scaleX(seg.start);
                  const x1 = scaleX(seg.end);
                  return (
                    <g key={i2}>
                      {prev && (
                        <>
                          <line
                            x1={scaleX(prev.end)}
                            x2={x0}
                            y1={y + rowH / 2}
                            y2={y + rowH / 2}
                            strokeDasharray="3,3"
                            className="stroke-muted-soft"
                            strokeWidth={1.5}
                          />
                          <text x={(scaleX(prev.end) + x0) / 2} y={y + rowH / 2 - 4} textAnchor="middle" className="fill-muted-soft text-[9px]">
                            ⏸
                          </text>
                        </>
                      )}
                      <rect x={x0} y={y} width={Math.max(1, x1 - x0)} height={rowH} rx={3} fill={row.color}>
                        <title>
                          {row.jobId} ({row.team}): {seg.start.toFixed(1)} – {seg.end.toFixed(1)}
                        </title>
                      </rect>
                    </g>
                  );
                })}
                {row.finalStatus === "incomplete" && (
                  <text x={scaleX(row.segments[row.segments.length - 1].end) + 4} y={y + rowH / 2} dominantBaseline="middle" className="fill-muted-soft text-[9px] italic">
                    incomplete
                  </text>
                )}
              </g>
            );
          })}
          <text x={0} y={chartH + 16} textAnchor="start" className="fill-muted-soft" fontSize={10}>
            0
          </text>
          <text x={plotW} y={chartH + 16} textAnchor="end" className="fill-muted-soft" fontSize={10}>
            {makespan.toFixed(1)}
          </text>
        </g>
      </svg>
    </div>
  );
}
