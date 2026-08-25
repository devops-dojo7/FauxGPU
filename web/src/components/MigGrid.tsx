"use client";

export interface MigSlice {
  tenant: string;
  profileId: string;
  color: string;
  memorySlots: number;
}

export interface MigGpuRow {
  gpuIndex: number;
  slices: MigSlice[];
}

/** MIG bin-packing grid: one row per physical GPU actually used, each row a
 * fixed 1x8 memory-slot bar subdivided into colored segments per placement,
 * proportional to how many of the 8 memory slots that slice consumes. */
export function MigGrid({ rows, memorySlotsTotal = 8 }: { rows: MigGpuRow[]; memorySlotsTotal?: number }) {
  const width = 640;
  const rowH = 32;
  const rowGap = 10;
  const leftPad = 70;
  const rightPad = 12;
  const topPad = 8;
  const bottomPad = 4;

  const plotW = width - leftPad - rightPad;
  const cellW = plotW / memorySlotsTotal;
  const chartH = rows.length * (rowH + rowGap) - rowGap;
  const height = topPad + chartH + bottomPad;

  if (rows.length === 0) {
    return <div className="text-sm text-muted">No GPUs used yet — add requests and click Pack.</div>;
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
      <g transform={`translate(${leftPad},${topPad})`}>
        {rows.map((row, i) => {
          const y = i * (rowH + rowGap);
          let x = 0;
          return (
            <g key={row.gpuIndex}>
              <text x={-8} y={y + rowH / 2} textAnchor="end" dominantBaseline="middle" className="fill-body text-[11px] font-medium">
                GPU {row.gpuIndex}
              </text>
              <rect x={0} y={y} width={plotW} height={rowH} className="fill-surface-strong" />
              {row.slices.map((slice, si) => {
                const w = slice.memorySlots * cellW;
                const rectX = x;
                x += w;
                return (
                  <rect key={si} x={rectX} y={y} width={w} height={rowH} fill={slice.color}>
                    <title>
                      {slice.profileId} — {slice.tenant} ({slice.memorySlots}/{memorySlotsTotal} memory slots)
                    </title>
                  </rect>
                );
              })}
              {Array.from({ length: memorySlotsTotal + 1 }).map((_, gi) => (
                <line key={gi} x1={gi * cellW} x2={gi * cellW} y1={y} y2={y + rowH} className="stroke-hairline-strong" strokeWidth={1} />
              ))}
            </g>
          );
        })}
      </g>
    </svg>
  );
}
