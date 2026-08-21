"use client";

export interface LineSeries {
  label: string;
  color: string;
  points: { x: number; y: number }[];
}

/** Small multi-series XY line chart (SVG) with axis min/max labels and a
 * legend — for sweeps (e.g. a parameter varied across several values and
 * the resulting metric plotted as a curve), not live time-series. */
export function LineChart({
  series,
  xLabel,
  yLabel,
  formatX,
  formatY,
  height = 200,
}: {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  height?: number;
}) {
  const width = 560;
  const padding = { top: 10, right: 10, bottom: 28, left: 48 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const allPoints = series.flatMap((s) => s.points);
  if (allPoints.length === 0) {
    return <div className="text-sm text-black/45 dark:text-white/45">No data.</div>;
  }
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = 0;
  const yMax = Math.max(...ys) * 1.1 || 1;

  const scaleX = (x: number) => (xMax > xMin ? ((x - xMin) / (xMax - xMin)) * plotW : plotW / 2);
  const scaleY = (y: number) => plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  const fx = formatX ?? ((v: number) => v.toString());
  const fy = formatY ?? ((v: number) => v.toString());

  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height }}>
        <g transform={`translate(${padding.left},${padding.top})`}>
          {[0, 0.5, 1].map((t) => (
            <line key={t} x1={0} x2={plotW} y1={plotH * t} y2={plotH * t} className="stroke-black/10 dark:stroke-white/10" strokeWidth={1} />
          ))}
          {series.map((s) => {
            const path = s.points
              .slice()
              .sort((a, b) => a.x - b.x)
              .map((p, i) => `${i === 0 ? "M" : "L"}${scaleX(p.x).toFixed(1)},${scaleY(p.y).toFixed(1)}`)
              .join(" ");
            return <path key={s.label} d={path} fill="none" stroke={s.color} strokeWidth={2} />;
          })}
          <text x={-8} y={4} textAnchor="end" className="fill-black/40 dark:fill-white/40" fontSize={10}>
            {fy(yMax)}
          </text>
          <text x={-8} y={plotH} textAnchor="end" className="fill-black/40 dark:fill-white/40" fontSize={10}>
            {fy(0)}
          </text>
          <text x={0} y={plotH + 18} textAnchor="start" className="fill-black/40 dark:fill-white/40" fontSize={10}>
            {fx(xMin)}
          </text>
          <text x={plotW} y={plotH + 18} textAnchor="end" className="fill-black/40 dark:fill-white/40" fontSize={10}>
            {fx(xMax)}
          </text>
        </g>
      </svg>
      <div className="flex items-center justify-between mt-1">
        <div className="flex flex-wrap gap-3">
          {series.map((s) => (
            <span key={s.label} className="flex items-center gap-1.5 text-xs text-black/60 dark:text-white/60">
              <span className="h-2 w-2 rounded-sm" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
        <span className="text-xs text-black/40 dark:text-white/40">
          {xLabel} → {yLabel}
        </span>
      </div>
    </div>
  );
}
