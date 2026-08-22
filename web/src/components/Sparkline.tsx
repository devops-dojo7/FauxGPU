"use client";

/** Small Grafana-style time-series panel: current value + a filled sparkline
 * of recent history. A little deterministic wiggle is added to the drawn
 * line purely for visual "it's alive" realism when the underlying value is
 * constant across samples — the displayed current value is always exact. */
export function Sparkline({
  label,
  unit,
  data,
  color,
  max,
  formatValue,
}: {
  label: string;
  unit?: string;
  data: number[];
  color: string;
  max?: number;
  formatValue?: (v: number) => string;
}) {
  const width = 260;
  const height = 44;
  const current = data.length > 0 ? data[data.length - 1] : 0;
  const visible = data.slice(-40);
  const peak = max ?? Math.max(1, ...visible);

  const points = visible.map((v, i) => {
    const x = visible.length > 1 ? (i / (visible.length - 1)) * width : width;
    const y = height - Math.min(1, v / peak) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const linePath = points.length > 0 ? `M${points.join(" L")}` : "";
  const areaPath = points.length > 0 ? `${linePath} L${width},${height} L0,${height} Z` : "";

  return (
    <div className="rounded-lg border border-hairline bg-surface-strong p-3">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-xs text-muted">{label}</span>
        <span className="text-sm font-semibold tabular-nums" style={{ color }}>
          {formatValue ? formatValue(current) : current.toFixed(1)}
          {unit && <span className="text-xs font-normal text-muted-soft ml-0.5">{unit}</span>}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-11" preserveAspectRatio="none">
        {areaPath && <path d={areaPath} fill={color} opacity={0.15} />}
        {linePath && <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
      </svg>
    </div>
  );
}
