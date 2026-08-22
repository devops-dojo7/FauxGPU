"use client";

export interface RadarAxis {
  key: string;
  label: string;
}

export interface RadarSeries {
  id: number;
  label: string;
  color: string;
  /** 0-100 per axis key; missing key = no data point for that axis. */
  values: Record<string, number>;
}

const SIZE = 340;
const CENTER = SIZE / 2;
const RADIUS = SIZE / 2 - 56;
const RINGS = [0.25, 0.5, 0.75, 1];

function pointFor(index: number, count: number, fraction: number) {
  const angle = -Math.PI / 2 + index * ((2 * Math.PI) / count);
  return { x: CENTER + RADIUS * fraction * Math.cos(angle), y: CENTER + RADIUS * fraction * Math.sin(angle) };
}

export function RadarChart({ axes, series }: { axes: RadarAxis[]; series: RadarSeries[] }) {
  const n = axes.length;
  if (n < 3) return null;

  return (
    <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`} className="overflow-visible">
      {RINGS.map((r) => (
        <polygon
          key={r}
          points={axes.map((_, i) => pointFor(i, n, r)).map((p) => `${p.x},${p.y}`).join(" ")}
          fill="none"
          className="stroke-hairline-strong"
          strokeWidth={1}
        />
      ))}

      {axes.map((_, i) => {
        const p = pointFor(i, n, 1);
        return <line key={i} x1={CENTER} y1={CENTER} x2={p.x} y2={p.y} className="stroke-hairline-strong" strokeWidth={1} />;
      })}

      {series.map((s) => {
        const pts = axes.map((a, i) => pointFor(i, n, Math.max(0, Math.min(100, a.key in s.values ? s.values[a.key] : 0)) / 100));
        return (
          <g key={s.id}>
            <polygon points={pts.map((p) => `${p.x},${p.y}`).join(" ")} fill={s.color} fillOpacity={0.12} stroke={s.color} strokeWidth={2} />
            {pts.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={3} fill={s.color} />
            ))}
          </g>
        );
      })}

      {axes.map((a, i) => {
        const p = pointFor(i, n, 1.18);
        const anchor = Math.abs(p.x - CENTER) < 8 ? "middle" : p.x > CENTER ? "start" : "end";
        return (
          <text key={a.key} x={p.x} y={p.y} textAnchor={anchor} dominantBaseline="middle" className="fill-body text-[11px]">
            {a.label}
          </text>
        );
      })}
    </svg>
  );
}
