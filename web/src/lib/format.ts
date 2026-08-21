export function formatGb(gb: number): string {
  if (gb < 1) return `${(gb * 1024).toFixed(0)} MB`;
  return `${gb.toFixed(2)} GB`;
}

export function formatUsd(v: number): string {
  if (v < 1) return `$${v.toFixed(6)}`;
  if (v < 1000) return `$${v.toFixed(2)}`;
  return `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

export function formatCompact(n: number): string {
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function formatParams(params: number): string {
  return `${(params / 1e9).toFixed(1)}B params`;
}
