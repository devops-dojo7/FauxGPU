"use client";

import { ReactNode } from "react";
import { ModelShape } from "@/lib/types";
import { isMoe, usesMla } from "@/lib/simEngine";
import { formatCompact } from "@/lib/format";

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-black/10 dark:border-white/10 bg-black/[0.02] dark:bg-white/[0.03] p-5 ${className}`}>
      {title && <h2 className="text-sm font-semibold uppercase tracking-wide text-black/50 dark:text-white/50 mb-4">{title}</h2>}
      {children}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-black/60 dark:text-white/60">{label}</span>
      {children}
    </label>
  );
}

const controlClass =
  "rounded-md border border-black/15 dark:border-white/15 bg-white dark:bg-black/40 px-2.5 py-1.5 text-sm outline-none focus:border-blue-500";

export function Select({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  children: ReactNode;
}) {
  return (
    <select className={controlClass} value={value} onChange={(e) => onChange(e.target.value)}>
      {children}
    </select>
  );
}

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
}: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      className={controlClass}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="flex items-center gap-2 text-sm text-black/70 dark:text-white/70 cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-blue-600" />
      {label}
    </label>
  );
}

export function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-black/45 dark:text-white/45">{label}</div>
      <div className="text-xl font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-xs text-black/45 dark:text-white/45">{sub}</div>}
    </div>
  );
}

export function ModelArchBadges({ model }: { model: ModelShape }) {
  const badges: string[] = [];
  if (isMoe(model)) {
    badges.push(`MoE — ${formatCompact(model.active_params!)}/${formatCompact(model.params)} active/total params`);
  } else {
    badges.push("Dense");
  }
  if (usesMla(model)) {
    badges.push(`MLA — ${model.kv_latent_dim} latent dim/layer`);
  } else if (model.num_kv_heads != null && model.num_kv_heads < model.num_heads) {
    badges.push(`GQA — ${model.num_kv_heads} of ${model.num_heads} KV heads`);
  } else {
    badges.push("MHA");
  }
  return (
    <div className="flex flex-wrap gap-2">
      {badges.map((b) => (
        <span key={b} className="text-xs px-2 py-1 rounded-md bg-violet-500/10 border border-violet-500/25 text-violet-600 dark:text-violet-400">
          {b}
        </span>
      ))}
    </div>
  );
}
