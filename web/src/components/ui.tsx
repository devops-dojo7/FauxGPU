"use client";

import { ReactNode, useState } from "react";
import { ModelPresetTier, ModelShape } from "@/lib/types";
import { isMoe, usesMla } from "@/lib/simEngine";
import { formatCompact } from "@/lib/format";

export function Card({ title, children, className = "" }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-hairline bg-surface-card p-5 ${className}`}>
      {title && <h2 className="text-xs font-semibold uppercase tracking-wider text-muted mb-4">{title}</h2>}
      {children}
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-body">{label}</span>
      {children}
    </label>
  );
}

const controlClass =
  "w-full min-w-0 rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-sm text-ink outline-none transition-colors focus:border-ink focus:border-2 focus:px-[11px] focus:py-[7px]";

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
    <label className="flex items-center gap-2 text-sm text-body cursor-pointer">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="accent-ink" />
      {label}
    </label>
  );
}

export function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted">{label}</div>
      <div className="text-xl font-medium tabular-nums text-ink">{value}</div>
      {sub && <div className="text-xs text-muted">{sub}</div>}
    </div>
  );
}

/** button-primary: near-black ink pill, the brand's one CTA color. */
export function ButtonPrimary({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { className = "", ...rest } = props;
  return (
    <button
      className={`inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-2.5 h-10 transition-colors hover:bg-primary-active disabled:opacity-40 disabled:hover:bg-primary ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

/** button-outline: transparent pill, 1px ink-ish border — the secondary CTA. */
export function ButtonOutline({
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const { className = "", ...rest } = props;
  return (
    <button
      className={`inline-flex items-center gap-2 rounded-full bg-transparent text-ink text-sm font-medium border border-hairline-strong px-5 py-2.5 h-10 transition-colors hover:bg-surface-strong disabled:opacity-40 ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}

export function BadgePill({ children, className = "", title }: { children: ReactNode; className?: string; title?: string }) {
  return (
    <span
      title={title}
      className={`inline-flex items-center text-[11px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded-full bg-surface-strong text-ink ${className}`}
    >
      {children}
    </span>
  );
}

/** Flags a model preset whose specs aren't from a real published source — nothing renders for "open" presets or Custom (tier === null). */
export function ConfidenceBadge({ tier }: { tier: ModelPresetTier | null }) {
  if (tier === null || tier === "open") return null;
  const text = tier === "est" ? "Est., unofficial" : "No public specs";
  return (
    <BadgePill className="normal-case tracking-normal font-medium text-warning border border-warning/40" title="This model's architecture isn't from a real published config — see the preset list for sourcing.">
      ⚠ {text}
    </BadgePill>
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
        <BadgePill key={b} className="normal-case tracking-normal font-medium text-body-strong">
          {b}
        </BadgePill>
      ))}
    </div>
  );
}

/** A terminal-styled, copyable command block. Presentational only — this
 * mirrors scripts/playground-*.sh and the README's k3d walkthrough, it
 * never executes anything itself. Wiring a browser button to run commands
 * on the host would need an unauthenticated local agent process; this app
 * has no auth anywhere, so that's a footgun this component deliberately
 * avoids. */
export function TerminalBlock({ title, lines }: { title: string; lines: string[] }) {
  const [copied, setCopied] = useState(false);
  const text = lines.filter((l) => !l.startsWith("#")).join("\n");

  const copy = () => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      })
      .catch(() => {});
  };

  return (
    <div className="rounded-lg overflow-hidden border border-hairline-strong">
      <div className="flex items-center justify-between px-3 py-1.5 bg-surface-strong">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber-400/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald-400/70" />
          </div>
          <span className="text-xs text-muted font-mono">{title}</span>
        </div>
        <button
          onClick={copy}
          className="text-xs font-medium text-muted hover:text-ink transition-colors"
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="text-xs font-mono p-3 overflow-x-auto bg-[#0b0e14] text-emerald-300 leading-relaxed">
        {lines.map((line, i) => (
          <div key={i} className={line.startsWith("#") ? "text-slate-500" : undefined}>
            {line === "" ? " " : line.startsWith("#") ? line : `$ ${line}`}
          </div>
        ))}
      </pre>
    </div>
  );
}
