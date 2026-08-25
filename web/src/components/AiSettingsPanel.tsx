"use client";

import { useEffect, useState } from "react";
import { deleteAiProviderKey, fetchAiProviders, fetchLangfuseStatus, setAiProviderKey } from "@/lib/api";
import { AiProvider, AiProviderStatus, LangfuseStatus } from "@/lib/types";
import { BadgePill, ButtonOutline } from "./ui";

const PROVIDER_LABELS: Record<AiProvider, string> = {
  anthropic: "Anthropic (Claude)",
  openai: "OpenAI",
  deepseek: "DeepSeek",
  kimi: "Kimi (Moonshot)",
  groq: "Groq",
  nvidia_nim: "NVIDIA NIM",
  openrouter: "OpenRouter",
};

export function AiSettingsPanel({ onClose }: { onClose: () => void }) {
  const [statuses, setStatuses] = useState<AiProviderStatus[]>([]);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [langfuse, setLangfuse] = useState<LangfuseStatus | null>(null);

  const refresh = () => fetchAiProviders().then(setStatuses).catch((e) => setError(e.message));

  useEffect(() => {
    refresh();
    fetchLangfuseStatus()
      .then(setLangfuse)
      .catch(() => {});
  }, []);

  const save = (provider: string) => {
    const apiKey = drafts[provider]?.trim();
    if (!apiKey) return;
    setBusy(provider);
    setError(null);
    setAiProviderKey(provider, apiKey)
      .then(() => {
        setDrafts((d) => ({ ...d, [provider]: "" }));
        return refresh();
      })
      .catch((e) => setError(e.message))
      .finally(() => setBusy(null));
  };

  const clear = (provider: string) => {
    setBusy(provider);
    setError(null);
    deleteAiProviderKey(provider)
      .then(refresh)
      .catch((e) => setError(e.message))
      .finally(() => setBusy(null));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/40 p-6 overflow-y-auto" onClick={onClose}>
      <div
        className="w-full max-w-xl rounded-xl border border-hairline bg-surface-card p-6 mt-16"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">AI provider keys</h2>
          <button onClick={onClose} className="text-muted hover:text-ink text-sm">
            Close
          </button>
        </div>
        <p className={`text-xs text-muted ${langfuse?.tracing_available ? "mb-2" : "mb-5"}`}>
          Keys are stored encrypted on this server (not in your browser) and used for AI requests you trigger
          here — anyone with access to this instance can use configured keys. Bring your own key per provider.
        </p>

        {langfuse?.tracing_available && (
          <p className="text-xs text-muted mb-5">
            Every AI chat, recommend, and trace-generate call is traced —{" "}
            <a
              href={langfuse.public_url}
              target="_blank"
              rel="noreferrer"
              className="text-body-strong underline decoration-hairline-strong hover:text-ink transition-colors"
            >
              view them in Langfuse ↗
            </a>
            .
          </p>
        )}

        {error && <p className="text-sm text-error mb-3">{error}</p>}

        <div className="flex flex-col gap-3">
          {statuses.map((s) => (
            <div key={s.provider} className="flex items-center gap-2">
              <div className="w-40 shrink-0 text-sm text-body">{PROVIDER_LABELS[s.provider] ?? s.provider}</div>
              <BadgePill className={s.configured ? "bg-success/15 text-success" : "text-muted"}>
                {s.configured ? "Configured" : "Not set"}
              </BadgePill>
              <input
                type="password"
                placeholder="API key"
                value={drafts[s.provider] ?? ""}
                onChange={(e) => setDrafts((d) => ({ ...d, [s.provider]: e.target.value }))}
                className="min-w-0 flex-1 rounded-md border border-hairline-strong bg-surface-card px-3 py-1.5 text-sm text-ink outline-none focus:border-ink focus:border-2"
              />
              <ButtonOutline
                className="px-3 py-1.5 h-8 text-xs"
                disabled={busy === s.provider || !drafts[s.provider]?.trim()}
                onClick={() => save(s.provider)}
              >
                Save
              </ButtonOutline>
              <ButtonOutline
                className="px-3 py-1.5 h-8 text-xs"
                disabled={busy === s.provider || !s.configured}
                onClick={() => clear(s.provider)}
              >
                Clear
              </ButtonOutline>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
