"use client";

import { useEffect, useState } from "react";
import { fetchAiModels } from "@/lib/api";
import { AiProvider } from "@/lib/types";
import { Select } from "./ui";

/** Loads the live model list from the provider's own API (so a renamed or
 * deprecated model id, like OpenRouter's 404 on a hardcoded guess, never
 * happens) and renders it as a dropdown. Falls back to a free-text field
 * while loading or if the provider's model-list call fails. */
export function AiModelSelect({
  provider,
  model,
  onModelChange,
}: {
  provider: AiProvider | "";
  model: string;
  onModelChange: (m: string) => void;
}) {
  const [models, setModels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- resetting for a provider switch, not derived state
    setModels([]);
    if (!provider) return;
    setLoading(true);
    setError(null);
    fetchAiModels(provider)
      .then((list) => {
        setModels(list);
        if (list.length > 0) onModelChange(list[0]);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- refetch on provider change only; model/onModelChange are parent-controlled
  }, [provider]);

  if (models.length > 0) {
    return (
      <Select value={model} onChange={onModelChange}>
        {models.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </Select>
    );
  }

  return (
    <input
      value={model}
      onChange={(e) => onModelChange(e.target.value)}
      placeholder={loading ? "Loading models…" : error ? "model id (list unavailable)" : "model id"}
      className="w-full min-w-0 rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-sm text-ink outline-none transition-colors focus:border-ink focus:border-2"
    />
  );
}
