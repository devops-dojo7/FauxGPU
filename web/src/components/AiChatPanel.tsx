"use client";

import { useEffect, useRef, useState } from "react";
import { fetchAiProviders, streamAiChat } from "@/lib/api";
import { AiProvider, AiProviderStatus, ChatMessage } from "@/lib/types";
import { Select } from "./ui";

const MODEL_PLACEHOLDERS: Record<AiProvider, string> = {
  anthropic: "claude-sonnet-4-5-20250929",
  openai: "gpt-4o",
  deepseek: "deepseek-chat",
  kimi: "moonshot-v1-8k",
  groq: "llama-3.3-70b-versatile",
  nvidia_nim: "meta/llama-3.1-70b-instruct",
  openrouter: "anthropic/claude-3.5-sonnet",
};

export function AiChatPanel() {
  const [open, setOpen] = useState(false);
  const [providers, setProviders] = useState<AiProviderStatus[]>([]);
  const [provider, setProvider] = useState<AiProvider | "">("");
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    fetchAiProviders()
      .then((all) => {
        setProviders(all);
        const configured = all.find((p) => p.configured);
        if (configured && !provider) {
          setProvider(configured.provider);
          setModel(MODEL_PLACEHOLDERS[configured.provider]);
        }
      })
      .catch((e) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const configuredProviders = providers.filter((p) => p.configured);

  const send = async () => {
    if (!provider || !model.trim() || !input.trim() || streaming) return;
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const nextMessages = [...messages, userMsg];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setInput("");
    setStreaming(true);
    setError(null);

    try {
      for await (const evt of streamAiChat({ provider, model: model.trim(), messages: nextMessages })) {
        if (evt.event === "token") {
          const delta = (evt.data as { delta: string }).delta;
          setMessages((cur) => {
            const copy = [...cur];
            copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + delta };
            return copy;
          });
        } else if (evt.event === "error") {
          setError((evt.data as { detail: string }).detail);
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStreaming(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        className="fixed bottom-6 right-6 z-40 inline-flex items-center gap-2 rounded-full bg-primary text-on-primary text-sm font-medium px-5 py-3 shadow-lg hover:bg-primary-active transition-colors"
      >
        {open ? "Close assistant" : "Ask the assistant"}
      </button>

      {open && (
        <div className="fixed bottom-24 right-6 z-40 flex w-full max-w-sm flex-col rounded-xl border border-hairline bg-surface-card shadow-xl">
          <div className="border-b border-hairline p-3 flex flex-col gap-2">
            {configuredProviders.length === 0 ? (
              <p className="text-xs text-muted">
                No AI provider is configured yet — add an API key in AI settings to use the assistant.
              </p>
            ) : (
              <div className="flex gap-2">
                <div className="w-32">
                  <Select
                    value={provider}
                    onChange={(v) => {
                      setProvider(v as AiProvider);
                      setModel(MODEL_PLACEHOLDERS[v as AiProvider]);
                    }}
                  >
                    <option value="" disabled>
                      Provider
                    </option>
                    {configuredProviders.map((p) => (
                      <option key={p.provider} value={p.provider}>
                        {p.provider}
                      </option>
                    ))}
                  </Select>
                </div>
                <input
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  placeholder="model id"
                  className="min-w-0 flex-1 rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-xs text-ink outline-none focus:border-ink focus:border-2"
                />
              </div>
            )}
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 flex flex-col gap-2 max-h-80 min-h-40">
            {messages.length === 0 && (
              <p className="text-xs text-muted">Ask about VRAM, cluster topology, autoscaling, or any tab here.</p>
            )}
            {messages.map((m, i) => (
              <div
                key={i}
                className={`text-sm rounded-lg px-3 py-2 max-w-[85%] whitespace-pre-wrap ${
                  m.role === "user" ? "self-end bg-primary text-on-primary" : "self-start bg-surface-strong text-ink"
                }`}
              >
                {m.content || (streaming && i === messages.length - 1 ? "…" : "")}
              </div>
            ))}
            {error && <p className="text-xs text-error">{error}</p>}
          </div>

          <div className="border-t border-hairline p-3 flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Ask a question…"
              disabled={configuredProviders.length === 0}
              className="min-w-0 flex-1 rounded-md border border-hairline-strong bg-surface-card px-3 py-2 text-sm text-ink outline-none focus:border-ink focus:border-2 disabled:opacity-40"
            />
            <button
              onClick={send}
              disabled={streaming || configuredProviders.length === 0 || !input.trim()}
              className="inline-flex items-center rounded-full bg-primary text-on-primary text-xs font-medium px-4 py-2 transition-colors hover:bg-primary-active disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </>
  );
}
