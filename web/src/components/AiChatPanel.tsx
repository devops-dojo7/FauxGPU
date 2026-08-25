"use client";

import { useEffect, useRef, useState } from "react";
import { fetchAiProviders, fetchLangfuseStatus, streamAiChat } from "@/lib/api";
import { AiProvider, AiProviderStatus, ChatMessage, LangfuseStatus } from "@/lib/types";
import { useAiAssistant } from "@/lib/aiAssistantContext";
import { Select } from "./ui";
import { AiModelSelect } from "./AiModelSelect";

function SparkleIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" className={className}>
      <path d="M11 2.5a1 1 0 0 1 .95.68l1.2 3.55a4.5 4.5 0 0 0 2.8 2.8l3.55 1.2a1 1 0 0 1 0 1.9l-3.55 1.2a4.5 4.5 0 0 0-2.8 2.8l-1.2 3.55a1 1 0 0 1-1.9 0l-1.2-3.55a4.5 4.5 0 0 0-2.8-2.8l-3.55-1.2a1 1 0 0 1 0-1.9l3.55-1.2a4.5 4.5 0 0 0 2.8-2.8l1.2-3.55a1 1 0 0 1 .95-.68Zm8-.5a.75.75 0 0 1 .71.51l.4 1.17a1.75 1.75 0 0 0 1.08 1.08l1.17.4a.75.75 0 0 1 0 1.42l-1.17.4a1.75 1.75 0 0 0-1.08 1.08l-.4 1.17a.75.75 0 0 1-1.42 0l-.4-1.17a1.75 1.75 0 0 0-1.08-1.08l-1.17-.4a.75.75 0 0 1 0-1.42l1.17-.4a1.75 1.75 0 0 0 1.08-1.08l.4-1.17A.75.75 0 0 1 19 2Z" />
    </svg>
  );
}

function ExpandIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.75">
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.75">
      <path strokeLinecap="round" d="M5 5l14 14M19 5L5 19" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}

export function AiChatPanel() {
  const { open, setOpen, wide, setWide } = useAiAssistant();
  const [providers, setProviders] = useState<AiProviderStatus[]>([]);
  const [provider, setProvider] = useState<AiProvider | "">("");
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [langfuse, setLangfuse] = useState<LangfuseStatus | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    fetchAiProviders()
      .then((all) => {
        setProviders(all);
        const configured = all.find((p) => p.configured);
        if (configured && !provider) {
          setProvider(configured.provider);
        }
      })
      .catch((e) => setError(e.message));
    fetchLangfuseStatus()
      .then(setLangfuse)
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  useEffect(() => {
    if (!open) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [open, setOpen]);

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
      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed top-6 right-6 z-40 inline-flex items-center gap-2 rounded-full border border-hairline-strong bg-surface-card text-ink text-sm font-medium px-4 py-2 shadow-lg hover:bg-surface-strong transition-colors"
        >
          <SparkleIcon />
          Ask Assistant
        </button>
      )}

      <div
        className={`fixed inset-y-0 right-0 z-50 flex w-full flex-col border-l border-hairline bg-surface-card shadow-2xl transition-transform duration-200 ${
          wide ? "max-w-xl" : "max-w-sm"
        } ${open ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="flex items-center justify-between px-4 py-3.5 border-b border-hairline shrink-0">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink">
            <SparkleIcon className="text-primary" />
            Assistant
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setWide(!wide)}
              aria-label={wide ? "Collapse panel" : "Expand panel"}
              title={wide ? "Collapse panel" : "Expand panel"}
              className="w-7 h-7 flex items-center justify-center rounded-md text-muted hover:text-ink hover:bg-surface-strong transition-colors"
            >
              <ExpandIcon />
            </button>
            <button
              onClick={() => setOpen(false)}
              aria-label="Close assistant"
              title="Close assistant"
              className="w-7 h-7 flex items-center justify-center rounded-md text-muted hover:text-ink hover:bg-surface-strong transition-colors"
            >
              <CloseIcon />
            </button>
          </div>
        </div>

        <p className="text-center text-xs text-muted px-6 py-2.5 border-b border-hairline shrink-0">
          Responses are generated using AI and may contain mistakes.
          {langfuse?.tracing_available && (
            <>
              {" "}
              <a
                href={langfuse.public_url}
                target="_blank"
                rel="noreferrer"
                className="text-body-strong underline decoration-hairline-strong hover:text-ink transition-colors"
              >
                View traces in Langfuse ↗
              </a>
            </>
          )}
        </p>

        <div className="px-4 py-3 border-b border-hairline shrink-0">
          {configuredProviders.length === 0 ? (
            <p className="text-xs text-muted">
              No AI provider is configured yet — add an API key in AI settings to use the assistant.
            </p>
          ) : (
            <div className="flex gap-2">
              <div className="w-32 shrink-0">
                <Select value={provider} onChange={(v) => setProvider(v as AiProvider)}>
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
              <div className="min-w-0 flex-1">
                <AiModelSelect provider={provider} model={model} onModelChange={setModel} />
              </div>
            </div>
          )}
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
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

        <div className="border-t border-hairline p-3 flex items-end gap-2 shrink-0">
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
            className="min-w-0 flex-1 rounded-md border border-hairline-strong bg-surface-card px-3 py-2.5 text-sm text-ink outline-none focus:border-ink focus:border-2 disabled:opacity-40"
          />
          <button
            onClick={send}
            disabled={streaming || configuredProviders.length === 0 || !input.trim()}
            aria-label="Send"
            className="w-9 h-9 shrink-0 flex items-center justify-center rounded-full bg-primary text-on-primary transition-colors hover:bg-primary-active disabled:opacity-40"
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </>
  );
}
