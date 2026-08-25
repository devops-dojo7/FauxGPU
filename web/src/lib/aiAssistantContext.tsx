"use client";

import { createContext, ReactNode, useContext, useState } from "react";

interface AiAssistantContextValue {
  open: boolean;
  setOpen: (v: boolean) => void;
  wide: boolean;
  setWide: (v: boolean) => void;
}

const AiAssistantContext = createContext<AiAssistantContextValue | null>(null);

export function AiAssistantProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [wide, setWide] = useState(false);
  return <AiAssistantContext.Provider value={{ open, setOpen, wide, setWide }}>{children}</AiAssistantContext.Provider>;
}

export function useAiAssistant() {
  const ctx = useContext(AiAssistantContext);
  if (!ctx) throw new Error("useAiAssistant must be used within AiAssistantProvider");
  return ctx;
}
