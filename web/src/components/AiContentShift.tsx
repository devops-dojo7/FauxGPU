"use client";

import { ReactNode } from "react";
import { useAiAssistant } from "@/lib/aiAssistantContext";

/** Shrinks the page's own flex-1 column to make room for the fixed assistant
 * sidebar, instead of letting the sidebar sit on top of content it then
 * covers — the rest of the page stays fully visible and interactive, it's
 * just narrower while the assistant is open. */
export function AiContentShift({ children }: { children: ReactNode }) {
  const { open, wide } = useAiAssistant();
  return (
    <div
      className={`flex-1 flex flex-col transition-[margin-right] duration-200 ${
        open ? (wide ? "md:mr-[36rem]" : "md:mr-[24rem]") : ""
      }`}
    >
      {children}
    </div>
  );
}
