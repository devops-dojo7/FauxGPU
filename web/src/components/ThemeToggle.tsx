"use client";

import { useEffect, useState } from "react";

type Theme = "light" | "dark";

function systemPrefersDark() {
  return typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

export function ThemeToggle() {
  // null until mounted — avoids a hydration mismatch between server-rendered
  // markup (no theme opinion) and whatever localStorage/system says client-side.
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reading localStorage/matchMedia must happen post-mount to avoid a server/client hydration mismatch
    setTheme(stored === "light" || stored === "dark" ? stored : systemPrefersDark() ? "dark" : "light");
  }, []);

  const toggle = () => {
    const next: Theme = (theme ?? (systemPrefersDark() ? "dark" : "light")) === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("theme", next);
    document.documentElement.setAttribute("data-theme", next);
  };

  if (theme === null) {
    return <div className="w-9 h-9" aria-hidden />;
  }

  return (
    <button
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className="w-9 h-9 shrink-0 rounded-full border border-hairline-strong bg-surface-card text-ink flex items-center justify-center hover:bg-surface-strong transition-colors"
    >
      {theme === "dark" ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
          <circle cx="12" cy="12" r="4.5" />
          <path
            strokeLinecap="round"
            d="M12 2.5v2M12 19.5v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M2.5 12h2M19.5 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"
          />
        </svg>
      ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
          <path strokeLinecap="round" strokeLinejoin="round" d="M20.5 14.5A8.5 8.5 0 1 1 9.5 3.5a7 7 0 0 0 11 11Z" />
        </svg>
      )}
    </button>
  );
}
