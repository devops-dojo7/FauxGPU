"use client";

/**
 * Pill tab bar. There are 13 tabs — more than fit on one line on most window
 * widths — so extra tabs wrap onto additional lines instead of requiring a
 * horizontal scroll (which had no way to communicate that the bar could be
 * scrolled without a large fade/scrollbar treatment).
 */
export function TabNav<T extends string>({
  tabs,
  active,
  onSelect,
}: {
  tabs: [T, string][];
  active: T;
  onSelect: (id: T) => void;
}) {
  return (
    <nav className="flex flex-wrap max-w-full rounded-xl border border-hairline bg-surface-card p-1 gap-1">
      {tabs.map(([id, label]) => (
        <button
          key={id}
          onClick={() => onSelect(id)}
          className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
            active === id ? "bg-primary text-on-primary" : "text-body hover:bg-surface-strong"
          }`}
        >
          {label}
        </button>
      ))}
    </nav>
  );
}
