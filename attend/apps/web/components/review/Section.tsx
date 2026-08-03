"use client";

import { useState, type ReactNode } from "react";

/**
 * Shared section shell for the review screen (Needs Your Check / Not Found /
 * Confirmed Present / Unrecognised People). UI contract: 8px corners (never
 * fully rounded except avatars), a colored top border is the ONLY use of
 * the section's semantic color -- everything else in the section stays
 * neutral gray, per "3 semantic colors used only for meaning."
 */
interface SectionProps {
  title: string;
  count: number;
  accentClass?: string;
  defaultExpanded?: boolean;
  collapsible?: boolean;
  children: ReactNode;
}

export default function Section({
  title,
  count,
  accentClass = "border-gray-200",
  defaultExpanded = true,
  collapsible = false,
  children,
}: SectionProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <section className={`border-t-4 ${accentClass} bg-white rounded-lg overflow-hidden`}>
      <button
        type="button"
        className="w-full flex items-center justify-between px-4 py-3 text-left min-h-11"
        onClick={() => collapsible && setExpanded((e) => !e)}
        aria-expanded={expanded}
        disabled={!collapsible}
      >
        <span className="text-emphasis font-medium text-gray-900">{title}</span>
        <span className="flex items-center gap-2">
          <span className="text-body text-gray-500">{count}</span>
          {collapsible && (
            <span aria-hidden="true" className="text-gray-400">
              {expanded ? "−" : "+"}
            </span>
          )}
        </span>
      </button>
      {expanded && children}
    </section>
  );
}
