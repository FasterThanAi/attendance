"use client";

import { mediaUrl } from "../../lib/api";
import type { DraftClusterMatch } from "../../lib/reviewApi";
import Section from "./Section";

/**
 * Clusters the system could not tie to any enrolled student at all (not
 * even an uncertain guess) -- e.g. a visitor, or a face too poor-quality to
 * match. No student to attach an attendance decision to, so the only
 * action is "dismiss" -- purely a client-side acknowledgement with no
 * backend effect (there is nothing to commit for a cluster with no
 * student_id).
 */
interface Props {
  items: DraftClusterMatch[];
  dismissed: Record<number, true>;
  onDismiss: (clusterId: number) => void;
}

export default function UnrecognisedPeopleSection({ items, dismissed, onDismiss }: Props) {
  const visible = items.filter((item) => !dismissed[item.cluster_id]);

  return (
    <Section
      title="Unrecognised people"
      count={visible.length}
      accentClass="border-gray-300"
      defaultExpanded={false}
      collapsible
    >
      <p className="text-body text-gray-600 px-4 pb-2">
        Detected in the video but not matched to anyone enrolled.
      </p>
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3 px-4 py-3">
        {visible.map((item) => (
          <div key={item.cluster_id} className="text-left">
            <div className="aspect-square w-full rounded-lg bg-gray-100 overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={mediaUrl(item.best_crop_uri) ?? undefined}
                alt="Unrecognised face"
                className="w-full h-full object-cover"
              />
            </div>
            <button
              type="button"
              onClick={() => onDismiss(item.cluster_id)}
              className="w-full min-h-11 mt-1 text-body text-gray-500 underline"
            >
              Dismiss
            </button>
          </div>
        ))}
      </div>
    </Section>
  );
}
