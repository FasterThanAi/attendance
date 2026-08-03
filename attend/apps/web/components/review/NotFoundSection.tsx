"use client";

import type { DraftAbsentStudent } from "../../lib/reviewApi";
import PhotoGrid from "./PhotoGrid";
import Section from "./Section";

/**
 * Expanded photo grid (roadmap Section 5): students the system did not find
 * in today's video, default state absent, tap to override present.
 */
interface Props {
  items: DraftAbsentStudent[];
  overrides: Record<number, true>;
  onToggle: (studentId: number) => void;
}

export default function NotFoundSection({ items, overrides, onToggle }: Props) {
  const overriddenCount = Object.keys(overrides).length;

  if (items.length === 0) {
    return (
      <Section title="Not found" count={0} accentClass="border-absent">
        <p className="text-body text-gray-600 px-4 py-3">Everyone enrolled was found.</p>
      </Section>
    );
  }

  return (
    <Section title="Not found" count={items.length} accentClass="border-absent">
      <p className="text-body text-gray-600 px-4 pb-2">
        Tap anyone who was actually here today.
      </p>
      {overriddenCount > 0 && (
        <p className="text-body text-review px-4 pb-2">{overriddenCount} marked present by you</p>
      )}
      <PhotoGrid
        items={items.map((s) => ({
          id: s.student_id,
          name: s.student_name,
          rollNumber: s.roll_number,
          photoUri: s.enrollment_photo_uri,
          selected: !!overrides[s.student_id],
        }))}
        onToggle={onToggle}
        selectedLabel="Present"
      />
    </Section>
  );
}
