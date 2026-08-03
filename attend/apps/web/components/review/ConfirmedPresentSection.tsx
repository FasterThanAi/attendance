"use client";

import type { DraftClusterMatch } from "../../lib/reviewApi";
import PhotoGrid from "./PhotoGrid";
import Section from "./Section";

/**
 * Collapsed by default, expandable (roadmap Section 5): students the
 * system confidently matched. A teacher rarely needs to open this, but can
 * tap a student here to override them to absent (e.g. a proxy/mistaken
 * match) -- always a teacher_override, never silently accepted.
 */
interface Props {
  items: DraftClusterMatch[];
  overrides: Record<number, true>;
  onToggle: (studentId: number) => void;
}

export default function ConfirmedPresentSection({ items, overrides, onToggle }: Props) {
  const overriddenCount = Object.keys(overrides).length;

  return (
    <Section
      title="Confirmed present"
      count={items.length}
      accentClass="border-confirmed"
      defaultExpanded={false}
      collapsible
    >
      <p className="text-body text-gray-600 px-4 pb-2">
        Tap anyone who was not actually here today.
      </p>
      {overriddenCount > 0 && (
        <p className="text-body text-review px-4 pb-2">{overriddenCount} marked absent by you</p>
      )}
      <PhotoGrid
        items={items
          .filter((item) => item.student_id !== null)
          .map((item) => ({
            id: item.student_id as number,
            name: item.student_name ?? "Unknown",
            rollNumber: item.roll_number,
            photoUri: item.enrollment_photo_uri,
            selected: !!overrides[item.student_id as number],
          }))}
        onToggle={onToggle}
        selectedLabel="Absent"
      />
    </Section>
  );
}
