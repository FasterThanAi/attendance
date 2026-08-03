"use client";

import { useEffect, useRef } from "react";
import type { DraftClusterMatch } from "../../lib/reviewApi";
import type { NeedsReviewDecision } from "../../lib/useReviewState";
import PhotoTile from "./PhotoTile";
import Section from "./Section";

/**
 * The priority section (roadmap Section 5): one uncertain match at a time,
 * side-by-side photos, a binary decision. Shown as a one-card stepper
 * (rather than a scrolling list of all needs_review items) so a teacher
 * makes one focused decision at a time -- matches the "under 30 seconds"
 * design principle much better than scanning a grid of ambiguous faces.
 *
 * Supports keyboard (Left = not them, Right = yes present) and swipe
 * (same mapping) in addition to the two on-screen buttons, per the
 * deliverable's explicit "keyboard + swipe" requirement.
 */
interface Props {
  items: DraftClusterMatch[];
  decisions: Record<number, NeedsReviewDecision>;
  onDecide: (studentId: number, decision: NeedsReviewDecision) => void;
}

const SWIPE_THRESHOLD_PX = 60;

export default function NeedsYourCheckSection({ items, decisions, onDecide }: Props) {
  const pending = items.filter((item) => item.student_id !== null && !decisions[item.student_id]);
  const current = pending[0];
  const touchStartX = useRef<number | null>(null);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (!current || current.student_id === null) return;
      if (e.key === "ArrowRight") onDecide(current.student_id, "present");
      if (e.key === "ArrowLeft") onDecide(current.student_id, "not_them");
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [current, onDecide]);

  if (items.length === 0) return null;

  if (!current) {
    return (
      <Section title="Needs your check" count={0} accentClass="border-review">
        <p className="text-body text-gray-600 px-4 py-3">All checked.</p>
      </Section>
    );
  }

  function handleTouchStart(e: React.TouchEvent) {
    const touch = e.touches[0];
    if (touch) touchStartX.current = touch.clientX;
  }

  function handleTouchEnd(e: React.TouchEvent) {
    const touch = e.changedTouches[0];
    if (touchStartX.current === null || !current || current.student_id === null || !touch) return;
    const deltaX = touch.clientX - touchStartX.current;
    if (deltaX > SWIPE_THRESHOLD_PX) onDecide(current.student_id, "present");
    else if (deltaX < -SWIPE_THRESHOLD_PX) onDecide(current.student_id, "not_them");
    touchStartX.current = null;
  }

  return (
    <Section title="Needs your check" count={pending.length} accentClass="border-review">
      <div className="px-4 py-3" onTouchStart={handleTouchStart} onTouchEnd={handleTouchEnd}>
        <p className="text-body text-gray-500 mb-2">
          {pending.length} left to check
        </p>
        <div className="grid grid-cols-2 gap-3 mb-4">
          <PhotoTile
            label="Enrollment photo"
            src={current.enrollment_photo_uri}
            alt={`${current.student_name ?? "student"}'s enrollment photo`}
          />
          <PhotoTile
            label="Detected today"
            src={current.best_crop_uri}
            alt={`Face detected in today's video, possibly ${current.student_name ?? "this student"}`}
          />
        </div>
        <div className="text-center mb-4">
          <div className="text-emphasis font-medium text-gray-900">{current.student_name}</div>
          <div className="text-body text-gray-500">Roll number {current.roll_number}</div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => current.student_id !== null && onDecide(current.student_id, "not_them")}
            className="min-h-11 rounded-lg border border-absent text-absent text-emphasis font-medium py-3"
          >
            Not them
          </button>
          <button
            type="button"
            onClick={() => current.student_id !== null && onDecide(current.student_id, "present")}
            className="min-h-11 rounded-lg bg-confirmed text-white text-emphasis font-medium py-3"
          >
            Yes, present
          </button>
        </div>
      </div>
    </Section>
  );
}
