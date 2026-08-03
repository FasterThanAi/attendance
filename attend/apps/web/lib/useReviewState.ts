import { useMemo, useState } from "react";
import type { CommitDecision, SessionDraftResponse } from "./reviewApi";

/**
 * Local review state for the Phase 8 review screen.
 *
 * Design (see roadmap Section 5 / UI contract): the four header numbers
 * (total_enrolled, proposed_present, needs_review, proposed_absent) are
 * STATIC -- they are the original system partition and must always sum to
 * total_enrolled, so a teacher can trust "the parts add up to the whole" no
 * matter what they've tapped. The ONE number that live-updates is the
 * attendance percentage, derived here from three pieces of local state:
 *
 *  - needsReviewDecisions: every needs_review student must get "present" or
 *    "not_them" before commit is allowed (the progressive-review gate).
 *  - notFoundOverrides: a "not found" student the teacher taps present.
 *  - confirmedPresentOverrides: a "confirmed present" student the teacher
 *    taps absent.
 *
 * Plain Records (not Sets) are used for state so React's shallow-copy
 * immutable updates stay simple, and so this hook's state is trivially
 * JSON-serialisable if it ever needs to be persisted mid-review.
 */

export type NeedsReviewDecision = "present" | "not_them";

export interface ReviewState {
  needsReviewDecisions: Record<number, NeedsReviewDecision>;
  notFoundOverrides: Record<number, true>;
  confirmedPresentOverrides: Record<number, true>;
  dismissedClusterIds: Record<number, true>;
}

const EMPTY_STATE: ReviewState = {
  needsReviewDecisions: {},
  notFoundOverrides: {},
  confirmedPresentOverrides: {},
  dismissedClusterIds: {},
};

export function useReviewState(draft: SessionDraftResponse | null) {
  const [state, setState] = useState<ReviewState>(EMPTY_STATE);

  function decideNeedsReview(studentId: number, decision: NeedsReviewDecision): void {
    setState((s) => ({ ...s, needsReviewDecisions: { ...s.needsReviewDecisions, [studentId]: decision } }));
  }

  function toggleNotFoundOverride(studentId: number): void {
    setState((s) => {
      const next = { ...s.notFoundOverrides };
      if (next[studentId]) delete next[studentId];
      else next[studentId] = true;
      return { ...s, notFoundOverrides: next };
    });
  }

  function toggleConfirmedPresentOverride(studentId: number): void {
    setState((s) => {
      const next = { ...s.confirmedPresentOverrides };
      if (next[studentId]) delete next[studentId];
      else next[studentId] = true;
      return { ...s, confirmedPresentOverrides: next };
    });
  }

  function dismissCluster(clusterId: number): void {
    setState((s) => ({ ...s, dismissedClusterIds: { ...s.dismissedClusterIds, [clusterId]: true } }));
  }

  const needsReviewTotal = draft?.needs_review.filter((item) => item.student_id !== null).length ?? 0;
  const needsReviewHandledCount = Object.keys(state.needsReviewDecisions).length;
  const canCommit = draft !== null && needsReviewHandledCount >= needsReviewTotal;

  const livePresentCount = useMemo(() => {
    if (!draft) return 0;
    let count = draft.confident.length;
    count -= Object.keys(state.confirmedPresentOverrides).length;
    count += Object.values(state.needsReviewDecisions).filter((d) => d === "present").length;
    count += Object.keys(state.notFoundOverrides).length;
    return count;
  }, [draft, state]);

  const livePercent =
    draft && draft.summary.total_enrolled > 0
      ? Math.round((livePresentCount / draft.summary.total_enrolled) * 100)
      : 0;

  function buildDecisions(): CommitDecision[] {
    const decisions: CommitDecision[] = [];
    for (const [studentIdStr, decision] of Object.entries(state.needsReviewDecisions)) {
      const studentId = Number(studentIdStr);
      decisions.push(
        decision === "present"
          ? { student_id: studentId, status: "present", source: "teacher_confirmed" }
          : { student_id: studentId, status: "absent", source: "teacher_override" }
      );
    }
    for (const studentIdStr of Object.keys(state.notFoundOverrides)) {
      decisions.push({ student_id: Number(studentIdStr), status: "present", source: "teacher_override" });
    }
    for (const studentIdStr of Object.keys(state.confirmedPresentOverrides)) {
      decisions.push({ student_id: Number(studentIdStr), status: "absent", source: "teacher_override" });
    }
    return decisions;
  }

  return {
    state,
    decideNeedsReview,
    toggleNotFoundOverride,
    toggleConfirmedPresentOverride,
    dismissCluster,
    needsReviewHandledCount,
    needsReviewTotal,
    canCommit,
    livePresentCount,
    livePercent,
    buildDecisions,
  };
}
