"use client";

import { useEffect, useState } from "react";
import CommitConfirmationModal from "../../../../components/review/CommitConfirmationModal";
import ConfirmedPresentSection from "../../../../components/review/ConfirmedPresentSection";
import NeedsYourCheckSection from "../../../../components/review/NeedsYourCheckSection";
import NotFoundSection from "../../../../components/review/NotFoundSection";
import SummaryHeader from "../../../../components/review/SummaryHeader";
import UnrecognisedPeopleSection from "../../../../components/review/UnrecognisedPeopleSection";
import { ApiError } from "../../../../lib/api";
import {
  cacheDraft,
  flushQueuedCommit,
  getCachedDraft,
  getQueuedCommit,
  queueCommit,
} from "../../../../lib/offlineStorage";
import { commitSession, fetchSessionDraft, type SessionDraftResponse } from "../../../../lib/reviewApi";
import { useReviewState } from "../../../../lib/useReviewState";

/**
 * Phase 8 deliverable 1-2, 5: the teacher review screen.
 *
 * Data flow: load the draft (network first, cached copy as a fallback so
 * the screen still renders with no signal at all -- "offline" is a normal
 * condition here, never an error, per the UI contract's core principle 2).
 * Local decisions accumulate in useReviewState until the teacher taps
 * commit; commit is gated on every needs-your-check item having a decision
 * (progressive review), and goes through a distinct confirmation step
 * before it fires, per core principle 4 (irreversible actions need to feel
 * irreversible).
 *
 * ASSUMPTION: there is no auth system yet anywhere in this codebase (no
 * phase has built one), so teacher_id is a temporary manual input here,
 * same convention as app/record/page.tsx's temporary session-id input for
 * Phase 2 testing.
 */
export default function ReviewPage({ params }: { params: { id: string } }) {
  const sessionId = Number(params.id);

  const [draft, setDraft] = useState<SessionDraftResponse | null>(null);
  const [teacherId, setTeacherId] = useState(1);
  const [isOffline, setIsOffline] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [committed, setCommitted] = useState(false);
  const [pendingOffline, setPendingOffline] = useState(false);
  const [requestId] = useState(() => crypto.randomUUID());

  const review = useReviewState(draft);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const cached = getCachedDraft(sessionId);
      if (cached && !cancelled) setDraft(cached);

      try {
        const fresh = await fetchSessionDraft(sessionId);
        if (cancelled) return;
        setDraft(fresh);
        cacheDraft(sessionId, fresh);
        setIsOffline(false);
        setLoadError(null);
      } catch (err) {
        if (cancelled) return;
        if (cached) {
          // We have something to show -- treat this quietly as offline,
          // not as an error state.
          setIsOffline(true);
        } else if (err instanceof ApiError) {
          setLoadError(err.message);
        } else {
          setIsOffline(true);
        }
      }

      const queued = getQueuedCommit(sessionId);
      if (queued && !cancelled) setPendingOffline(true);
      const flushed = await flushQueuedCommit(sessionId);
      if (flushed && !cancelled) {
        setPendingOffline(false);
        setCommitted(true);
      }
    }

    load();
    const handleOnline = () => {
      setIsOffline(false);
      flushQueuedCommit(sessionId).then((flushed) => {
        if (flushed) {
          setPendingOffline(false);
          setCommitted(true);
        }
      });
    };
    window.addEventListener("online", handleOnline);
    return () => {
      cancelled = true;
      window.removeEventListener("online", handleOnline);
    };
  }, [sessionId]);

  if (loadError) {
    return (
      <main className="max-w-md mx-auto p-4">
        <p className="text-body text-gray-600">{loadError}</p>
      </main>
    );
  }

  if (!draft) {
    return (
      <main className="max-w-md mx-auto p-4">
        <p className="text-body text-gray-500">Loading...</p>
      </main>
    );
  }

  if (committed) {
    return (
      <main className="max-w-md mx-auto p-4 space-y-2">
        <p className="text-emphasis font-medium text-gray-900">Attendance recorded</p>
        <p className="text-body text-gray-600">
          You can still correct an individual student from the class history if needed.
        </p>
      </main>
    );
  }

  async function handleConfirmCommit() {
    setIsSubmitting(true);
    setSubmitError(null);
    const request = { request_id: requestId, teacher_id: teacherId, decisions: review.buildDecisions() };
    try {
      await commitSession(sessionId, request);
      setShowConfirm(false);
      setCommitted(true);
    } catch (err) {
      if (isOffline || !(err instanceof ApiError)) {
        // No network -- queue it and show an honest pending state, not an
        // error (core principle 2).
        queueCommit(sessionId, request);
        setPendingOffline(true);
        setShowConfirm(false);
      } else {
        setSubmitError(err.message);
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="max-w-md mx-auto pb-24">
      <SummaryHeader
        summary={draft.summary}
        livePresentCount={review.livePresentCount}
        livePercent={review.livePercent}
      />

      {isOffline && (
        <p className="text-body text-gray-500 px-4 py-2">
          Working offline. Showing the last saved copy of this class.
        </p>
      )}

      {pendingOffline && (
        <p className="text-body text-review px-4 py-2">
          Attendance is ready and will be recorded once you are back online.
        </p>
      )}

      <div className="space-y-3 p-4">
        <NeedsYourCheckSection
          items={draft.needs_review}
          decisions={review.state.needsReviewDecisions}
          onDecide={review.decideNeedsReview}
        />
        <NotFoundSection
          items={draft.proposed_absent}
          overrides={review.state.notFoundOverrides}
          onToggle={review.toggleNotFoundOverride}
        />
        <ConfirmedPresentSection
          items={draft.confident}
          overrides={review.state.confirmedPresentOverrides}
          onToggle={review.toggleConfirmedPresentOverride}
        />
        <UnrecognisedPeopleSection
          items={draft.unrecognised_clusters}
          dismissed={review.state.dismissedClusterIds}
          onDismiss={review.dismissCluster}
        />

        <div className="pt-1">
          <label className="block text-body text-gray-500 mb-1" htmlFor="teacher-id">
            Teacher id (temporary, for testing)
          </label>
          <input
            id="teacher-id"
            type="number"
            className="h-11 min-h-11 w-full rounded border border-gray-300 px-3 text-body"
            value={teacherId}
            onChange={(e) => setTeacherId(Number(e.target.value))}
          />
        </div>
      </div>

      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 p-4">
        {submitError && <p className="text-body text-absent mb-2">{submitError}</p>}
        {!review.canCommit && (
          <p className="text-body text-gray-500 mb-2">
            {review.needsReviewTotal - review.needsReviewHandledCount} left to check before you can record attendance
          </p>
        )}
        <button
          type="button"
          disabled={!review.canCommit}
          onClick={() => setShowConfirm(true)}
          className="w-full min-h-11 rounded-lg bg-confirmed text-white text-emphasis font-medium py-3 disabled:opacity-40"
        >
          Record attendance
        </button>
      </div>

      {showConfirm && (
        <CommitConfirmationModal
          presentCount={review.livePresentCount}
          absentCount={draft.summary.total_enrolled - review.livePresentCount}
          totalEnrolled={draft.summary.total_enrolled}
          isSubmitting={isSubmitting}
          onConfirm={handleConfirmCommit}
          onCancel={() => setShowConfirm(false)}
        />
      )}
    </main>
  );
}
