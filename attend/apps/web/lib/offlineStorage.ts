import type { CommitRequest, CommitResponse, SessionDraftResponse } from "./reviewApi";
import { commitSession } from "./reviewApi";

/**
 * Offline behaviour (Phase 8 deliverable 5, and the UI contract's core
 * principle 2: "Nothing blocks on the network... the word 'offline' should
 * never appear as an error -- it is a normal condition").
 *
 * Two things get cached to localStorage, both keyed by class_session_id:
 *  - the draft itself, so the review screen can render from a cold start
 *    with no network at all (a teacher walking into a classroom with one
 *    bar of signal).
 *  - a queued commit, if the teacher finishes review and taps "commit" with
 *    no network -- the decision is captured immediately and shown as
 *    pending honestly, then flushed automatically the next time the app
 *    detects it's online (see flushQueuedCommit, called from the review
 *    page's `online` event listener and on mount).
 */

const DRAFT_CACHE_PREFIX = "attend-draft-cache:";
const PENDING_COMMIT_PREFIX = "attend-pending-commit:";

export interface QueuedCommit {
  sessionId: number;
  request: CommitRequest;
  queuedAt: string;
}

function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function cacheDraft(sessionId: number, draft: SessionDraftResponse): void {
  if (!isBrowser()) return;
  window.localStorage.setItem(`${DRAFT_CACHE_PREFIX}${sessionId}`, JSON.stringify(draft));
}

export function getCachedDraft(sessionId: number): SessionDraftResponse | null {
  if (!isBrowser()) return null;
  const raw = window.localStorage.getItem(`${DRAFT_CACHE_PREFIX}${sessionId}`);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as SessionDraftResponse;
  } catch {
    return null; // corrupted cache entry -- treat as absent rather than crash the review screen
  }
}

export function queueCommit(sessionId: number, request: CommitRequest): void {
  if (!isBrowser()) return;
  const entry: QueuedCommit = { sessionId, request, queuedAt: new Date().toISOString() };
  window.localStorage.setItem(`${PENDING_COMMIT_PREFIX}${sessionId}`, JSON.stringify(entry));
}

export function getQueuedCommit(sessionId: number): QueuedCommit | null {
  if (!isBrowser()) return null;
  const raw = window.localStorage.getItem(`${PENDING_COMMIT_PREFIX}${sessionId}`);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as QueuedCommit;
  } catch {
    return null;
  }
}

export function clearQueuedCommit(sessionId: number): void {
  if (!isBrowser()) return;
  window.localStorage.removeItem(`${PENDING_COMMIT_PREFIX}${sessionId}`);
}

/**
 * Attempts to send whatever commit is queued for this session. Safe to
 * call speculatively (e.g. on mount, on the browser's `online` event, on a
 * timer) -- a no-op if nothing is queued. The commit endpoint's own
 * idempotency (request_id) means calling this more than once for the same
 * queued request is harmless even if an earlier attempt actually succeeded
 * but the response was lost to a flaky connection.
 */
export async function flushQueuedCommit(sessionId: number): Promise<CommitResponse | null> {
  const queued = getQueuedCommit(sessionId);
  if (!queued) return null;

  try {
    const response = await commitSession(sessionId, queued.request);
    clearQueuedCommit(sessionId);
    return response;
  } catch {
    // Still offline, or the server is unreachable -- leave it queued for
    // the next attempt. Never surface this as an error; see core principle 2.
    return null;
  }
}
