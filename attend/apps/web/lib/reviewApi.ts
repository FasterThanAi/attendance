import { apiFetch } from "./api";

/**
 * Types mirror services/api/app/schemas/session_draft.py and
 * schemas/attendance.py field-for-field. Kept in one file (not split per
 * concept) since both the draft and the commit/correct shapes are only
 * used by the review screen -- see components/review/*.
 */

export type ClassSessionStatus =
  | "scheduled"
  | "recording"
  | "processing"
  | "awaiting_review"
  | "committed"
  | "failed";

export type SessionHealth = "good" | "fair" | "poor";

export interface DraftSessionSummary {
  total_enrolled: number;
  proposed_present: number;
  needs_review: number;
  proposed_absent: number;
  unrecognised_clusters: number;
  coverage_percent: number;
  mean_confident_similarity: number | null;
  session_health: SessionHealth;
}

export interface DraftClusterMatch {
  cluster_id: number;
  best_crop_uri: string;
  student_id: number | null;
  student_name: string | null;
  roll_number: string | null;
  similarity: number | null;
  runner_up_similarity: number | null;
  enrollment_photo_uri: string | null;
}

export interface DraftAbsentStudent {
  student_id: number;
  student_name: string;
  roll_number: string;
  enrollment_photo_uri: string | null;
}

export interface SessionDraftResponse {
  session_id: number;
  status: ClassSessionStatus;
  summary: DraftSessionSummary;
  confident: DraftClusterMatch[];
  needs_review: DraftClusterMatch[];
  proposed_absent: DraftAbsentStudent[];
  unrecognised_clusters: DraftClusterMatch[];
}

export type AttendanceStatus = "present" | "absent";
export type AttendanceSource = "auto" | "teacher_confirmed" | "teacher_override";

export interface CommitDecision {
  student_id: number;
  status: AttendanceStatus;
  source: AttendanceSource;
}

export interface CommitRequest {
  request_id: string;
  teacher_id: number;
  decisions: CommitDecision[];
}

export interface CommitCounts {
  total_enrolled: number;
  present: number;
  absent: number;
  auto_count: number;
  teacher_confirmed_count: number;
  teacher_override_count: number;
}

export interface CommitResponse {
  class_session_id: number;
  status: ClassSessionStatus;
  counts: CommitCounts;
  committed_at: string;
  idempotent_replay: boolean;
}

export interface CorrectionRequest {
  status: AttendanceStatus;
  teacher_id: number;
}

export interface CorrectionResponse {
  attendance_record_id: number;
  student_id: number;
  status: AttendanceStatus;
  supersedes_id: number;
  created_at: string;
}

export function fetchSessionDraft(sessionId: number): Promise<SessionDraftResponse> {
  return apiFetch<SessionDraftResponse>(`/sessions/${sessionId}/draft`);
}

export function commitSession(sessionId: number, request: CommitRequest): Promise<CommitResponse> {
  return apiFetch<CommitResponse>(`/sessions/${sessionId}/commit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export function correctAttendance(
  sessionId: number,
  studentId: number,
  body: CorrectionRequest
): Promise<CorrectionResponse> {
  return apiFetch<CorrectionResponse>(`/sessions/${sessionId}/attendance/${studentId}/correct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
