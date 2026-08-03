import { API_BASE_URL, apiFetch } from "./api";

/**
 * Resumable chunked upload (Phase 2 deliverable 6).
 *
 * Design notes, matching the UI contract (Section 5 of the roadmap):
 *  - "Nothing blocks on the network... the word 'offline' should never
 *    appear as an error -- it is a normal condition." A failed chunk retries
 *    with exponential backoff rather than surfacing a scary error; only a
 *    hard, unrecoverable failure (e.g. the server rejects the video outright)
 *    is shown as an actual error state.
 *  - "Must survive the browser tab being backgrounded" and resume on reopen:
 *    upload progress (upload_id + which chunks are confirmed) is persisted to
 *    localStorage, keyed by a fingerprint of the file itself (name + size +
 *    lastModified) plus the class session, so re-selecting the same file
 *    after closing the tab picks up from GET /uploads/{id}'s server-reported
 *    state rather than re-uploading from scratch.
 */

const CHUNK_SIZE_BYTES = 5 * 1024 * 1024;
const MAX_RETRIES_PER_CHUNK = 6;
const BASE_BACKOFF_MS = 500;

interface UploadCreateResponse {
  upload_id: string;
  chunk_size_bytes: number;
  total_chunks: number;
}

interface UploadStatusResponse {
  upload_id: string;
  total_chunks: number;
  received_chunks: number[];
  is_complete: boolean;
}

export interface PreflightCheckResult {
  code: string;
  severity: "info" | "warn" | "fail";
  message: string;
}

export interface UploadCompleteResponse {
  video_upload: {
    id: number;
    class_session_id: number;
    storage_uri: string;
    duration_seconds: number;
    width: number;
    height: number;
    fps: number;
    bytes: number;
    uploaded_at: string;
  };
  preflight: {
    status: "pass" | "warn" | "fail";
    checks: PreflightCheckResult[];
  };
}

export interface UploadProgress {
  chunksReceived: number;
  totalChunks: number;
  fractionComplete: number;
}

type ProgressCallback = (progress: UploadProgress) => void;

function fingerprint(file: File, classSessionId: number): string {
  return `attend-upload:${classSessionId}:${file.name}:${file.size}:${file.lastModified}`;
}

function loadPersistedUploadId(file: File, classSessionId: number): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(fingerprint(file, classSessionId));
}

function persistUploadId(file: File, classSessionId: number, uploadId: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(fingerprint(file, classSessionId), uploadId);
}

function clearPersistedUploadId(file: File, classSessionId: number): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(fingerprint(file, classSessionId));
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function uploadChunkWithRetry(uploadId: string, chunkIndex: number, chunk: Blob): Promise<void> {
  for (let attempt = 0; attempt < MAX_RETRIES_PER_CHUNK; attempt++) {
    try {
      const response = await fetch(`${API_BASE_URL}/uploads/${uploadId}/chunks/${chunkIndex}`, {
        method: "PUT",
        body: chunk,
      });
      if (response.ok) return;
      // A 4xx here (e.g. invalid_chunk_index) won't fix itself by retrying --
      // still capped, since MAX_RETRIES_PER_CHUNK bounds all cases.
    } catch {
      // Network error -- exactly the "offline is normal" case. Fall through to backoff.
    }
    const backoff = BASE_BACKOFF_MS * 2 ** attempt;
    await sleep(backoff);
  }
  throw new Error(`Chunk ${chunkIndex} failed after ${MAX_RETRIES_PER_CHUNK} attempts.`);
}

/**
 * Uploads `file` in 5MB chunks, resuming from wherever a previous attempt
 * (in this session or a prior one, via localStorage) left off. Calls
 * `onProgress` after every chunk. Returns the assembled + validated
 * video_upload record plus its pre-flight result.
 */
export async function uploadVideoResumable(
  file: File,
  classSessionId: number,
  onProgress?: ProgressCallback
): Promise<UploadCompleteResponse> {
  let uploadId = loadPersistedUploadId(file, classSessionId);
  let totalChunks: number;
  let alreadyReceived: Set<number>;

  if (uploadId) {
    // Resuming: ask the server what it already has rather than trusting
    // local state, which could be stale if e.g. the server data was reset.
    try {
      const status = await apiFetch<UploadStatusResponse>(`/uploads/${uploadId}`);
      totalChunks = status.total_chunks;
      alreadyReceived = new Set(status.received_chunks);
    } catch {
      // The server doesn't recognise this upload_id any more -- start over.
      uploadId = null;
      alreadyReceived = new Set();
      totalChunks = 0;
    }
  } else {
    alreadyReceived = new Set();
    totalChunks = 0;
  }

  if (!uploadId) {
    const created = await apiFetch<UploadCreateResponse>("/uploads", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        class_session_id: classSessionId,
        filename: file.name,
        total_size_bytes: file.size,
      }),
    });
    uploadId = created.upload_id;
    totalChunks = created.total_chunks;
    persistUploadId(file, classSessionId, uploadId);
  }

  for (let i = 0; i < totalChunks; i++) {
    if (alreadyReceived.has(i)) {
      onProgress?.({ chunksReceived: alreadyReceived.size, totalChunks, fractionComplete: alreadyReceived.size / totalChunks });
      continue;
    }
    const start = i * CHUNK_SIZE_BYTES;
    const end = Math.min(start + CHUNK_SIZE_BYTES, file.size);
    const chunk = file.slice(start, end);

    await uploadChunkWithRetry(uploadId, i, chunk);
    alreadyReceived.add(i);
    onProgress?.({ chunksReceived: alreadyReceived.size, totalChunks, fractionComplete: alreadyReceived.size / totalChunks });
  }

  const result = await apiFetch<UploadCompleteResponse>(`/uploads/${uploadId}/complete`, { method: "POST" });

  // Only clear local resume state once the server has confirmed the video
  // is valid -- if /complete rejects the video (too short, wrong
  // resolution), the raw chunks are still there in case a retry is useful,
  // and re-selecting the SAME bad file shouldn't re-upload it from scratch.
  clearPersistedUploadId(file, classSessionId);

  return result;
}
