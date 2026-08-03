export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Phase 8 gap fix: the api rewrites raw pipeline artifact paths into
 * "/media/..." paths (see services/api/app/media.py), but a path like that
 * is relative -- an <img src="/media/...">  resolves against whatever
 * origin the browser is currently on, which is the Next.js app
 * (localhost:3000), not the api (localhost:8000). This prefixes any
 * relative media path with API_BASE_URL so images actually load. Already-
 * absolute URLs (http/https, e.g. a future S3 uri) pass through unchanged.
 */
export function mediaUrl(uri: string | null): string | null {
  if (!uri) return uri;
  if (uri.startsWith("http://") || uri.startsWith("https://")) return uri;
  return `${API_BASE_URL}${uri}`;
}

export class ApiError extends Error {
  code: string;

  constructor(code: string, message: string) {
    super(message);
    this.code = code;
  }
}

/** Every error response from the api is {code, message} (non-negotiable
 * rule #7 in the global brief) -- this turns that into a typed ApiError
 * instead of a generic fetch failure, so callers can match on `code`.
 */
export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);

  if (!response.ok) {
    let code = "unknown_error";
    let message = `Request to ${path} failed with status ${response.status}.`;
    try {
      const body = await response.json();
      const detail = body.detail ?? body;
      if (detail?.code) code = detail.code;
      if (detail?.message) message = detail.message;
    } catch {
      // Response wasn't JSON -- keep the generic message above.
    }
    throw new ApiError(code, message);
  }

  return (await response.json()) as T;
}
