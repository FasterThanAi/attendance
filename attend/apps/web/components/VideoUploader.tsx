"use client";

import { useState } from "react";
import { ApiError } from "../lib/api";
import {
  UploadCompleteResponse,
  UploadProgress,
  uploadVideoResumable,
} from "../lib/uploadClient";

type Phase = "idle" | "uploading" | "processing_check" | "done" | "error";

interface VideoUploaderProps {
  classSessionId: number;
}

/**
 * Record screen's upload step (Phase 2 deliverable 6; the camera-guidance
 * overlay itself is Phase 8's job per the UI contract's screen list).
 *
 * Follows the UI contract: sentence case, no exclamation marks, no
 * "successfully", errors state what happened and what to do next in one
 * sentence. "Still uploading" language, never "offline" as an error.
 */
export default function VideoUploader({ classSessionId }: VideoUploaderProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [result, setResult] = useState<UploadCompleteResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleFileSelected(file: File) {
    setPhase("uploading");
    setErrorMessage(null);
    setResult(null);

    try {
      const onProgress = (p: UploadProgress) => setProgress(p);
      setPhase("uploading");
      const response = await uploadVideoResumable(file, classSessionId, onProgress);
      setPhase(response.preflight.status === "fail" ? "error" : "done");
      setResult(response);
      if (response.preflight.status === "fail") {
        const failMessages = response.preflight.checks
          .filter((c) => c.severity === "fail")
          .map((c) => c.message)
          .join(" ");
        setErrorMessage(failMessages || "This video could not be used.");
      }
    } catch (err) {
      setPhase("error");
      if (err instanceof ApiError) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Still uploading in the background. This will keep retrying.");
      }
    }
  }

  return (
    <div className="max-w-md mx-auto p-4 space-y-4">
      <h1 className="text-title font-medium text-gray-900">Record</h1>

      {phase === "idle" && (
        <label className="flex items-center justify-center h-11 min-h-11 rounded border border-gray-300 text-body text-gray-700 cursor-pointer hover:bg-gray-50">
          Choose a video to upload
          <input
            type="file"
            accept="video/mp4,video/quicktime"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) void handleFileSelected(file);
            }}
          />
        </label>
      )}

      {phase === "uploading" && progress && (
        <div className="space-y-2">
          <p className="text-body text-gray-700">
            Still uploading ({progress.chunksReceived} of {progress.totalChunks} parts sent)
          </p>
          <div className="h-2 rounded bg-gray-200 overflow-hidden">
            <div
              className="h-full bg-gray-500"
              style={{ width: `${Math.round(progress.fractionComplete * 100)}%` }}
            />
          </div>
          <p className="text-body text-gray-500">
            You can close this tab. Upload continues and picks up where it left off.
          </p>
        </div>
      )}

      {phase === "done" && result && (
        <div className="rounded border border-gray-200 p-4 space-y-2">
          <p className="text-emphasis font-medium text-gray-900">Video received</p>
          <p className="text-body text-gray-700">
            {result.video_upload.width}x{result.video_upload.height},{" "}
            {Math.round(result.video_upload.duration_seconds)}s
          </p>
          {result.preflight.status === "warn" && (
            <ul className="text-body text-review space-y-1">
              {result.preflight.checks.map((check) => (
                <li key={check.code} className="flex items-start gap-2">
                  <span aria-hidden>&#9888;</span>
                  <span>{check.message}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {phase === "error" && (
        <div className="rounded border border-gray-200 p-4 space-y-3">
          <p className="text-body text-absent">{errorMessage}</p>
          <button
            type="button"
            className="h-11 min-h-11 px-4 rounded border border-gray-300 text-body text-gray-700"
            onClick={() => {
              setPhase("idle");
              setErrorMessage(null);
              setResult(null);
              setProgress(null);
            }}
          >
            Record again
          </button>
        </div>
      )}
    </div>
  );
}
