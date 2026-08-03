"use client";

import { useState } from "react";
import VideoUploader from "../../components/VideoUploader";

/**
 * Phase 2 testing entry point. Real navigation (teacher taps a class from
 * "today", camera-guidance overlay, capture instructions) is Phase 8's job
 * per the UI contract -- this just lets you pick a session id to test
 * against until that exists.
 */
export default function RecordPage() {
  const [classSessionId, setClassSessionId] = useState(1);

  return (
    <main className="space-y-4">
      <div className="max-w-md mx-auto p-4 pb-0">
        <label className="block text-body text-gray-600 mb-1" htmlFor="session-id">
          Class session id (temporary, for testing)
        </label>
        <input
          id="session-id"
          type="number"
          className="h-11 min-h-11 w-full rounded border border-gray-300 px-3 text-body"
          value={classSessionId}
          onChange={(e) => setClassSessionId(Number(e.target.value))}
        />
      </div>
      <VideoUploader classSessionId={classSessionId} />
    </main>
  );
}
