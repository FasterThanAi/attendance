import Link from "next/link";

/**
 * Placeholder home screen. The real "Today" screen (list of the teacher's
 * classes for today, each with status) is Phase 8-9's job per the UI
 * contract's screen list -- this just gets you to the Phase 2 upload flow
 * for testing.
 */
export default function HomePage() {
  return (
    <main className="max-w-md mx-auto p-4 space-y-4">
      <h1 className="text-title font-medium">Attend</h1>
      <p className="text-body text-gray-600">
        The full "today" screen comes later. For now, use record to try the upload flow.
      </p>
      <Link
        href="/record"
        className="inline-flex items-center justify-center h-11 min-h-11 px-4 rounded border border-gray-300 text-body"
      >
        Record
      </Link>
    </main>
  );
}
