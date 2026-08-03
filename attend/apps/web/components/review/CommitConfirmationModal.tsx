"use client";

/**
 * Distinct confirmation step (UI contract core principle 4: "destructive
 * and irreversible actions require a distinct confirmation... must feel
 * like it") -- not a browser confirm(), a full-screen step restating the
 * final counts and stating plainly that the record can only be corrected
 * afterward, not edited.
 */
interface CommitConfirmationModalProps {
  presentCount: number;
  absentCount: number;
  totalEnrolled: number;
  isSubmitting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function CommitConfirmationModal({
  presentCount,
  absentCount,
  totalEnrolled,
  isSubmitting,
  onConfirm,
  onCancel,
}: CommitConfirmationModalProps) {
  return (
    <div className="fixed inset-0 z-20 bg-black/40 flex items-end md:items-center justify-center px-4 pb-4 md:pb-0">
      <div className="bg-white rounded-lg w-full max-w-md p-4 space-y-4">
        <div>
          <h2 className="text-title font-medium text-gray-900">Confirm attendance</h2>
          <p className="text-body text-gray-600 mt-1">
            This will record attendance for this class. You can still correct an individual
            student afterward, but the record itself will not be editable as a whole.
          </p>
        </div>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg border border-gray-200 px-2 py-2">
            <div className="text-title font-medium text-gray-900">{totalEnrolled}</div>
            <div className="text-body text-gray-500">enrolled</div>
          </div>
          <div className="rounded-lg border border-gray-200 px-2 py-2">
            <div className="text-title font-medium text-confirmed">{presentCount}</div>
            <div className="text-body text-gray-500">present</div>
          </div>
          <div className="rounded-lg border border-gray-200 px-2 py-2">
            <div className="text-title font-medium text-absent">{absentCount}</div>
            <div className="text-body text-gray-500">absent</div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={isSubmitting}
            className="min-h-11 rounded-lg border border-gray-300 text-gray-700 text-emphasis font-medium py-3 disabled:opacity-50"
          >
            Go back
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isSubmitting}
            className="min-h-11 rounded-lg bg-confirmed text-white text-emphasis font-medium py-3 disabled:opacity-50"
          >
            {isSubmitting ? "Recording..." : "Confirm"}
          </button>
        </div>
      </div>
    </div>
  );
}
