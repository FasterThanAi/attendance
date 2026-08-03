import type { DraftSessionSummary } from "../../lib/reviewApi";

/**
 * Sticky header: the four STATIC partition numbers (always sum to
 * total_enrolled -- see lib/useReviewState.ts's docstring) plus the ONE
 * live number, the attendance percentage, which updates as the teacher
 * works. session_health is shown as color + text together, never color
 * alone, per accessibility.
 */
interface SummaryHeaderProps {
  summary: DraftSessionSummary;
  livePresentCount: number;
  livePercent: number;
}

const HEALTH_LABEL: Record<string, string> = { good: "good", fair: "fair", poor: "needs attention" };
const HEALTH_COLOR: Record<string, string> = { good: "text-confirmed", fair: "text-review", poor: "text-absent" };

export default function SummaryHeader({ summary, livePresentCount, livePercent }: SummaryHeaderProps) {
  const healthLabel = HEALTH_LABEL[summary.session_health] ?? summary.session_health;
  const healthColor = HEALTH_COLOR[summary.session_health] ?? "text-gray-500";

  return (
    <div className="sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-title font-medium text-gray-900">Review attendance</span>
        <span className={`text-body font-medium ${healthColor}`}>Capture quality: {healthLabel}</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Total enrolled" value={summary.total_enrolled} />
        <Stat label="Confirmed present" value={summary.proposed_present} colorClass="text-confirmed" />
        <Stat label="Need your check" value={summary.needs_review} colorClass="text-review" />
        <Stat label="Not found" value={summary.proposed_absent} colorClass="text-absent" />
      </div>
      <div className="flex items-center gap-2 pt-1">
        <div
          className="flex-1 h-2 rounded bg-gray-100 overflow-hidden"
          role="progressbar"
          aria-valuenow={livePercent}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div className="h-full bg-confirmed" style={{ width: `${livePercent}%` }} />
        </div>
        <span className="text-body text-gray-600 whitespace-nowrap">
          {livePresentCount} of {summary.total_enrolled} present ({livePercent}%)
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value, colorClass }: { label: string; value: number; colorClass?: string }) {
  return (
    <div className="rounded-lg border border-gray-200 px-3 py-2">
      <div className="text-body text-gray-500">{label}</div>
      <div className={`text-title font-medium ${colorClass ?? "text-gray-900"}`}>{value}</div>
    </div>
  );
}
