import { mediaUrl } from "../../lib/api";

/**
 * Reusable tap-to-toggle photo grid for Not Found / Confirmed Present /
 * Unrecognised People. UI contract: 3 cols on phone, 6 on desktop, square
 * crops, 8px corners, no exclamation marks / uppercase.
 */
export interface PhotoGridItem {
  id: number;
  name: string;
  rollNumber: string | null;
  photoUri: string | null;
  selected: boolean;
}

interface PhotoGridProps {
  items: PhotoGridItem[];
  onToggle: (id: number) => void;
  selectedLabel: string;
}

export default function PhotoGrid({ items, onToggle, selectedLabel }: PhotoGridProps) {
  return (
    <div className="grid grid-cols-3 md:grid-cols-6 gap-3 px-4 py-3">
      {items.map((item) => (
        <button
          key={item.id}
          type="button"
          onClick={() => onToggle(item.id)}
          className={`text-left rounded-lg border-2 min-h-11 ${
            item.selected ? "border-review" : "border-transparent"
          }`}
          aria-pressed={item.selected}
        >
          <div className="relative aspect-square w-full rounded-lg bg-gray-100 overflow-hidden">
            {item.photoUri ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={mediaUrl(item.photoUri) ?? undefined}
                alt={`${item.name}'s enrollment photo`}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-body text-gray-400">
                No photo
              </div>
            )}
            {item.selected && (
              <span className="absolute top-1 right-1 rounded bg-review text-white text-[11px] px-1.5 py-0.5">
                {selectedLabel}
              </span>
            )}
          </div>
          <div className="text-[11px] text-gray-700 mt-1 truncate">{item.name}</div>
          {item.rollNumber && <div className="text-[11px] text-gray-500 truncate">{item.rollNumber}</div>}
        </button>
      ))}
    </div>
  );
}
