import { mediaUrl } from "../../lib/api";

/**
 * A single labelled photo tile -- used side-by-side (enrollment photo vs.
 * today's detected crop) in the Needs Your Check card. Square crop per the
 * UI contract's review-grid rule, 8px corners.
 */
interface PhotoTileProps {
  label: string;
  src: string | null;
  alt: string;
}

export default function PhotoTile({ label, src, alt }: PhotoTileProps) {
  return (
    <div>
      <div className="aspect-square w-full rounded-lg bg-gray-100 overflow-hidden flex items-center justify-center">
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={mediaUrl(src) ?? undefined} alt={alt} className="w-full h-full object-cover" />
        ) : (
          <span className="text-body text-gray-400">No image</span>
        )}
      </div>
      <div className="text-body text-gray-500 mt-1">{label}</div>
    </div>
  );
}
