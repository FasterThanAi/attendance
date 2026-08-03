"""Face alignment: 5-point similarity transform to a canonical 112x112 crop.

Built in full now (Phase 1), not stubbed, because enrollment cannot embed a
face without an aligned crop, and the alignment math itself never changes
across phases -- Phase 4's `align_crops` (the batch, dataframe-driven,
memmap-writing version for classroom video) is a thin wrapper around the
same `align_face` function defined here.

The reference landmark positions below are the standard ArcFace/InsightFace
112x112 canonical positions (the same constants used by insightface's
`face_align.norm_crop`). Getting these wrong silently degrades every
embedding downstream, which is why there's a test asserting eyes land within
2px of the expected position for a synthetic, known input.
"""

from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace/InsightFace canonical reference points for a 112x112
# aligned crop, in (x, y) order: left eye, right eye, nose tip, left mouth
# corner, right mouth corner.
ARCFACE_REFERENCE_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    image_bgr: np.ndarray,
    landmarks: tuple[tuple[float, float], ...],
    output_size: int = 112,
) -> np.ndarray:
    """Warp `image_bgr` so the given 5 landmarks map onto the canonical
    ArcFace reference positions, producing an `output_size` x `output_size`
    BGR uint8 crop.

    `landmarks` must be in the same order as ARCFACE_REFERENCE_LANDMARKS_112
    (left eye, right eye, nose, left mouth corner, right mouth corner) -- this
    is the order insightface's SCRFD detector returns them in (see detect.py).
    """
    src = np.array(landmarks, dtype=np.float32)

    if output_size != 112:
        # Scale the reference points if a caller ever wants a different
        # output size; 112 is the only size ArcFace r100 actually accepts,
        # so this is defensive rather than expected to be exercised.
        scale = output_size / 112.0
        dst = ARCFACE_REFERENCE_LANDMARKS_112 * scale
    else:
        dst = ARCFACE_REFERENCE_LANDMARKS_112

    transform_matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if transform_matrix is None:
        raise ValueError("cv2.estimateAffinePartial2D failed to find a transform for the given landmarks")

    aligned = cv2.warpAffine(image_bgr, transform_matrix, (output_size, output_size), borderValue=0.0)
    return aligned
