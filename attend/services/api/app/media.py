"""Phase 8 gap fix: turning a raw pipeline artifact path into a browser-loadable URL.

Pipeline artifacts (gallery photos from enrollment.py, best-crop images from
cluster.py) are written to a shared filesystem volume -- settings.job_data_dir,
e.g. "/data/jobs" -- and that raw absolute path is exactly what gets stored
in gallery_photo.storage_uri / detected_cluster.best_crop_uri. That's the
right thing for worker-to-worker file access, but it is not something a
browser's <img src> can ever load: it's a path inside the api/worker
containers' filesystem, not a URL.

This was invisible through Phase 0-7 because nothing rendered these images
in a browser yet -- Phase 8's review screen is the first consumer that does,
which is why the gap only surfaced now (as literal 404s for
"http://localhost:3000/data/jobs/...", the frontend origin, not even the
api's).

Fix: main.py mounts StaticFiles at /media against the same job_data_dir
directory, and every response that used to hand back a raw storage_uri now
hands back to_media_url(storage_uri) instead -- a path like
"/media/enrollment/1/frontal_x.jpg" that the frontend resolves against the
api's own base URL (see apps/web/lib/api.ts's mediaUrl helper).
"""

from __future__ import annotations

from app.config import settings


def to_media_url(storage_uri: str | None) -> str | None:
    if not storage_uri:
        return storage_uri
    prefix = settings.job_data_dir.rstrip("/")
    if storage_uri.startswith(prefix):
        return "/media" + storage_uri[len(prefix):]
    # Doesn't start with job_data_dir -- either already a URL (a future
    # S3/http storage_uri) or some other shape. Return unchanged rather
    # than guessing at a rewrite that might be wrong.
    return storage_uri
