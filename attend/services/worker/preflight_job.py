"""RQ entrypoint for the pre-flight check, enqueued by the api's
POST /uploads/{id}/complete and polled synchronously (see
app/services/preflight_client.py) so the ~30-second check still feels
synchronous to the teacher without putting ML dependencies in the api image.
"""

from __future__ import annotations

from pathlib import Path

from config import settings
from pipeline.detect import load_detector
from pipeline.params import PipelineParams
from pipeline.preflight import preflight_check


def run_preflight(video_path: str, expected_students: int) -> dict:
    detector = load_detector(Path(settings.insightface_home))
    params = PipelineParams()

    result = preflight_check(Path(video_path), expected_students, params, detector)

    return {
        "status": result.status,
        "checks": [
            {"code": c.code, "severity": c.severity, "message": c.message}
            for c in result.checks
        ],
    }
