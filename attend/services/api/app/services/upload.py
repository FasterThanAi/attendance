"""Resumable chunked upload (Phase 2 deliverable 1-2).

Chunk state lives on disk under JOB_DATA_DIR/uploads/{upload_id}/, not in the
database -- an in-progress upload isn't a real video_upload row yet (that's
only created once assembly + ffprobe validation succeed), and tracking
"which of N chunks have arrived" is exactly the kind of transient state a
filesystem marker handles more simply than a DB table + migration would.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.config import settings

CHUNK_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB, per the Phase 2 prompt

MIN_DURATION_SECONDS = 20.0
MAX_DURATION_SECONDS = 5 * 60.0
MIN_SHORTER_DIMENSION_PX = 1080


class UploadValidationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _upload_dir(upload_id: str) -> Path:
    return Path(settings.job_data_dir) / "uploads" / upload_id


def _manifest_path(upload_id: str) -> Path:
    return _upload_dir(upload_id) / "manifest.json"


def create_upload_session(class_session_id: int, filename: str, total_size_bytes: int) -> dict:
    upload_id = uuid.uuid4().hex
    upload_dir = _upload_dir(upload_id)
    (upload_dir / "chunks").mkdir(parents=True, exist_ok=True)

    total_chunks = max(1, -(-total_size_bytes // CHUNK_SIZE_BYTES))  # ceil division

    manifest = {
        "upload_id": upload_id,
        "class_session_id": class_session_id,
        "filename": filename,
        "total_size_bytes": total_size_bytes,
        "chunk_size_bytes": CHUNK_SIZE_BYTES,
        "total_chunks": total_chunks,
    }
    _manifest_path(upload_id).write_text(json.dumps(manifest))

    return {"upload_id": upload_id, "chunk_size_bytes": CHUNK_SIZE_BYTES, "total_chunks": total_chunks}


def _load_manifest(upload_id: str) -> dict:
    manifest_path = _manifest_path(upload_id)
    if not manifest_path.exists():
        raise UploadValidationError("upload_not_found", f"No upload session with id={upload_id}.")
    return json.loads(manifest_path.read_text())


def get_manifest(upload_id: str) -> dict:
    """Public accessor for callers outside this module (routers/upload.py
    needs class_session_id after assembly completes).
    """
    return _load_manifest(upload_id)


def write_chunk(upload_id: str, chunk_index: int, data: bytes) -> None:
    manifest = _load_manifest(upload_id)
    if not (0 <= chunk_index < manifest["total_chunks"]):
        raise UploadValidationError(
            "invalid_chunk_index",
            f"chunk_index {chunk_index} out of range for {manifest['total_chunks']} total chunks.",
        )
    chunk_path = _upload_dir(upload_id) / "chunks" / f"{chunk_index:06d}.part"
    # Overwriting an already-received chunk is a no-op from the client's
    # perspective (same bytes go to the same path) -- this IS the idempotency
    # the Phase 2 prompt asks for: "PUT of an already-received chunk must
    # return 200 and change nothing... the client will retry."
    chunk_path.write_bytes(data)


def get_upload_status(upload_id: str) -> dict:
    manifest = _load_manifest(upload_id)
    chunks_dir = _upload_dir(upload_id) / "chunks"
    received = sorted(
        int(p.stem) for p in chunks_dir.glob("*.part")
    )
    is_complete = len(received) == manifest["total_chunks"] and received == list(range(manifest["total_chunks"]))
    return {"upload_id": upload_id, "total_chunks": manifest["total_chunks"], "received_chunks": received, "is_complete": is_complete}


@dataclass(frozen=True)
class ProbedVideo:
    duration_seconds: float
    width: int
    height: int
    fps: float
    bytes: int


def _probe_video(path: Path) -> ProbedVideo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration:stream=width,height,r_frame_rate",
            "-of", "json",
            str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise UploadValidationError(
            "unreadable_video",
            "This file could not be read as a video. Re-record and try again.",
        )

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    duration = float(data["format"]["duration"])
    width = int(stream["width"])
    height = int(stream["height"])

    num_str, _, den_str = stream["r_frame_rate"].partition("/")
    fps = float(num_str) / float(den_str or 1)

    return ProbedVideo(duration_seconds=duration, width=width, height=height, fps=fps, bytes=path.stat().st_size)


def assemble_and_validate(upload_id: str) -> tuple[Path, ProbedVideo]:
    """Concatenates chunks in order, then validates the result with ffprobe.

    Raises UploadValidationError with a plain-language message (never a raw
    exception string, per non-negotiable rule #7) if the video fails any
    hard technical check. Does NOT create the video_upload DB row -- that's
    the router's job, once this returns successfully.
    """
    manifest = _load_manifest(upload_id)
    status = get_upload_status(upload_id)
    if not status["is_complete"]:
        missing = sorted(set(range(manifest["total_chunks"])) - set(status["received_chunks"]))
        raise UploadValidationError(
            "chunks_missing",
            f"Upload is incomplete -- missing chunk(s): {missing[:10]}{'...' if len(missing) > 10 else ''}.",
        )

    assembled_dir = _upload_dir(upload_id) / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)
    assembled_path = assembled_dir / manifest["filename"]

    with open(assembled_path, "wb") as out_f:
        for i in range(manifest["total_chunks"]):
            chunk_path = _upload_dir(upload_id) / "chunks" / f"{i:06d}.part"
            out_f.write(chunk_path.read_bytes())

    probed = _probe_video(assembled_path)

    if probed.duration_seconds < MIN_DURATION_SECONDS:
        raise UploadValidationError(
            "video_too_short",
            f"This video is too short ({probed.duration_seconds:.0f}s). "
            "Record at least 30 seconds, panning slowly across the room.",
        )
    if probed.duration_seconds > MAX_DURATION_SECONDS:
        raise UploadValidationError(
            "video_too_long",
            f"This video is too long ({probed.duration_seconds:.0f}s, max {int(MAX_DURATION_SECONDS)}s).",
        )
    if min(probed.width, probed.height) < MIN_SHORTER_DIMENSION_PX:
        raise UploadValidationError(
            "resolution_too_low",
            f"Record in 4K. This video's shorter side is {min(probed.width, probed.height)}px "
            f"(need at least {MIN_SHORTER_DIMENSION_PX}px). Change it in your camera settings and try again.",
        )

    return assembled_path, probed
