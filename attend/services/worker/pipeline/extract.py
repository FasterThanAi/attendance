"""Frame extraction via ffmpeg (subprocess, never moviepy -- global brief).

Built in full now (Phase 1 needs it for enrollment video; Phase 3 reuses this
exact function for classroom video, just with a different fps). One ffmpeg
invocation with the fps filter does the sampling -- no per-frame Python loop.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameManifest:
    frame_dir: Path
    frame_count: int
    fps: float
    source_width: int
    source_height: int


def _probe_dimensions(video_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    width_str, height_str = result.stdout.strip().split("x")
    return int(width_str), int(height_str)


def extract_frames(video_path: Path, out_dir: Path, fps: float) -> FrameManifest:
    """Sample `video_path` at `fps` frames per second into `out_dir` as full
    resolution JPEGs (quality 95), named frame_00001.jpg, frame_00002.jpg, ...

    Does NOT resize -- downscaling here would destroy exactly the pixels the
    rest of the pipeline depends on (see the roadmap's capture-protocol
    section: "Do not downscale before upload," which applies just as much to
    any resampling done server-side).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    width, height = _probe_dimensions(video_path)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-qscale:v", "2",  # ~quality 95 equivalent for mjpeg output
            str(out_dir / "frame_%05d.jpg"),
        ],
        capture_output=True, check=True,
    )

    frame_count = len(list(out_dir.glob("frame_*.jpg")))
    return FrameManifest(frame_dir=out_dir, frame_count=frame_count, fps=fps, source_width=width, source_height=height)
