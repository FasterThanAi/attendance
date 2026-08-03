"""Runs the pre-flight check by enqueueing it to the worker and polling for
the result, synchronously, from within the upload-complete request handler.

Why not just run it in-process: the pre-flight check needs face detection
(insightface/onnxruntime/opencv), which deliberately lives only in the
worker's Docker image (see services/worker/db.py's docstring for the same
"separate image, separate deps" reasoning applied elsewhere). Why not make
it "properly" async (return immediately, teacher polls a status endpoint):
the roadmap is explicit that pre-flight is supposed to feel synchronous --
the teacher is still standing in the classroom and needs an answer in
seconds, not a job to check back on. Polling a worker-executed RQ job from
inside an API request for ~10-25 seconds is a reasonable, honest way to get
both properties without duplicating ML code into the api image.
"""

from __future__ import annotations

import asyncio
import time

import redis
from rq import Queue
from rq.job import Job

from app.config import settings

POLL_INTERVAL_SECONDS = 0.5
MAX_WAIT_SECONDS = 28.0  # stay under the "under 30 seconds" target with margin


class PreflightTimeoutError(Exception):
    pass


async def run_preflight_and_wait(video_path: str, expected_students: int) -> dict:
    redis_conn = redis.from_url(settings.redis_url)
    queue = Queue("attend", connection=redis_conn)
    job = queue.enqueue_call(
        func="preflight_job.run_preflight",
        args=(video_path, expected_students),
        timeout=MAX_WAIT_SECONDS + 10,
    )

    deadline = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        job.refresh()
        if job.is_finished:
            return job.result
        if job.is_failed:
            raise RuntimeError(f"Pre-flight check job failed: {job.exc_info}")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise PreflightTimeoutError(
        f"Pre-flight check did not finish within {MAX_WAIT_SECONDS}s (job_id={job.id})."
    )
