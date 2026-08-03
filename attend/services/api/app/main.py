"""FastAPI entrypoint (deliverable 6, Phase 0).

Only wiring lives here: middleware, routers, exception handlers. Business
logic belongs in app/services/ (non-negotiable rule: "services/ business
logic, no FastAPI imports here" -- see the repository layout in the global
brief), not in this file.
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logging_config import configure_logging, request_id_ctx
from app.routers import attendance, enrollment, health, job, session, upload
from app.schemas.errors import ErrorResponse
from app.services.consent import ConsentError

configure_logging()
logger = logging.getLogger("attend.api")

app = FastAPI(title=settings.app_name, version=settings.app_version)

origins = [origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()]
if not origins:
    origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    incoming_id = request.headers.get("x-request-id")
    request_id = incoming_id or str(uuid.uuid4())
    token = request_id_ctx.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(ConsentError)
async def consent_error_handler(request: Request, exc: ConsentError) -> JSONResponse:
    logger.warning("consent check failed: code=%s student_id=%s", exc.code, exc.student_id)
    return JSONResponse(
        status_code=403,
        content=ErrorResponse(code=exc.code, message=str(exc)).model_dump(),
    )


app.include_router(health.router)
app.include_router(enrollment.router)
app.include_router(session.router)
app.include_router(upload.router)
app.include_router(job.router)
app.include_router(attendance.router)

# Phase 8 gap fix (see app/media.py's docstring): pipeline artifacts
# (enrollment photos, best-crop images) live on the same shared volume the
# worker writes to (settings.job_data_dir). Mounting it here is what makes
# app/media.py's rewritten "/media/..." URLs actually resolvable. check_dir
# is off so a fresh checkout with no jobs run yet (dir doesn't exist) still
# boots the API instead of crashing on startup.
app.mount("/media", StaticFiles(directory=settings.job_data_dir, check_dir=False), name="media")
