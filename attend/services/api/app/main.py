"""FastAPI entrypoint (deliverable 6, Phase 0).

Only wiring lives here: middleware, routers, exception handlers. Business
logic belongs in app/services/ (non-negotiable rule: "services/ business
logic, no FastAPI imports here" -- see the repository layout in the global
brief), not in this file.
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import configure_logging, request_id_ctx
from app.routers import enrollment, health
from app.schemas.errors import ErrorResponse
from app.services.consent import ConsentError

configure_logging()
logger = logging.getLogger("attend.api")

app = FastAPI(title=settings.app_name, version=settings.app_version)


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
