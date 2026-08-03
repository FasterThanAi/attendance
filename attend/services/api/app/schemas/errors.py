"""Structured error model (non-negotiable rule #7: no bare HTTPException strings).

Every error response from this API is this shape. `code` is machine-readable
and stable across releases -- frontends and tests should match on `code`,
never on `message` text.
"""

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: str
    message: str
