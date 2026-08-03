import redis.asyncio as redis_async
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.session import get_db
from app.schemas.health import DependencyStatus, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    db_status = await _check_database(db)
    redis_status = await _check_redis()

    return HealthResponse(
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        database=db_status,
        redis=redis_status,
    )


async def _check_database(db: AsyncSession) -> DependencyStatus:
    try:
        await db.execute(text("SELECT 1"))
        return DependencyStatus(ok=True)
    except Exception as exc:  # noqa: BLE001 -- health check must never crash the endpoint
        return DependencyStatus(ok=False, detail=str(exc))


async def _check_redis() -> DependencyStatus:
    client = redis_async.from_url(settings.redis_url)
    try:
        await client.ping()
        return DependencyStatus(ok=True)
    except Exception as exc:  # noqa: BLE001
        return DependencyStatus(ok=False, detail=str(exc))
    finally:
        await client.aclose()
