"""Async SQLAlchemy engine/session setup, used by both the API and by tests.

The worker (services/worker) intentionally does NOT import from here -- it
gets its own engine in Phase 2, because non-negotiable rule #1 says pipeline
stages must be independently testable without a database, and importing the
api app's session module would blur that boundary.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: `db: AsyncSession = Depends(get_db)`."""
    async with AsyncSessionLocal() as session:
        yield session
