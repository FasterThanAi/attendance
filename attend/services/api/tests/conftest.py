"""Test database fixture (deliverable 9, Phase 0).

Uses an in-memory SQLite database via aiosqlite rather than a real Postgres
instance. This keeps the test suite fast and dependency-free for local runs
and CI, at the cost of not exercising Postgres-only behavior (e.g. native
constraint enforcement quirks). Every column type used in models.py
(BigInteger, String, Text, Boolean, Float, DateTime(timezone=True),
LargeBinary, and Enum(native_enum=False)) is chosen specifically to behave
the same way on both backends -- see the ASSUMPTIONS note at the top of
models.py. Integration-level testing against real Postgres happens via
docker-compose (`docker-compose up postgres` + `DATABASE_URL` pointed at it).
"""

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.models import Base


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
