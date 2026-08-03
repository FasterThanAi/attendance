"""Worker's own environment-driven configuration, mirroring the shape of
services/api/app/config.py but as a separate module -- same reasoning as
db.py's docstring: separate Docker image, separate dependency set, no cross-
package import. Keep the two in sync by hand when a shared setting changes;
this is a small, stable set of values, so the duplication cost is low.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = Field(default="development")

    database_url: str = Field(
        default="postgresql+asyncpg://attend:attend@localhost:5432/attend",
        description="Same DSN as the api service's DATABASE_URL; db.py converts "
        "the driver segment to sync psycopg2 itself, so this stays in the "
        "asyncpg-style form for consistency with the shared .env file.",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    job_data_dir: str = Field(default="/data/jobs")
    biometric_retention_days: int = Field(default=180, ge=1)

    insightface_home: str = Field(default="/opt/models/insightface")


@lru_cache
def get_settings() -> WorkerSettings:
    return WorkerSettings()


settings = get_settings()
