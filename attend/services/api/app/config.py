"""All environment-driven configuration for the API lives here, and only here.

Non-negotiable rule #3 in the global brief is about pipeline tunables living in
one place (services/worker/pipeline/params.py). This module is the equivalent
rule for infrastructure config: no other module should read os.environ
directly. Import `settings` from here instead.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- identity / meta (used by the /health endpoint) ---
    app_name: str = "attend-api"
    app_version: str = "0.1.0"
    environment: str = Field(default="development")  # development | staging | production

    # --- database ---
    database_url: str = Field(
        default="postgresql+asyncpg://attend:attend@localhost:5432/attend",
        description="Async SQLAlchemy connection string. Must use the asyncpg driver.",
    )

    # --- job queue ---
    redis_url: str = Field(default="redis://localhost:6379/0")

    # --- object storage (S3-compatible: MinIO locally, Supabase Storage / S3 in prod) ---
    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_bucket: str = Field(default="attend-media")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")

    # --- pipeline job artifacts ---
    # Shared volume mounted identically into the api and worker containers
    # (see docker-compose.yml). The api writes upload manifests here; the
    # worker writes every stage's intermediate artifacts here.
    job_data_dir: str = Field(default="/data/jobs")

    # --- biometric data retention (non-negotiable rule #5) ---
    biometric_retention_days: int = Field(default=180, ge=1)

    # --- ML models (InsightFace) ---
    # Directory containing the downloaded .onnx model files (det_10g.onnx,
    # the recognition model -- see embed.py's ASSUMPTION docstring for the
    # exact filename to verify). insightface downloads here on first use if
    # nothing is already present; Phase 9 bakes these into the worker image
    # instead of downloading at deploy time.
    insightface_home: str = Field(default="/opt/models/insightface")

    # --- CORS ---
    cors_allowed_origins: str = Field(default="http://localhost:3000")

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached; env vars don't change mid-process."""
    return Settings()


settings = get_settings()
