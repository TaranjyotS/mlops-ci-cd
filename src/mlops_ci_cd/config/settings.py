from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    app_name: str = "MLOps CI/CD Inference API"
    app_version: str = "1.0.0"
    model_path: Path = Path("models/model.joblib")
    model_uri: str | None = None
    mlflow_tracking_uri: str | None = None
    prediction_threshold: float = 0.5
    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=("settings_",))


@lru_cache
def get_settings() -> Settings:
    return Settings()
