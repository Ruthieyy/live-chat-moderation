"""Process settings. Environment variables use the MODERATION_ prefix.

Cloud Run injects PORT, which is not prefixed; ``resolve_port`` honors it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_model_dir() -> Path:
    env = os.environ.get("MODERATION_MODEL_DIR")
    if env:
        return Path(env)
    bundled = Path(__file__).resolve().parents[2] / "models"
    if (bundled / "config.json").exists():
        return bundled
    return Path("models")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODERATION_", extra="ignore")

    model_dir: Path = Field(default_factory=default_model_dir)
    redis_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "info"
    max_text_chars: int = 2000
    sync_batch_limit: int = 16
    result_ttl_seconds: int = 300
    ort_intra_op_threads: int = 1
    max_concurrent_inference: int = 0
    worker_batch_size: int = 32
    worker_batch_wait_ms: int = 10
    worker_block_ms: int = 2000
    worker_metrics_port: int = 8001
    worker_consumer: str = ""
    worker_claim_idle_ms: int = 30_000
    drift_window: int = 2000
    drift_min_samples: int = 200
    host: str = "0.0.0.0"
    port: int = 8000
    access_log: bool = False

    @property
    def inference_slots(self) -> int:
        if self.max_concurrent_inference > 0:
            return self.max_concurrent_inference
        return max(1, os.cpu_count() or 1)


def load_settings() -> Settings:
    return Settings()


def resolve_port(settings: Settings) -> int:
    raw = os.environ.get("PORT")
    if raw:
        return int(raw)
    return settings.port
