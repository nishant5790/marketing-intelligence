"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Marketing Data Intelligence")
    app_env: Literal["development", "staging", "production"] = Field(default="development")
    debug: bool = Field(default=False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Gemini LLM
    gemini_api_key: str = Field(default="")

    # Qdrant Vector Database
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="products")

    # ML Model
    model_path: Path = Field(default=Path("models/discount_predictor.joblib"))
    drift_threshold: float = Field(default=0.1)
    retrain_trigger_count: int = Field(default=1000)

    # RAG
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    top_k_results: int = Field(default=5)
    similarity_threshold: float = Field(default=0.7)

    # Data
    data_path: Path = Field(default=Path("data/amazon.csv"))

    # Monitoring
    prometheus_port: int = Field(default=9090)
    enable_metrics: bool = Field(default=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Expose settings for easy import
settings = get_settings()
