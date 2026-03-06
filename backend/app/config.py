"""Study-app backend configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Polymarket Study Hub API"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8001

    # Docs path
    docs_path: str = "/app/docs"

    # Default LLM settings
    default_provider: str = "groq"      # Free tier friendly default
    default_model: str = "llama-3.3-70b-versatile"
    default_temperature: float = 0.7

    # CORS
    cors_origins: str = "http://localhost:5174,http://localhost:5173,http://127.0.0.1:5174"

    model_config = {"env_prefix": "STUDY_", "env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
