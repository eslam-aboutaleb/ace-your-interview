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
    docs_path: str = "docs"

    # Default LLM settings
    default_provider: str = "groq"      # Free tier friendly default
    default_model: str = "llama-3.3-70b-versatile"
    default_temperature: float = 0.7

    # Learning/adaptive configuration
    learning_db_path: str = "/tmp/study_hub_learning.db"
    enable_v2_generation: bool = True
    enable_adaptive_learning: bool = True

    # CORS
    cors_origins: str = "http://localhost:5174,http://localhost:5173,http://127.0.0.1:5174"

    # Auth
    auth_secret_key: str = ""
    frontend_url: str = "http://localhost:5174"
    dev_auth_bypass_localhost: bool = False
    github_client_id: str = ""
    github_client_secret: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    allowed_github_users: str = ""
    allowed_google_emails: str = ""

    model_config = {"env_prefix": "STUDY_", "env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
