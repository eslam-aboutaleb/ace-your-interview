"""Study-app backend configuration."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Ace Your Interview API"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    environment: str = "development"

    # Docs path
    docs_path: str = "docs"

    # Default LLM settings
    default_provider: str = "groq"      # Free tier friendly default
    default_model: str = "llama-3.3-70b-versatile"
    default_temperature: float = 0.7
    llm_task_routing_enabled: bool = False
    llm_task_route_map_json: str = ""

    # Learning/adaptive configuration
    learning_db_path: str = "/tmp/study_hub_learning.db"
    enable_v2_generation: bool = True
    enable_adaptive_learning: bool = True
    enable_mock_interview_v1: bool = False

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
    admin_users: str = ""
    credentials_encryption_key: str = ""
    user_settings_file: str = "user_llm_settings.json"
    llm_service_users_file: str = "llm_service_users.json"
    llm_assignments_file: str = "llm_assignments.json"

    # MCP gateway
    enable_mcp_gateway: bool = False
    mcp_rollout_stage: int = 1  # 1=custom+chat, 2=+questions+quiz, 3=+interview
    mcp_timeout_seconds: float = 8.0
    mcp_max_context_chars: int = 6000

    mcp_enable_custom_topic: bool = True
    mcp_enable_chat: bool = True
    mcp_enable_questions: bool = True
    mcp_enable_quiz: bool = True
    mcp_enable_interview: bool = True
    mcp_agentic_loop_enabled: bool = False
    mcp_agentic_max_steps: int = 2

    # Tavily
    tavily_api_key: str = ""
    mcp_tavily_enabled: bool = False

    # Firecrawl
    firecrawl_api_key: str = ""
    mcp_firecrawl_enabled: bool = False

    # GitHub (optional token for higher limits)
    github_token: str = ""
    mcp_github_enabled: bool = False

    model_config = {"env_prefix": "STUDY_", "env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
