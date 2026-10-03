"""Study-app backend configuration."""

from functools import lru_cache
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "Ace Your Interview API"
    app_version: str = "2.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    environment: str = "development"

    # Static curriculum source
    curriculum_path: str = "app/data/static_curriculum.json"

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
    enable_mock_interview_v1: bool = True
    enable_topic_videos: bool = False
    youtube_api_key: str = ""
    topic_videos_default_limit: int = 3
    topic_videos_cache_ttl_hours: int = 168
    topic_videos_disable_on_quota: bool = True

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
    allow_insecure_dev_encryption_fallback: bool = False
    insecure_dev_credentials_encryption_secret: str = ""
    user_settings_file: str = "user_llm_settings.json"
    llm_service_users_file: str = "llm_service_users.json"
    llm_assignments_file: str = "llm_assignments.json"
    allowed_users_file: str = "allowed_users.json"
    enable_rate_limiting: bool = True
    oauth_rate_limit_requests: int = 20
    oauth_rate_limit_window_seconds: int = 300
    session_rate_limit_requests: int = 90
    session_rate_limit_window_seconds: int = 60
    llm_rate_limit_requests: int = 30
    llm_rate_limit_window_seconds: int = 60
    # Per-LLM-call budget: bounds provider calls per user identity, so one
    # permitted HTTP request cannot fan out unbounded (0 disables the limit).
    llm_user_call_budget: int = 120
    llm_user_call_budget_window_seconds: int = 60
    llm_user_max_concurrency: int = 4
    # Trust X-Forwarded-For only when running behind a known proxy.
    trusted_proxy_enabled: bool = False

    # Pluggable rate limiter backend: "memory" (default, per-replica
    # state) or "redis" (shared sliding-window state across replicas,
    # via REDIS_URL). Strict mode fails startup when Redis is
    # unreachable instead of silently degrading to in-memory state.
    rate_limit_backend: str = "memory"
    rate_limit_strict: bool = False
    redis_url: str = Field(
        default="",
        validation_alias=AliasChoices("STUDY_REDIS_URL", "REDIS_URL"),
    )

    # Interview memory (Stage 3.2)
    memory_summary_max_chars: int = 3000
    memory_verbatim_turns: int = 4
    memory_answer_excerpt_chars: int = 320

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

    # Voice agent
    enable_voice_agent: bool = False
    voice_tiers_enabled: str = "browser"  # comma-separated: browser,cloud
    voice_default_tier: str = "browser"
    voice_stt_provider: str = "groq"       # groq or openai
    voice_stt_model: str = "whisper-large-v3"
    voice_tts_provider: str = "edge"       # edge (free) or openai
    voice_tts_voice: str = "en-US-AriaNeural"
    # Per-connection LLM turn budget for the voice WebSocket.
    voice_max_turns_per_connection: int = 40

    model_config = {
        "env_prefix": "STUDY_",
        "env_file": ".env",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


_WEAK_AUTH_SECRET_VALUES = {
    "",
    "changeme",
    "change-me",
    "default",
    "dev-secret",
    "insecure",
    "password",
    "replace-me",
    "secret",
    "test",
    "your-secret-key",
}


def resolve_auth_secret_key(settings: Any) -> str:
    """Validate and return the JWT signing secret."""
    secret = str(getattr(settings, "auth_secret_key", "") or "").strip()
    normalized = secret.lower()
    if not secret or len(secret) < 32 or normalized in _WEAK_AUTH_SECRET_VALUES:
        raise RuntimeError(
            "STUDY_AUTH_SECRET_KEY must be set to a strong value (at least 32 characters) "
            "and must not use common placeholder values."
        )
    return secret
