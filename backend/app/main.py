"""Study-app FastAPI application."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings, resolve_auth_secret_key
from app.dependencies import require_auth
from app.routers import (
    auth,
    chat,
    interview_sessions,
    learning,
    llm_settings,
    questions,
    topics,
    user_settings,
    voice,
)
from app.services.doc_parser import DocParser
from app.services.llm_assignments_store import LLMAssignmentsStore
from app.services.http_clients import close_http_clients
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
    policy_error_detail,
)
from app.services.llm_service_access import LLMServiceAccess
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.mcp_gateway import MCPGateway
from app.services.rate_limit import InMemoryRateLimiter, classify_rate_limit_scope, request_ip
from app.services.user_settings_store import UserSettingsStore, resolve_credentials_encryption_secret
from app.services.auth import decode_jwt_token

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None
_learning_store: LearningStore | None = None
_learning_planner: LearningPlannerStore | None = None
_user_settings_store: UserSettingsStore | None = None
_llm_service_access: LLMServiceAccess | None = None
_llm_assignments_store: LLMAssignmentsStore | None = None
_mcp_gateway: MCPGateway | None = None
_rate_limiter = InMemoryRateLimiter()


def _rate_limit_rule(scope: str) -> tuple[int, int]:
    settings = get_settings()
    if scope == "oauth":
        return int(settings.oauth_rate_limit_requests), int(settings.oauth_rate_limit_window_seconds)
    if scope == "session":
        return int(settings.session_rate_limit_requests), int(settings.session_rate_limit_window_seconds)
    return int(settings.llm_rate_limit_requests), int(settings.llm_rate_limit_window_seconds)


def _rate_limit_key(scope: str, request: Request) -> str:
    ip = request_ip(request)
    if scope == "oauth":
        return f"ip:{ip}"
    session_token = request.cookies.get("session", "")
    if session_token:
        try:
            payload = decode_jwt_token(session_token)
            provider = str(payload.get("provider", "")).strip().lower()
            subject = str(payload.get("sub", "")).strip().lower()
            if subject:
                return f"user:{provider}:{subject}"
        except Exception:
            pass
    return f"ip:{ip}"


def _load_dotenv():
    """Load .env file into os.environ so LiteLLM picks up API keys."""
    for candidate in [".env", "../.env"]:
        env_path = Path(candidate).resolve()
        if env_path.is_file():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value
            logger.info("Loaded env vars from %s", env_path)
            return
    logger.warning("No .env file found — API keys must be set in the environment")


_load_dotenv()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle."""
    global _llm_client, _learning_store, _learning_planner, _user_settings_store, _llm_service_access, _llm_assignments_store, _mcp_gateway

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    resolve_auth_secret_key(settings)
    resolve_credentials_encryption_secret(settings)

    # Initialise shared services
    _llm_service_access = LLMServiceAccess()
    _llm_assignments_store = LLMAssignmentsStore()
    _user_settings_store = UserSettingsStore(
        llm_service_access=_llm_service_access,
        llm_assignments_store=_llm_assignments_store,
    )
    _llm_client = LLMClient(user_settings_store=_user_settings_store)
    _mcp_gateway = MCPGateway(settings)
    _learning_store = LearningStore(settings.learning_db_path)
    _learning_planner = LearningPlannerStore(settings.learning_db_path)
    parser = DocParser(curriculum_path=settings.curriculum_path)

    # Wire routers to shared instances
    questions.init(_llm_client, parser, _learning_store, _mcp_gateway)
    topics.init(parser, _llm_client, _learning_store, _mcp_gateway)
    llm_settings.init(_llm_client)
    user_settings.init(_user_settings_store)
    auth.init_llm_service_access(_llm_service_access)
    auth.init_llm_assignments_store(_llm_assignments_store)
    chat.init(_llm_client, parser, _learning_store, _mcp_gateway)
    learning.init(_learning_store, _learning_planner)
    interview_sessions.init(
        _llm_client,
        parser,
        _learning_store,
        settings.learning_db_path,
        _mcp_gateway,
    )
    voice.init(_llm_client)

    logger.info("Loaded %d topics from curriculum %s", len(parser.list_topics()), parser.source_path)

    try:
        yield  # Application runs
    finally:
        await close_http_clients()
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS – merge explicit origins + frontend_url so cookies work
    origins = {o.strip() for o in settings.cors_origins.split(",") if o.strip()}
    if settings.frontend_url:
        origins.add(settings.frontend_url.rstrip("/"))
    origins.discard("")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=sorted(origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def _apply_rate_limits(request: Request, call_next):
        settings = get_settings()
        if not settings.enable_rate_limiting:
            return await call_next(request)

        scope = classify_rate_limit_scope(request.url.path, request.method)
        if not scope:
            return await call_next(request)

        limit, window_seconds = _rate_limit_rule(scope)
        key = _rate_limit_key(scope, request)
        decision = _rate_limiter.check(
            scope=scope,
            key=key,
            limit=limit,
            window_seconds=window_seconds,
        )
        if decision.allowed:
            return await call_next(request)

        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(decision.retry_after_seconds)},
            content={
                "detail": {
                    "code": "rate_limit_exceeded",
                    "message": "Too many requests. Please retry later.",
                    "scope": scope,
                    "retry_after_seconds": decision.retry_after_seconds,
                }
            },
        )

    # Auth router — no auth dependency (contains login endpoints)
    application.include_router(auth.router)

    # Protected routers
    auth_dep = [Depends(require_auth)]
    application.include_router(topics.router)
    application.include_router(topics.features_router)
    application.include_router(questions.router, dependencies=auth_dep)
    application.include_router(llm_settings.router, dependencies=auth_dep)
    application.include_router(user_settings.router, dependencies=auth_dep)
    application.include_router(chat.router, dependencies=auth_dep)
    application.include_router(learning.router)
    application.include_router(interview_sessions.router, dependencies=auth_dep)
    application.include_router(voice.router)  # WebSocket handles its own auth

    @application.exception_handler(LLMServiceApprovalRequiredError)
    async def _handle_llm_service_access_denied(
        _request: Request,
        exc: LLMServiceApprovalRequiredError,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": policy_error_detail(exc),
            },
        )

    @application.exception_handler(StudyAppLLMNotAssignedError)
    async def _handle_study_app_not_assigned(
        _request: Request,
        exc: StudyAppLLMNotAssignedError,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": policy_error_detail(exc),
            },
        )

    @application.exception_handler(PersonalCredentialRequiredError)
    async def _handle_personal_credential_required(
        _request: Request,
        exc: PersonalCredentialRequiredError,
    ):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": policy_error_detail(exc),
            },
        )

    @application.get("/health")
    async def root_health():
        return {"status": "ok", "service": settings.app_name}

    return application


app = create_app()
