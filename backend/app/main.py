"""Study-app FastAPI application."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
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
)
from app.services.doc_parser import DocParser
from app.services.llm_policy import LLMServiceApprovalRequiredError
from app.services.llm_service_access import LLMServiceAccess
from app.services.learning_store import LearningStore
from app.services.llm_client import LLMClient
from app.services.user_settings_store import UserSettingsStore

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None
_learning_store: LearningStore | None = None
_user_settings_store: UserSettingsStore | None = None
_llm_service_access: LLMServiceAccess | None = None


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
    global _llm_client, _learning_store, _user_settings_store, _llm_service_access

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    if settings.environment.strip().lower() != "development" and not settings.credentials_encryption_key.strip():
        raise RuntimeError(
            "STUDY_CREDENTIALS_ENCRYPTION_KEY must be set when STUDY_ENVIRONMENT is not development"
        )

    # Initialise shared services
    _llm_service_access = LLMServiceAccess()
    _user_settings_store = UserSettingsStore(llm_service_access=_llm_service_access)
    _llm_client = LLMClient(user_settings_store=_user_settings_store)
    _learning_store = LearningStore(settings.learning_db_path)
    parser = DocParser(settings.docs_path)

    # Wire routers to shared instances
    questions.init(_llm_client, parser, _learning_store)
    topics.init(parser, _llm_client, _learning_store)
    llm_settings.init(_llm_client)
    user_settings.init(_user_settings_store)
    auth.init_llm_service_access(_llm_service_access)
    chat.init(_llm_client)
    learning.init(_learning_store)
    interview_sessions.init(
        _llm_client,
        parser,
        _learning_store,
        settings.learning_db_path,
    )

    logger.info("Loaded %d topics from %s", len(parser.list_topics()), settings.docs_path)

    yield  # Application runs

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

    # Auth router — no auth dependency (contains login endpoints)
    application.include_router(auth.router)

    # Protected routers
    auth_dep = [Depends(require_auth)]
    application.include_router(topics.router)
    application.include_router(questions.router, dependencies=auth_dep)
    application.include_router(llm_settings.router, dependencies=auth_dep)
    application.include_router(user_settings.router, dependencies=auth_dep)
    application.include_router(chat.router, dependencies=auth_dep)
    application.include_router(learning.router)
    application.include_router(interview_sessions.router, dependencies=auth_dep)

    @application.exception_handler(LLMServiceApprovalRequiredError)
    async def _handle_llm_service_access_denied(
        _request: Request,
        exc: LLMServiceApprovalRequiredError,
    ):
        return JSONResponse(
            status_code=403,
            content={
                "detail": {
                    "code": exc.code,
                    "message": exc.message,
                }
            },
        )

    @application.get("/health")
    async def root_health():
        return {"status": "ok", "service": settings.app_name}

    return application


app = create_app()
