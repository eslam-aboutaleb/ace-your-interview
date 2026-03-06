"""Study-app FastAPI application."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import auth, chat, llm_settings, questions, topics
from app.services.doc_parser import DocParser
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_llm_client: LLMClient | None = None


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
    global _llm_client

    settings = get_settings()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # Initialise shared services
    _llm_client = LLMClient()
    parser = DocParser(settings.docs_path)

    # Wire routers to shared instances
    questions.init(_llm_client, parser)
    llm_settings.init(_llm_client)
    chat.init(_llm_client)

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

    # CORS
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Auth router — no auth dependency (contains login endpoints)
    application.include_router(auth.router)

    # Protected routers
    auth_dep = [Depends(require_auth)]
    application.include_router(topics.router, dependencies=auth_dep)
    application.include_router(questions.router, dependencies=auth_dep)
    application.include_router(llm_settings.router, dependencies=auth_dep)
    application.include_router(chat.router, dependencies=auth_dep)

    @application.get("/health")
    async def root_health():
        return {"status": "ok", "service": settings.app_name}

    return application


app = create_app()
