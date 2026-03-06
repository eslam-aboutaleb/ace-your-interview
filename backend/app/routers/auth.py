"""OAuth login (GitHub + Google), session management, allowed-users CRUD."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.config import get_settings
from app.dependencies import require_auth
from app.services import auth as auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

_TEN_YEARS = 10 * 365 * 24 * 60 * 60


def _cookie_kwargs() -> dict:
    settings = get_settings()
    return dict(
        httponly=True,
        samesite="lax",
        secure=settings.frontend_url.startswith("https"),
        path="/",
    )


# ── GitHub OAuth ─────────────────────────────────────────────


@router.get("/github")
async def github_login():
    settings = get_settings()
    state = secrets.token_urlsafe(32)
    params = urlencode(
        {
            "client_id": settings.github_client_id,
            "redirect_uri": f"{settings.frontend_url}/api/auth/github/callback",
            "scope": "read:user",
            "state": state,
        }
    )
    response = RedirectResponse(
        url=f"https://github.com/login/oauth/authorize?{params}"
    )
    response.set_cookie(
        "oauth_state", state, httponly=True, max_age=600, samesite="lax", path="/"
    )
    return response


@router.get("/github/callback")
async def github_callback(request: Request, code: str, state: str = ""):
    settings = get_settings()
    stored = request.cookies.get("oauth_state")
    if not stored or stored != state:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=invalid_state"
        )

    try:
        username = await auth_service.exchange_github_code(code)
    except Exception as exc:
        logger.error("GitHub exchange failed: %s", exc)
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=exchange_failed"
        )

    if not auth_service.is_allowed("github", username):
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=access_denied"
        )

    token = auth_service.create_jwt_token(username.lower(), "github")
    response = RedirectResponse(url=f"{settings.frontend_url}/")
    response.set_cookie("session", token, max_age=_TEN_YEARS, **_cookie_kwargs())
    response.delete_cookie("oauth_state", path="/")
    return response


# ── Google OAuth ─────────────────────────────────────────────


@router.get("/google")
async def google_login():
    settings = get_settings()
    state = secrets.token_urlsafe(32)
    params = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": f"{settings.frontend_url}/api/auth/google/callback",
            "response_type": "code",
            "scope": "email profile",
            "access_type": "offline",
            "state": state,
        }
    )
    response = RedirectResponse(
        url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
    )
    response.set_cookie(
        "oauth_state", state, httponly=True, max_age=600, samesite="lax", path="/"
    )
    return response


@router.get("/google/callback")
async def google_callback(request: Request, code: str, state: str = ""):
    settings = get_settings()
    stored = request.cookies.get("oauth_state")
    if not stored or stored != state:
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=invalid_state"
        )

    redirect_uri = f"{settings.frontend_url}/api/auth/google/callback"
    try:
        email = await auth_service.exchange_google_code(code, redirect_uri)
    except Exception as exc:
        logger.error("Google exchange failed: %s", exc)
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=exchange_failed"
        )

    if not auth_service.is_allowed("google", email):
        return RedirectResponse(
            url=f"{settings.frontend_url}/login?error=access_denied"
        )

    token = auth_service.create_jwt_token(email.lower(), "google")
    response = RedirectResponse(url=f"{settings.frontend_url}/")
    response.set_cookie("session", token, max_age=_TEN_YEARS, **_cookie_kwargs())
    response.delete_cookie("oauth_state", path="/")
    return response


# ── Session helpers ──────────────────────────────────────────


@router.get("/me")
async def me(user: dict = Depends(require_auth)):
    return user


@router.post("/logout")
async def logout():
    response = JSONResponse(content={"ok": True})
    response.delete_cookie("session", path="/")
    return response


# ── Allowed-users management ─────────────────────────────────


class AllowedUserBody(BaseModel):
    provider: str  # "github" | "google"
    identifier: str  # username or email


@router.get("/allowed-users")
async def list_allowed_users(_user: dict = Depends(require_auth)):
    return auth_service.get_allowed_users()


@router.post("/allowed-users")
async def add_user(body: AllowedUserBody, _user: dict = Depends(require_auth)):
    return auth_service.add_allowed_user(body.provider, body.identifier)


@router.delete("/allowed-users/{provider}/{identifier:path}")
async def remove_user(
    provider: str, identifier: str, _user: dict = Depends(require_auth)
):
    return auth_service.remove_allowed_user(provider, identifier)
