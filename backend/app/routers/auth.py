"""OAuth login (GitHub + Google), session management, allowed-users CRUD."""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.dependencies import is_admin_identity, require_admin, require_auth
from app.schemas.models import (
    LLMMyAssignmentResponse,
    LLMUserAssignmentItem,
    LLMUserAssignmentsResponse,
    LLMUserAssignmentUpdateRequest,
    StudyAppAssignment,
)
from app.services import auth as auth_service
from app.services.llm_assignments_store import LLMAssignmentsStore, normalise_identity
from app.services.llm_service_access import LLMServiceAccess

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

_TEN_YEARS = 10 * 365 * 24 * 60 * 60
_LLM_SERVICE_ACCESS: LLMServiceAccess | None = None
_LLM_ASSIGNMENTS: LLMAssignmentsStore | None = None


def init_llm_service_access(store: LLMServiceAccess):
    global _LLM_SERVICE_ACCESS
    _LLM_SERVICE_ACCESS = store


def init_llm_assignments_store(store: LLMAssignmentsStore):
    global _LLM_ASSIGNMENTS
    _LLM_ASSIGNMENTS = store


def _get_llm_service_access() -> LLMServiceAccess:
    global _LLM_SERVICE_ACCESS
    if _LLM_SERVICE_ACCESS is None:
        _LLM_SERVICE_ACCESS = LLMServiceAccess()
    return _LLM_SERVICE_ACCESS


def _get_llm_assignments() -> LLMAssignmentsStore:
    global _LLM_ASSIGNMENTS
    if _LLM_ASSIGNMENTS is None:
        _LLM_ASSIGNMENTS = LLMAssignmentsStore()
    return _LLM_ASSIGNMENTS


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
    return {
        **user,
        "is_admin": is_admin_identity(
            user=user.get("user", ""),
            provider=user.get("provider", ""),
        ),
    }


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
async def list_allowed_users(_user: dict = Depends(require_admin)):
    return auth_service.get_allowed_users()


@router.post("/allowed-users")
async def add_user(body: AllowedUserBody, _user: dict = Depends(require_admin)):
    return auth_service.add_allowed_user(body.provider, body.identifier)


@router.delete("/allowed-users/{provider}/{identifier:path}")
async def remove_user(
    provider: str, identifier: str, _user: dict = Depends(require_admin)
):
    return auth_service.remove_allowed_user(provider, identifier)


# ── LLM-service allowlist management ────────────────────────


class LLMServiceUserBody(BaseModel):
    provider: str
    identifier: str


@router.get("/llm-service-users")
async def list_llm_service_users(_user: dict = Depends(require_admin)):
    return _get_llm_service_access().list_users()


@router.post("/llm-service-users")
async def add_llm_service_user(
    body: LLMServiceUserBody,
    _user: dict = Depends(require_admin),
):
    return _get_llm_service_access().add_user(body.provider, body.identifier)


@router.delete("/llm-service-users/{provider}/{identifier:path}")
async def remove_llm_service_user(
    provider: str,
    identifier: str,
    _user: dict = Depends(require_admin),
):
    return _get_llm_service_access().remove_user(provider, identifier)


# ── Study App LLM assignments ───────────────────────────────


class IdentityBody(BaseModel):
    provider: str = Field(..., pattern="^(google|github)$")
    identifier: str


def _normalise_login_identity(provider: str, identifier: str) -> tuple[str, str, str]:
    p = (provider or "").strip().lower()
    i = (identifier or "").strip().lower()
    if p not in {"google", "github"} or not i:
        return "", "", ""
    return p, i, normalise_identity(p, i)


def _allowed_login_identities() -> list[tuple[str, str, str]]:
    allowed = auth_service.get_allowed_users()
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for provider, values in (
        ("github", allowed.get("github_users", [])),
        ("google", allowed.get("google_emails", [])),
    ):
        for identifier_raw in values:
            identifier = str(identifier_raw or "").strip().lower()
            identity_key = normalise_identity(provider, identifier)
            if not identity_key or identity_key in seen:
                continue
            seen.add(identity_key)
            out.append((provider, identifier, identity_key))
    return out


def _build_assignment_item(login_provider: str, identifier: str) -> LLMUserAssignmentItem:
    identity_key = normalise_identity(login_provider, identifier)
    assignment = _get_llm_assignments().get_assignment(identity_key=identity_key)
    assignment_payload = (
        StudyAppAssignment(
            provider=assignment.provider,
            model=assignment.model,
            updated_at=assignment.updated_at,
        )
        if assignment
        else None
    )
    approved = _get_llm_service_access().is_user_allowed(
        user=identifier,
        provider=login_provider,
    )
    return LLMUserAssignmentItem(
        login_provider=login_provider,
        identifier=identifier,
        identity_key=identity_key,
        is_backend_approved=approved,
        assignment=assignment_payload,
    )


@router.get("/llm-assignments/users", response_model=LLMUserAssignmentsResponse)
async def list_llm_assignment_users(_user: dict = Depends(require_admin)):
    users = [
        _build_assignment_item(provider, identifier)
        for provider, identifier, _identity_key in _allowed_login_identities()
    ]
    return LLMUserAssignmentsResponse(
        users=users,
        provider_models=_get_llm_assignments().provider_models(),
    )


@router.put(
    "/llm-assignments/{provider}/{identifier:path}",
    response_model=LLMUserAssignmentItem,
)
async def upsert_llm_assignment_for_user(
    provider: str,
    identifier: str,
    body: LLMUserAssignmentUpdateRequest,
    _user: dict = Depends(require_admin),
):
    login_provider, login_identifier, identity_key = _normalise_login_identity(provider, identifier)
    if not identity_key:
        raise HTTPException(status_code=400, detail="Invalid user identity")
    try:
        _get_llm_assignments().set_assignment(
            identity_key=identity_key,
            provider=body.provider,
            model=body.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _build_assignment_item(login_provider, login_identifier)


@router.delete(
    "/llm-assignments/{provider}/{identifier:path}",
    response_model=LLMUserAssignmentItem,
)
async def delete_llm_assignment_for_user(
    provider: str,
    identifier: str,
    _user: dict = Depends(require_admin),
):
    login_provider, login_identifier, identity_key = _normalise_login_identity(provider, identifier)
    if not identity_key:
        raise HTTPException(status_code=400, detail="Invalid user identity")
    _get_llm_assignments().delete_assignment(identity_key=identity_key)
    return _build_assignment_item(login_provider, login_identifier)


@router.get("/llm-assignments/me", response_model=LLMMyAssignmentResponse)
async def get_my_llm_assignment(user: dict = Depends(require_admin)):
    provider = str(user.get("provider", "")).strip().lower()
    identifier = str(user.get("user", "")).strip().lower()
    item = _build_assignment_item(provider, identifier)
    return LLMMyAssignmentResponse(
        login_provider=item.login_provider,
        identifier=item.identifier,
        identity_key=item.identity_key,
        is_backend_approved=item.is_backend_approved,
        assignment=item.assignment,
        provider_models=_get_llm_assignments().provider_models(),
    )


@router.put("/llm-assignments/me", response_model=LLMMyAssignmentResponse)
async def upsert_my_llm_assignment(
    body: LLMUserAssignmentUpdateRequest,
    user: dict = Depends(require_admin),
):
    provider = str(user.get("provider", "")).strip().lower()
    identifier = str(user.get("user", "")).strip().lower()
    identity_key = normalise_identity(provider, identifier)
    try:
        _get_llm_assignments().set_assignment(
            identity_key=identity_key,
            provider=body.provider,
            model=body.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    item = _build_assignment_item(provider, identifier)
    return LLMMyAssignmentResponse(
        login_provider=item.login_provider,
        identifier=item.identifier,
        identity_key=item.identity_key,
        is_backend_approved=item.is_backend_approved,
        assignment=item.assignment,
        provider_models=_get_llm_assignments().provider_models(),
    )
