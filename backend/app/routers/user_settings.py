"""User-level LLM settings and personal credentials."""

from __future__ import annotations

import os
import secrets
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    ProviderConnectionStatus,
    StudyAppAssignment,
    UserApiKeyUpdateRequest,
    UserPreferences,
    UserPreferencesUpdateRequest,
    UserSettingsResponse,
)
from app.services.llm_policy import APPROVAL_REQUIRED_CODE, APPROVAL_REQUIRED_MESSAGE
from app.services.user_settings_store import UserSettingsStore, identity_key_for_user

router = APIRouter(prefix="/api/user-settings", tags=["user-settings"])

_STORE: UserSettingsStore | None = None

_PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}


def init(store: UserSettingsStore):
    global _STORE
    _STORE = store


def _ensure_store() -> UserSettingsStore:
    if _STORE is None:
        raise HTTPException(status_code=503, detail="User settings store not initialised")
    return _STORE


def _backend_provider_configured(provider: str) -> bool:
    env_key = _PROVIDER_ENV_KEYS.get(provider, "")
    return bool(env_key and os.getenv(env_key))


def _cookie_kwargs() -> dict:
    settings = get_settings()
    return dict(
        httponly=True,
        samesite="lax",
        secure=settings.frontend_url.startswith("https"),
        path="/",
        max_age=600,
    )


def _build_response(identity_key: str, store: UserSettingsStore) -> UserSettingsResponse:
    prefs_dict, has_saved = store.get_user_state(identity_key)
    prefs = UserPreferences(**prefs_dict)
    backend_fallback_approved = store.is_identity_approved_for_backend_fallback(identity_key)
    study_app_assignment_raw = store.get_study_app_assignment(identity_key)
    study_app_assignment = (
        StudyAppAssignment(**study_app_assignment_raw)
        if study_app_assignment_raw
        else None
    )

    providers: list[ProviderConnectionStatus] = []
    for provider, models in store.provider_models().items():
        api_key_connected = store.has_api_key(identity_key, provider)
        account_connected = provider == "google" and store.get_google_oauth_status(identity_key)
        has_personal_credential = api_key_connected or account_connected
        backend_fallback_eligible = (
            _backend_provider_configured(provider)
            and backend_fallback_approved
        )
        using_backend_fallback = (
            prefs.llm_source.value == "study_app"
            and backend_fallback_eligible
            and bool(study_app_assignment)
            and study_app_assignment.provider == provider
        )
        providers.append(
            ProviderConnectionStatus(
                provider=provider,
                models=models,
                supports_account_connect=(provider == "google"),
                api_key_connected=api_key_connected,
                account_connected=account_connected,
                using_backend_fallback=using_backend_fallback,
                backend_fallback_eligible=backend_fallback_eligible,
            )
        )

    return UserSettingsResponse(
        has_saved_preferences=has_saved,
        preferences=prefs,
        providers=providers,
        study_app_available=backend_fallback_approved,
        study_app_assignment=study_app_assignment,
    )


@router.get("", response_model=UserSettingsResponse)
async def get_user_settings(user: dict = Depends(require_auth)):
    store = _ensure_store()
    identity_key = identity_key_for_user(user)
    return _build_response(identity_key, store)


@router.put("/preferences", response_model=UserPreferences)
async def update_preferences(
    body: UserPreferencesUpdateRequest,
    user: dict = Depends(require_auth),
):
    if body.llm_source.value == "study_app":
        identity_key = identity_key_for_user(user)
        store = _ensure_store()
        if not store.is_identity_approved_for_backend_fallback(identity_key):
            raise HTTPException(
                status_code=403,
                detail={
                    "code": APPROVAL_REQUIRED_CODE,
                    "message": APPROVAL_REQUIRED_MESSAGE,
                },
            )

    if body.auth_mode.value == "account" and body.provider != "google":
        raise HTTPException(
            status_code=422,
            detail="auth_mode=account is only supported for provider=google",
        )

    store = _ensure_store()
    identity_key = identity_key_for_user(user)
    prefs = store.save_preferences(identity_key, body.model_dump())
    return UserPreferences(**prefs)


@router.put("/api-key/{provider}")
async def save_api_key(
    provider: str,
    body: UserApiKeyUpdateRequest,
    user: dict = Depends(require_auth),
):
    store = _ensure_store()
    identity_key = identity_key_for_user(user)
    try:
        store.set_api_key(identity_key, provider=provider, api_key=body.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _build_response(identity_key, store)


@router.delete("/api-key/{provider}")
async def delete_api_key(provider: str, user: dict = Depends(require_auth)):
    store = _ensure_store()
    identity_key = identity_key_for_user(user)
    store.delete_api_key(identity_key, provider=provider)
    return _build_response(identity_key, store)


@router.get("/google/connect")
async def connect_google(user: dict = Depends(require_auth)):
    settings = get_settings()
    if not settings.google_client_id:
        raise HTTPException(status_code=400, detail="Google OAuth is not configured")

    state = secrets.token_urlsafe(32)
    redirect_uri = f"{settings.frontend_url}/api/user-settings/google/callback"
    params = urlencode(
        {
            "client_id": settings.google_client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/generative-language.retriever https://www.googleapis.com/auth/cloud-platform",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
    )

    response = RedirectResponse(
        url=f"https://accounts.google.com/o/oauth2/v2/auth?{params}"
    )
    response.set_cookie("oauth_user_settings_state", state, **_cookie_kwargs())
    response.set_cookie("oauth_user_settings_identity", identity_key_for_user(user), **_cookie_kwargs())
    return response


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str,
    state: str = "",
    user: dict = Depends(require_auth),
):
    settings = get_settings()
    redirect_base = f"{settings.frontend_url}/user-settings"

    stored_state = request.cookies.get("oauth_user_settings_state")
    stored_identity = request.cookies.get("oauth_user_settings_identity")
    current_identity = identity_key_for_user(user)

    if not stored_state or stored_state != state or stored_identity != current_identity:
        response = RedirectResponse(url=f"{redirect_base}?error=invalid_state")
        response.delete_cookie("oauth_user_settings_state", path="/")
        response.delete_cookie("oauth_user_settings_identity", path="/")
        return response

    redirect_uri = f"{settings.frontend_url}/api/user-settings/google/callback"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            token_payload = resp.json()
    except Exception:
        response = RedirectResponse(url=f"{redirect_base}?error=exchange_failed")
        response.delete_cookie("oauth_user_settings_state", path="/")
        response.delete_cookie("oauth_user_settings_identity", path="/")
        return response

    access_token = str(token_payload.get("access_token", "")).strip()
    refresh_token = str(token_payload.get("refresh_token", "")).strip() or None
    expires_in = int(token_payload.get("expires_in", 3600) or 3600)

    if not access_token:
        response = RedirectResponse(url=f"{redirect_base}?error=token_missing")
        response.delete_cookie("oauth_user_settings_state", path="/")
        response.delete_cookie("oauth_user_settings_identity", path="/")
        return response

    store = _ensure_store()
    store.set_google_oauth_tokens(
        current_identity,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
    )

    prefs, _ = store.get_user_state(current_identity)
    prefs["provider"] = "google"
    prefs["auth_mode"] = "account"
    if not prefs.get("model") or prefs.get("model") not in store.provider_models().get("google", []):
        models = store.provider_models().get("google", [])
        prefs["model"] = models[0] if models else ""
    store.save_preferences(current_identity, prefs)

    response = RedirectResponse(url=f"{redirect_base}?google_connected=1")
    response.delete_cookie("oauth_user_settings_state", path="/")
    response.delete_cookie("oauth_user_settings_identity", path="/")
    return response


@router.post("/google/disconnect")
async def disconnect_google(user: dict = Depends(require_auth)):
    store = _ensure_store()
    identity_key = identity_key_for_user(user)
    store.clear_google_oauth(identity_key)

    prefs, has_saved = store.get_user_state(identity_key)
    if has_saved and prefs.get("provider") == "google" and prefs.get("auth_mode") == "account":
        prefs["auth_mode"] = "api_key"
        store.save_preferences(identity_key, prefs)

    return _build_response(identity_key, store)
