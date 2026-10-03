"""Shared FastAPI dependencies."""

from ipaddress import ip_address
from typing import Any

from fastapi import Cookie, Depends, HTTPException, Request

from app.config import get_settings
from app.services.auth import decode_jwt_token


def _is_localhost_host(host: str) -> bool:
    normalised = (host or "").strip().lower().rstrip(".")
    if normalised == "localhost":
        return True
    try:
        return ip_address(normalised).is_loopback
    except ValueError:
        return False


def _is_local_frontend_config(frontend_url: str) -> bool:
    frontend = (frontend_url or "").strip().lower()
    return frontend.startswith("http://localhost") or frontend.startswith(
        "http://127.0.0.1"
    )


def _is_loopback_host(client_host: str) -> bool:
    normalised = (client_host or "").strip()
    if not normalised:
        return False
    try:
        return ip_address(normalised).is_loopback
    except ValueError:
        return False


def _is_dev_environment(environment: str) -> bool:
    return (environment or "").strip().lower() in {"development", "dev", "local", "test"}


def is_dev_auth_bypass_enabled(*, settings: Any, request_host: str, client_host: str) -> bool:
    """Return True only when every dev-bypass precondition holds.

    Non-HTTP transports (WebSocket) call this with the equivalent host values
    so the HTTP and WebSocket auth paths cannot drift apart.
    """
    return bool(
        settings.dev_auth_bypass_localhost
        and _is_dev_environment(settings.environment)
        and _is_local_frontend_config(settings.frontend_url)
        and _is_localhost_host(request_host)
        and _is_loopback_host(client_host)
    )


def _normalise_identity(provider: str, user: str) -> tuple[str, str]:
    return (provider or "").strip().lower(), (user or "").strip().lower()


def is_admin_identity(user: str, provider: str) -> bool:
    settings = get_settings()
    p_norm, u_norm = _normalise_identity(provider, user)
    if not u_norm:
        return False

    entries = [x.strip().lower() for x in settings.admin_users.split(",") if x.strip()]
    if not entries:
        return False

    return (
        u_norm in entries
        or f"{p_norm}:{u_norm}" in entries
    )


async def require_auth(request: Request, session: str = Cookie(default=None)) -> dict:
    """Validate the session cookie and return ``{user, provider}``."""
    settings = get_settings()

    if is_dev_auth_bypass_enabled(
        settings=settings,
        request_host=request.url.hostname or "",
        client_host=request.client.host if request.client else "",
    ):
        if session:
            try:
                payload = decode_jwt_token(session)
                return {"user": payload["sub"], "provider": payload["provider"]}
            except Exception:
                pass
        return {"user": "local-dev", "provider": "local"}

    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_jwt_token(session)
        return {"user": payload["sub"], "provider": payload["provider"]}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")


async def optional_auth(request: Request, session: str = Cookie(default=None)) -> dict | None:
    """Return the authenticated user dict, or ``None`` for anonymous visitors."""
    try:
        return await require_auth(request, session)
    except HTTPException:
        return None


async def require_admin(user: dict = Depends(require_auth)) -> dict:
    if not is_admin_identity(user=user.get("user", ""), provider=user.get("provider", "")):
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
