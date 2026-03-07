"""Shared FastAPI dependencies."""

from ipaddress import ip_address

from fastapi import Cookie, Depends, HTTPException, Request

from app.config import get_settings
from app.services.auth import decode_jwt_token


def _is_localhost_request(request: Request) -> bool:
    host = (request.url.hostname or "").strip().lower().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _is_local_frontend_config(frontend_url: str) -> bool:
    frontend = (frontend_url or "").strip().lower()
    return frontend.startswith("http://localhost") or frontend.startswith(
        "http://127.0.0.1"
    )


def _is_loopback_client_request(request: Request) -> bool:
    client_host = (request.client.host if request.client else "").strip()
    if not client_host:
        return False
    try:
        return ip_address(client_host).is_loopback
    except ValueError:
        return False


def _is_dev_environment(environment: str) -> bool:
    return (environment or "").strip().lower() in {"development", "dev", "local", "test"}


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

    if (
        settings.dev_auth_bypass_localhost
        and _is_dev_environment(settings.environment)
        and _is_local_frontend_config(settings.frontend_url)
        and _is_localhost_request(request)
        and _is_loopback_client_request(request)
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
