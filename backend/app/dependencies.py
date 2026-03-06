"""Shared FastAPI dependencies."""

from fastapi import Cookie, HTTPException

from app.services.auth import decode_jwt_token


async def require_auth(session: str = Cookie(default=None)) -> dict:
    """Validate the session cookie and return ``{user, provider}``."""
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode_jwt_token(session)
        return {"user": payload["sub"], "provider": payload["provider"]}
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
