"""Helpers for signed OAuth state tokens."""

from __future__ import annotations

import secrets
import time
from typing import Any

import jwt

from app.config import get_settings, resolve_auth_secret_key

_RESERVED_CLAIMS = {"purpose", "iat", "exp", "nonce"}


def create_oauth_state(payload: dict[str, Any], ttl_seconds: int = 600) -> str:
    purpose = str(payload.get("purpose", "")).strip()
    if not purpose:
        raise ValueError("OAuth state payload must include a non-empty purpose")

    now = int(time.time())
    claims: dict[str, Any] = {
        "purpose": purpose,
        "iat": now,
        "exp": now + max(1, int(ttl_seconds)),
        "nonce": secrets.token_urlsafe(24),
    }
    for key, value in payload.items():
        if key in _RESERVED_CLAIMS:
            continue
        claims[key] = value

    secret = resolve_auth_secret_key(get_settings())
    token = jwt.encode(claims, secret, algorithm="HS256")
    return str(token)


def verify_oauth_state(token: str, expected_purpose: str) -> dict[str, Any]:
    purpose = str(expected_purpose or "").strip()
    if not purpose:
        raise ValueError("Expected OAuth state purpose is required")
    if not token:
        raise ValueError("OAuth state is missing")

    secret = resolve_auth_secret_key(get_settings())
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            options={"require": ["purpose", "iat", "exp", "nonce"]},
        )
    except Exception as exc:
        raise ValueError("Invalid OAuth state token") from exc

    if str(payload.get("purpose", "")).strip() != purpose:
        raise ValueError("Unexpected OAuth state purpose")
    if not str(payload.get("nonce", "")).strip():
        raise ValueError("OAuth state nonce is missing")
    return payload
