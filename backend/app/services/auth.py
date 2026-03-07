"""Authentication helpers: JWT, OAuth exchange, allowed-users store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jwt

from app.config import get_settings, resolve_auth_secret_key
from app.services.http_clients import get_oauth_http_client

logger = logging.getLogger(__name__)

ALLOWED_USERS_FILE = Path("allowed_users.json")


# ── Allowed-users store (JSON file, seeded from env vars) ────────────


def _load_allowed_users() -> dict:
    if ALLOWED_USERS_FILE.exists():
        with open(ALLOWED_USERS_FILE) as f:
            return json.load(f)
    # First run – seed from env vars
    settings = get_settings()
    data = {
        "github_users": [
            u.strip().lower()
            for u in settings.allowed_github_users.split(",")
            if u.strip()
        ],
        "google_emails": [
            e.strip().lower()
            for e in settings.allowed_google_emails.split(",")
            if e.strip()
        ],
    }
    _save_allowed_users(data)
    return data


def _save_allowed_users(data: dict) -> None:
    with open(ALLOWED_USERS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_allowed_users() -> dict:
    return _load_allowed_users()


def add_allowed_user(provider: str, identifier: str) -> dict:
    data = _load_allowed_users()
    val = identifier.strip().lower()
    if provider == "github" and val not in data["github_users"]:
        data["github_users"].append(val)
    elif provider == "google" and val not in data["google_emails"]:
        data["google_emails"].append(val)
    _save_allowed_users(data)
    return data


def remove_allowed_user(provider: str, identifier: str) -> dict:
    data = _load_allowed_users()
    val = identifier.strip().lower()
    if provider == "github":
        data["github_users"] = [u for u in data["github_users"] if u != val]
    elif provider == "google":
        data["google_emails"] = [e for e in data["google_emails"] if e != val]
    _save_allowed_users(data)
    return data


def is_allowed(provider: str, identifier: str) -> bool:
    data = _load_allowed_users()
    val = identifier.strip().lower()
    if provider == "github":
        return val in data["github_users"]
    if provider == "google":
        return val in data["google_emails"]
    return False


# ── JWT helpers ──────────────────────────────────────────────


def create_jwt_token(subject: str, provider: str) -> str:
    settings = get_settings()
    secret = resolve_auth_secret_key(settings)
    return jwt.encode(
        {"sub": subject, "provider": provider},
        secret,
        algorithm="HS256",
    )


def decode_jwt_token(token: str) -> dict:
    settings = get_settings()
    secret = resolve_auth_secret_key(settings)
    return jwt.decode(token, secret, algorithms=["HS256"])


# ── OAuth code exchange ──────────────────────────────────────


async def exchange_github_code(code: str) -> str:
    """Return the GitHub username for an OAuth *code*."""
    settings = get_settings()
    client = get_oauth_http_client()
    resp = await client.post(
        "https://github.com/login/oauth/access_token",
        json={
            "client_id": settings.github_client_id,
            "client_secret": settings.github_client_secret,
            "code": code,
        },
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    token_data = resp.json()
    if "access_token" not in token_data:
        raise ValueError(f"GitHub token error: {token_data}")
    access_token = token_data["access_token"]

    user_resp = await client.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_resp.raise_for_status()
    return user_resp.json()["login"]


async def exchange_google_code(code: str, redirect_uri: str) -> str:
    """Return the Google email for an OAuth *code*."""
    settings = get_settings()
    client = get_oauth_http_client()
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
    access_token = resp.json()["access_token"]

    user_resp = await client.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    user_resp.raise_for_status()
    return user_resp.json()["email"]
