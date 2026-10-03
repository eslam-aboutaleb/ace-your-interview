"""Admin-managed allowlist for backend-funded LLM fallback usage."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

from app.config import get_settings


def _normalise_token(value: str) -> str:
    return (value or "").strip().lower()


def _normalise_entry(provider: str, identifier: str) -> str:
    p = _normalise_token(provider)
    u = _normalise_token(identifier)
    if not u:
        return ""
    if not p:
        return u
    return f"{p}:{u}"


def _normalise_admin_entries(raw_admins: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in (raw_admins or "").split(","):
        token = _normalise_token(raw)
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


class LLMServiceAccess:
    """Persistent allowlist for users approved for backend LLM fallback."""

    def __init__(self):
        settings = get_settings()
        self._path = Path(settings.llm_service_users_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._settings = settings

    def _load(self) -> dict[str, list[str]]:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("users"), list):
                    users = []
                    seen: set[str] = set()
                    for raw in data["users"]:
                        # Skip nulls/objects outright: `str(None)` would admit a
                        # literal "none" identity into the allowlist.
                        if not isinstance(raw, str):
                            continue
                        token = _normalise_token(raw)
                        if not token or token in seen:
                            continue
                        seen.add(token)
                        users.append(token)
                    return {"users": users}
            except Exception:
                pass

        seeded = _normalise_admin_entries(self._settings.admin_users)
        data = {"users": seeded}
        self._save(data)
        return data

    def _save(self, data: dict[str, list[str]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix="llm-service-users-",
            suffix=".json",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(temp_path, self._path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def list_users(self) -> dict[str, list[str]]:
        with self._lock:
            data = self._load()
        return {"users": list(data.get("users", []))}

    def add_user(self, provider: str, identifier: str) -> dict[str, list[str]]:
        entry = _normalise_entry(provider, identifier)
        if not entry:
            return self.list_users()
        with self._lock:
            data = self._load()
            users = data.setdefault("users", [])
            if entry not in users:
                users.append(entry)
                users.sort()
                self._save(data)
            return {"users": list(users)}

    def remove_user(self, provider: str, identifier: str) -> dict[str, list[str]]:
        entry = _normalise_entry(provider, identifier)
        if not entry:
            return self.list_users()
        bare_entry = _normalise_entry("", identifier)
        with self._lock:
            data = self._load()
            users = [u for u in data.get("users", []) if u not in {entry, bare_entry}]
            data["users"] = users
            self._save(data)
            return {"users": list(users)}

    def is_user_allowed(self, *, user: str, provider: str) -> bool:
        u = _normalise_token(user)
        p = _normalise_token(provider)
        if not u:
            return False
        provider_scoped = _normalise_entry(p, u)
        with self._lock:
            users = set(self._load().get("users", []))
        return u in users or provider_scoped in users
