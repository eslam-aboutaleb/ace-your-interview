"""Admin-managed per-user Study App LLM assignments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from app.config import get_settings
from app.schemas.models import MODEL_SUGGESTIONS, USER_SETTINGS_PROVIDER_WHITELIST


ALLOWED_PROVIDERS = set(USER_SETTINGS_PROVIDER_WHITELIST)


def normalise_identity(provider: str, identifier: str) -> str:
    p = (provider or "").strip().lower()
    i = (identifier or "").strip().lower()
    if not p or not i:
        return ""
    return f"{p}:{i}"


@dataclass
class LLMUserAssignment:
    provider: str
    model: str
    updated_at: str


class LLMAssignmentsStore:
    """JSON-backed store for admin-assigned backend LLM provider/model per identity."""

    def __init__(self):
        settings = get_settings()
        self._path = Path(settings.llm_assignments_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def provider_models(self) -> dict[str, list[str]]:
        return {
            provider: list(MODEL_SUGGESTIONS.get(provider, []))
            for provider in USER_SETTINGS_PROVIDER_WHITELIST
        }

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {"assignments": {}}
        try:
            with open(self._path) as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("assignments"), dict):
                return data
        except Exception:
            pass
        return {"assignments": {}}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix="llm-assignments-",
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

    def _validate(self, provider: str, model: str) -> tuple[str, str]:
        p = (provider or "").strip().lower()
        m = (model or "").strip()
        if p not in ALLOWED_PROVIDERS:
            raise ValueError("Unsupported provider for assignment")
        allowed_models = set(MODEL_SUGGESTIONS.get(p, []))
        if not m or m not in allowed_models:
            raise ValueError("Unsupported model for assigned provider")
        return p, m

    def set_assignment(
        self,
        *,
        identity_key: str,
        provider: str,
        model: str,
    ) -> LLMUserAssignment:
        key = (identity_key or "").strip().lower()
        if not key:
            raise ValueError("Missing identity")
        p, m = self._validate(provider, model)
        payload = {
            "provider": p,
            "model": m,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            data = self._load()
            data["assignments"][key] = payload
            self._save(data)
        return LLMUserAssignment(**payload)

    def get_assignment(self, *, identity_key: str) -> Optional[LLMUserAssignment]:
        key = (identity_key or "").strip().lower()
        if not key:
            return None
        with self._lock:
            data = self._load()
            raw = data.get("assignments", {}).get(key)
        if not isinstance(raw, dict):
            return None
        provider = str(raw.get("provider", "")).strip().lower()
        model = str(raw.get("model", "")).strip()
        updated_at = str(raw.get("updated_at", "")).strip()
        if provider not in ALLOWED_PROVIDERS or not model:
            return None
        if model not in set(MODEL_SUGGESTIONS.get(provider, [])):
            return None
        return LLMUserAssignment(
            provider=provider,
            model=model,
            updated_at=updated_at,
        )

    def delete_assignment(self, *, identity_key: str) -> bool:
        key = (identity_key or "").strip().lower()
        if not key:
            return False
        with self._lock:
            data = self._load()
            existed = key in data.get("assignments", {})
            data.get("assignments", {}).pop(key, None)
            self._save(data)
        return existed

    def list_assignments(self) -> dict[str, LLMUserAssignment]:
        with self._lock:
            data = self._load()
        out: dict[str, LLMUserAssignment] = {}
        for identity_key in data.get("assignments", {}):
            assignment = self.get_assignment(identity_key=identity_key)
            if assignment is not None:
                out[identity_key] = assignment
        return out
