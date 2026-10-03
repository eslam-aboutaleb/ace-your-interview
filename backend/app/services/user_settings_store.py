"""Per-user LLM settings and credentials store with encryption at rest."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.fernet import Fernet, InvalidToken

from app.config import get_settings
from app.schemas.models import (
    MODEL_SUGGESTIONS,
    USER_SETTINGS_PROVIDER_WHITELIST,
    VoiceTierEnum,
)
from app.services.http_clients import get_oauth_http_client
from app.services.llm_assignments_store import LLMAssignmentsStore, normalise_identity
from app.services.llm_policy import (
    PERSONAL_CREDENTIAL_REQUIRED_CODE,
    APPROVAL_REQUIRED_CODE,
    STUDY_APP_NOT_ASSIGNED_CODE,
)
from app.services.llm_service_access import LLMServiceAccess

logger = logging.getLogger(__name__)

USER_SETTINGS_PROVIDERS = list(USER_SETTINGS_PROVIDER_WHITELIST)

#: Voice tiers a user may pin as their preference. Unknown or disabled
#: tiers fall back to the admin default at read time.
VALID_VOICE_TIERS = frozenset(tier.value for tier in VoiceTierEnum)


def identity_key_for_user(user_identity: dict | None) -> str:
    if not user_identity:
        return ""
    provider = str(user_identity.get("provider", "")).strip().lower()
    user = str(user_identity.get("user", "")).strip().lower()
    if not user:
        return ""
    return f"{provider}:{user}"


def resolve_credentials_encryption_secret(settings: Any) -> str:
    secret = str(settings.credentials_encryption_key or "").strip()
    if secret:
        return secret

    environment = str(getattr(settings, "environment", "") or "").strip().lower()
    allow_fallback = bool(getattr(settings, "allow_insecure_dev_encryption_fallback", False))
    fallback_secret = str(
        getattr(settings, "insecure_dev_credentials_encryption_secret", "") or ""
    ).strip()

    if environment == "development" and allow_fallback:
        if not fallback_secret:
            raise RuntimeError(
                "STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET must be set when "
                "STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK=true"
            )
        logger.warning("Using explicitly configured development credentials encryption fallback secret.")
        return fallback_secret

    raise RuntimeError(
        "STUDY_CREDENTIALS_ENCRYPTION_KEY must be set. For local development only, set "
        "STUDY_ALLOW_INSECURE_DEV_ENCRYPTION_FALLBACK=true and provide "
        "STUDY_INSECURE_DEV_CREDENTIALS_ENCRYPTION_SECRET."
    )


class UserSettingsStore:
    """JSON-backed encrypted store for user LLM preferences and credentials."""

    def __init__(
        self,
        llm_service_access: Optional[LLMServiceAccess] = None,
        llm_assignments_store: Optional[LLMAssignmentsStore] = None,
    ):
        settings = get_settings()
        self._path = Path(settings.user_settings_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._settings = settings
        self._llm_service_access = llm_service_access or LLMServiceAccess()
        self._llm_assignments_store = llm_assignments_store or LLMAssignmentsStore()
        self._cache_data: dict[str, Any] | None = None
        self._cache_mtime_ns: int | None = None
        self._cache_size: int | None = None

        secret = resolve_credentials_encryption_secret(settings)
        digest = hashlib.sha256(secret.encode("utf-8")).digest()
        key = base64.urlsafe_b64encode(digest)
        self._cipher = Fernet(key)

    @staticmethod
    def _identity_parts(identity_key: str) -> tuple[str, str]:
        raw = (identity_key or "").strip().lower()
        if not raw:
            return "", ""
        if ":" in raw:
            provider, user = raw.split(":", 1)
            return provider.strip(), user.strip()
        return "", raw

    @staticmethod
    def providers() -> list[str]:
        return list(USER_SETTINGS_PROVIDERS)

    def provider_models(self) -> dict[str, list[str]]:
        return {
            provider: list(MODEL_SUGGESTIONS.get(provider, []))
            for provider in USER_SETTINGS_PROVIDERS
        }

    def default_preferences(self) -> dict[str, Any]:
        provider = self._settings.default_provider.strip().lower()
        if provider not in USER_SETTINGS_PROVIDERS:
            provider = "openai"
        suggestions = MODEL_SUGGESTIONS.get(provider, [])
        model = suggestions[0] if suggestions else ""
        return {
            "provider": provider,
            "model": model,
            "temperature": float(self._settings.default_temperature),
            "max_tokens": 0,
            "auth_mode": "api_key",
            "llm_source": "personal",
            "require_answer_reveal": False,
            "voice_tier": None,
        }

    def _encrypt(self, value: str) -> str:
        return self._cipher.encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, encrypted: str) -> Optional[str]:
        if not encrypted:
            return None
        try:
            return self._cipher.decrypt(encrypted.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            logger.warning("Failed to decrypt user secret: invalid token")
            return None

    @staticmethod
    def _empty_data() -> dict[str, Any]:
        return {"users": {}}

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            self._cache_data = self._empty_data()
            self._cache_mtime_ns = None
            self._cache_size = None
            return self._cache_data

        try:
            stat = self._path.stat()
            mtime_ns = int(stat.st_mtime_ns)
            size = int(stat.st_size)
        except OSError:
            return self._empty_data()

        if (
            self._cache_data is not None
            and self._cache_mtime_ns == mtime_ns
            and self._cache_size == size
        ):
            return self._cache_data

        data = self._empty_data()
        try:
            with open(self._path) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("users"), dict):
                data = loaded
        except Exception as exc:
            logger.warning("Failed to load user settings store: %s", exc)
        self._cache_data = data
        self._cache_mtime_ns = mtime_ns
        self._cache_size = size
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix="user-settings-", suffix=".json", dir=str(self._path.parent))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, sort_keys=True)
            os.replace(temp_path, self._path)
            self._cache_data = data
            try:
                stat = self._path.stat()
                self._cache_mtime_ns = int(stat.st_mtime_ns)
                self._cache_size = int(stat.st_size)
            except OSError:
                self._cache_mtime_ns = None
                self._cache_size = None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @staticmethod
    def _sanitise_preferences(raw: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
        provider = str(raw.get("provider", defaults["provider"])).strip().lower()
        if provider not in USER_SETTINGS_PROVIDERS:
            provider = defaults["provider"]

        models = MODEL_SUGGESTIONS.get(provider, [])
        model = str(raw.get("model", defaults.get("model", ""))).strip()
        if model and models and model not in models:
            model = ""

        auth_mode = str(raw.get("auth_mode", defaults.get("auth_mode", "api_key"))).strip().lower()
        if auth_mode not in {"api_key", "account"}:
            auth_mode = "api_key"
        if provider != "google" and auth_mode == "account":
            auth_mode = "api_key"

        llm_source = str(raw.get("llm_source", defaults.get("llm_source", "personal"))).strip().lower()
        if llm_source not in {"personal", "study_app"}:
            llm_source = "personal"

        try:
            temperature = float(raw.get("temperature", defaults["temperature"]))
        except (TypeError, ValueError):
            temperature = float(defaults["temperature"])
        temperature = max(0.0, min(2.0, temperature))

        try:
            max_tokens = int(raw.get("max_tokens", defaults["max_tokens"]))
        except (TypeError, ValueError):
            max_tokens = int(defaults["max_tokens"])
        max_tokens = max(0, max_tokens)

        raw_require_answer_reveal = raw.get(
            "require_answer_reveal",
            defaults.get("require_answer_reveal", False),
        )
        if isinstance(raw_require_answer_reveal, bool):
            require_answer_reveal = raw_require_answer_reveal
        elif isinstance(raw_require_answer_reveal, (int, float)):
            require_answer_reveal = bool(raw_require_answer_reveal)
        elif isinstance(raw_require_answer_reveal, str):
            norm = raw_require_answer_reveal.strip().lower()
            if norm in {"1", "true", "yes", "on"}:
                require_answer_reveal = True
            elif norm in {"0", "false", "no", "off"}:
                require_answer_reveal = False
            else:
                require_answer_reveal = bool(defaults.get("require_answer_reveal", False))
        else:
            require_answer_reveal = bool(defaults.get("require_answer_reveal", False))

        raw_voice_tier = raw.get("voice_tier", defaults.get("voice_tier"))
        voice_tier = str(raw_voice_tier).strip() if raw_voice_tier else ""
        if voice_tier and voice_tier not in VALID_VOICE_TIERS:
            # Unknown tiers fall back to the admin default (None).
            voice_tier = ""

        return {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "auth_mode": auth_mode,
            "llm_source": llm_source,
            "require_answer_reveal": require_answer_reveal,
            "voice_tier": voice_tier or None,
        }

    def get_user_state(self, identity_key: str) -> tuple[dict[str, Any], bool]:
        defaults = self.default_preferences()
        if not identity_key:
            return defaults, False

        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})

        prefs = rec.get("preferences") if isinstance(rec, dict) else None
        if isinstance(prefs, dict):
            return self._sanitise_preferences(prefs, defaults), True
        return defaults, False

    def save_preferences(self, identity_key: str, prefs: dict[str, Any]) -> dict[str, Any]:
        defaults = self.default_preferences()
        clean = self._sanitise_preferences(prefs, defaults)
        if not identity_key:
            return clean

        with self._lock:
            data = self._load()
            rec = data["users"].setdefault(identity_key, {})
            rec["preferences"] = clean
            rec.setdefault("credentials", {})
            rec.setdefault("google_oauth", {})
            self._save(data)
        return clean

    def set_api_key(self, identity_key: str, provider: str, api_key: str) -> None:
        p = provider.strip().lower()
        if p not in USER_SETTINGS_PROVIDERS:
            raise ValueError("Unsupported provider")
        if not identity_key:
            raise ValueError("Missing identity")
        key = api_key.strip()
        if not key:
            raise ValueError("API key is required")

        with self._lock:
            data = self._load()
            rec = data["users"].setdefault(identity_key, {})
            creds = rec.setdefault("credentials", {})
            creds[p] = self._encrypt(key)
            rec.setdefault("google_oauth", {})
            rec.setdefault("preferences", self.default_preferences())
            self._save(data)

    def delete_api_key(self, identity_key: str, provider: str) -> None:
        p = provider.strip().lower()
        if p not in USER_SETTINGS_PROVIDERS or not identity_key:
            return

        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})
            creds = rec.get("credentials") if isinstance(rec, dict) else None
            if isinstance(creds, dict):
                creds.pop(p, None)
                self._save(data)

    def get_api_key(self, identity_key: str, provider: str) -> Optional[str]:
        p = provider.strip().lower()
        if p not in USER_SETTINGS_PROVIDERS or not identity_key:
            return None

        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})
            creds = rec.get("credentials") if isinstance(rec, dict) else None
            encrypted = creds.get(p) if isinstance(creds, dict) else None
        if not isinstance(encrypted, str):
            return None
        return self._decrypt(encrypted)

    def has_api_key(self, identity_key: str, provider: str) -> bool:
        return bool(self.get_api_key(identity_key, provider))

    def get_google_oauth_status(self, identity_key: str) -> bool:
        if not identity_key:
            return False
        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})
            oauth = rec.get("google_oauth") if isinstance(rec, dict) else None
            refresh_enc = oauth.get("refresh_token") if isinstance(oauth, dict) else None
            access_enc = oauth.get("access_token") if isinstance(oauth, dict) else None
        return bool(refresh_enc or access_enc)

    def set_google_oauth_tokens(
        self,
        identity_key: str,
        *,
        access_token: str,
        refresh_token: str | None,
        expires_in: int,
    ) -> None:
        if not identity_key:
            raise ValueError("Missing identity")
        if not access_token.strip():
            raise ValueError("Missing access token")

        with self._lock:
            data = self._load()
            rec = data["users"].setdefault(identity_key, {})
            oauth = rec.setdefault("google_oauth", {})

            oauth["access_token"] = self._encrypt(access_token.strip())
            if refresh_token and refresh_token.strip():
                oauth["refresh_token"] = self._encrypt(refresh_token.strip())
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(1, int(expires_in)))
            oauth["expires_at"] = expires_at.isoformat()

            rec.setdefault("credentials", {})
            rec.setdefault("preferences", self.default_preferences())
            self._save(data)

    def clear_google_oauth(self, identity_key: str) -> None:
        if not identity_key:
            return
        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})
            if isinstance(rec, dict):
                rec["google_oauth"] = {}
                self._save(data)

    async def get_google_access_token(self, identity_key: str) -> Optional[str]:
        if not identity_key:
            return None

        with self._lock:
            data = self._load()
            rec = data["users"].get(identity_key, {})
            oauth = rec.get("google_oauth") if isinstance(rec, dict) else None
            access_enc = oauth.get("access_token") if isinstance(oauth, dict) else ""
            refresh_enc = oauth.get("refresh_token") if isinstance(oauth, dict) else ""
            expires_at_raw = oauth.get("expires_at") if isinstance(oauth, dict) else ""

        access_token = self._decrypt(access_enc) if isinstance(access_enc, str) else None
        refresh_token = self._decrypt(refresh_enc) if isinstance(refresh_enc, str) else None

        if access_token:
            expires_at = None
            if isinstance(expires_at_raw, str) and expires_at_raw:
                try:
                    expires_at = datetime.fromisoformat(expires_at_raw)
                except ValueError:
                    expires_at = None
            if expires_at and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if not expires_at or expires_at > datetime.now(timezone.utc) + timedelta(seconds=30):
                return access_token

        if not refresh_token:
            return None

        try:
            client = get_oauth_http_client()
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": self._settings.google_client_id,
                    "client_secret": self._settings.google_client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            logger.warning("Google OAuth refresh failed for %s: %s", identity_key, exc)
            return None

        next_access = str(payload.get("access_token", "")).strip()
        if not next_access:
            return None
        next_refresh = str(payload.get("refresh_token", "")).strip() or None
        expires_in = int(payload.get("expires_in", 3600) or 3600)
        self.set_google_oauth_tokens(
            identity_key,
            access_token=next_access,
            refresh_token=next_refresh,
            expires_in=expires_in,
        )
        return next_access

    async def resolve_personal_runtime_credential(
        self,
        *,
        identity_key: str,
        provider: str,
        auth_mode: str,
    ) -> tuple[Optional[str | dict[str, str]], str, Optional[str]]:
        p = provider.strip().lower()
        if p not in USER_SETTINGS_PROVIDERS:
            return None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE

        google_access_token: Optional[str] = None
        if p == "google":
            google_access_token = await self.get_google_access_token(identity_key)
            if auth_mode == "account" and google_access_token:
                return {"Authorization": f"Bearer {google_access_token}"}, "user_account", None

        api_key = self.get_api_key(identity_key, p)
        if api_key:
            return api_key, "user_api_key", None

        if p == "google" and google_access_token:
            return {"Authorization": f"Bearer {google_access_token}"}, "user_account", None

        return None, "missing_personal", PERSONAL_CREDENTIAL_REQUIRED_CODE

    def is_identity_approved_for_backend_fallback(self, identity_key: str) -> bool:
        provider, user = self._identity_parts(identity_key)
        return self._llm_service_access.is_user_allowed(user=user, provider=provider)

    def get_study_app_assignment(self, identity_key: str) -> Optional[dict[str, str]]:
        assignment = self._llm_assignments_store.get_assignment(identity_key=identity_key)
        if not assignment:
            return None
        return {
            "provider": assignment.provider,
            "model": assignment.model,
            "updated_at": assignment.updated_at,
        }

    def get_study_app_provider_models(self) -> dict[str, list[str]]:
        return self._llm_assignments_store.provider_models()

    def resolve_study_app_provider_model(
        self,
        *,
        identity_key: str,
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        provider_identity, user_identity = self._identity_parts(identity_key)
        approved = self._llm_service_access.is_user_allowed(
            user=user_identity,
            provider=provider_identity,
        )
        if not approved:
            return None, None, APPROVAL_REQUIRED_CODE

        assignment = self._llm_assignments_store.get_assignment(identity_key=identity_key)
        if assignment is None:
            return None, None, STUDY_APP_NOT_ASSIGNED_CODE
        return assignment.provider, assignment.model, None

    def get_identity_key(self, provider: str, identifier: str) -> str:
        return normalise_identity(provider, identifier)
