"""Tests for the shared auth dependencies and the dev-bypass predicate.

Two layers are exercised:

1. ``is_dev_auth_bypass_enabled`` called directly — the pure predicate that both
   ``require_auth`` and the voice WebSocket share.
2. A real ASGI request driven end-to-end through ``require_auth``, with the
   scope's ``client`` overridden so loopback vs. remote is genuinely varied.

Note: Starlette's ``TestClient`` hardcodes ``scope["client"]`` to
``("testclient", 50000)`` and no longer accepts a ``client=`` kwarg, so the
client address is injected with a scope-rewriting ASGI wrapper instead.
"""

import asyncio
import os
import unittest
from typing import Any

import httpx
from fastapi import Depends, FastAPI

from app.config import get_settings
from app.dependencies import is_dev_auth_bypass_enabled, require_auth
from app.services.auth import create_jwt_token

AUTH_SECRET_ENV = "STUDY_AUTH_SECRET_KEY"
STRONG_TEST_SECRET = "b" * 64


def _settings_stub(
    *,
    bypass: bool,
    environment: str,
    frontend_url: str,
) -> Any:
    """Minimal stand-in exposing exactly the attributes the predicate reads."""
    return type(
        "_SettingsStub",
        (),
        {
            "dev_auth_bypass_localhost": bypass,
            "environment": environment,
            "frontend_url": frontend_url,
        },
    )()


def _client_override(app: FastAPI, client_host: str) -> FastAPI:
    """Wrap ``app`` so every ASGI scope reports ``client_host`` as the peer."""

    async def asgi(scope, receive, send):
        if scope.get("type") in {"http", "websocket"}:
            scope = {**scope, "client": (client_host, 50000)}
        await app(scope, receive, send)

    return asgi  # type: ignore[return-value]


async def _async_get(app: FastAPI, client_host: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=_client_override(app, client_host))
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://localhost:8001",
        **kwargs,
    ) as client:
        return await client.get("/secure")


def _get(app: FastAPI, client_host: str, **kwargs) -> httpx.Response:
    return asyncio.run(_async_get(app, client_host, **kwargs))


class DevAuthBypassPredicateTests(unittest.TestCase):
    """Pure predicate — no request plumbing involved."""

    def test_bypass_fires_for_localhost_loopback_dev(self):
        settings = _settings_stub(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        self.assertTrue(
            is_dev_auth_bypass_enabled(
                settings=settings,
                request_host="localhost",
                client_host="127.0.0.1",
            )
        )

    def test_bypass_does_not_fire_when_flag_is_false(self):
        settings = _settings_stub(
            bypass=False,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        self.assertFalse(
            is_dev_auth_bypass_enabled(
                settings=settings,
                request_host="localhost",
                client_host="127.0.0.1",
            )
        )

    def test_bypass_does_not_fire_outside_dev_environment(self):
        settings = _settings_stub(
            bypass=True,
            environment="production",
            frontend_url="http://localhost:5174",
        )
        self.assertFalse(
            is_dev_auth_bypass_enabled(
                settings=settings,
                request_host="localhost",
                client_host="127.0.0.1",
            )
        )

    def test_bypass_does_not_fire_for_non_loopback_client(self):
        settings = _settings_stub(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        self.assertFalse(
            is_dev_auth_bypass_enabled(
                settings=settings,
                request_host="localhost",
                client_host="203.0.113.10",
            )
        )

    def test_bypass_does_not_fire_for_non_local_frontend_config(self):
        settings = _settings_stub(
            bypass=True,
            environment="development",
            frontend_url="https://app.example.com",
        )
        self.assertFalse(
            is_dev_auth_bypass_enabled(
                settings=settings,
                request_host="localhost",
                client_host="127.0.0.1",
            )
        )


class RequireAuthRequestTests(unittest.TestCase):
    """End-to-end through ``require_auth`` over a real ASGI request."""

    def setUp(self):
        self.env_keys = [
            "STUDY_DEV_AUTH_BYPASS_LOCALHOST",
            "STUDY_ENVIRONMENT",
            "STUDY_FRONTEND_URL",
            AUTH_SECRET_ENV,
        ]
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ[AUTH_SECRET_ENV] = STRONG_TEST_SECRET

        self.app = FastAPI()

        @self.app.get("/secure")
        async def secure(user: dict = Depends(require_auth)):
            return user

    def tearDown(self):
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _apply_settings(self, *, bypass: bool, environment: str, frontend_url: str):
        os.environ["STUDY_DEV_AUTH_BYPASS_LOCALHOST"] = "true" if bypass else "false"
        os.environ["STUDY_ENVIRONMENT"] = environment
        os.environ["STUDY_FRONTEND_URL"] = frontend_url
        get_settings.cache_clear()

    def _get(self, client_host: str, **kwargs) -> httpx.Response:
        return _get(self.app, client_host, **kwargs)

    def test_local_dev_bypass_allows_loopback_requests(self):
        self._apply_settings(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        response = self._get("127.0.0.1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": "local-dev", "provider": "local"})

    def test_bypass_is_disabled_outside_dev_environment(self):
        self._apply_settings(
            bypass=True,
            environment="production",
            frontend_url="http://localhost:5174",
        )
        response = self._get("127.0.0.1")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_bypass_rejects_non_loopback_client(self):
        self._apply_settings(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        response = self._get("203.0.113.10")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_bypass_is_disabled_when_flag_is_false(self):
        self._apply_settings(
            bypass=False,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        response = self._get("127.0.0.1")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_valid_session_cookie_is_accepted_without_bypass(self):
        self._apply_settings(
            bypass=False,
            environment="production",
            frontend_url="http://localhost:5174",
        )
        token = create_jwt_token("alice", "github")
        response = self._get("203.0.113.10", cookies={"session": token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": "alice", "provider": "github"})

    def test_invalid_session_cookie_is_rejected(self):
        self._apply_settings(
            bypass=False,
            environment="production",
            frontend_url="http://localhost:5174",
        )
        response = self._get("203.0.113.10", cookies={"session": "not-a-jwt"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid or expired session")


if __name__ == "__main__":
    unittest.main()
