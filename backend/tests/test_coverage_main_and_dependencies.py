"""Coverage for ``app.main`` rate-limit wiring/handlers and ``app.dependencies``.

``TestClient`` in this Starlette version has no ``client=`` kwarg, so loopback
vs. remote peers come from a scope-rewriting ASGI wrapper (the pattern already
used by ``tests/test_dependencies_auth.py``). Redis is never contacted: the
limiter factory is patched with an in-process double.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app import main as main_module
from app.config import get_settings
from app.dependencies import (
    _is_local_frontend_config,
    _is_localhost_host,
    _is_loopback_host,
    is_admin_identity,
    require_auth,
)
from app.main import (
    _init_rate_limiter,
    _load_dotenv,
    _rate_limit_key,
    _rate_limit_rule,
    create_app,
)
from app.services.auth import create_jwt_token
from app.services.llm_policy import (
    LLMServiceApprovalRequiredError,
    PersonalCredentialRequiredError,
    StudyAppLLMNotAssignedError,
)
from app.services.rate_limit import RateLimitDecision

STRONG_SECRET = "m" * 64
_LOOPBACK = "127.0.0.1"
_REMOTE = "203.0.113.10"


def _build_request(path: str, headers: dict, client_host: str) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": (client_host, 54321),
        "server": ("testserver", 80),
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


class _FakeRedisLimiter:
    def __init__(self):
        self.pings = 0

    def ping(self):
        self.pings += 1

    def check(self, *, scope, key, limit, window_seconds):
        return RateLimitDecision(allowed=True, retry_after_seconds=0)

    def clear(self):
        return None


# ── main._init_rate_limiter ──────────────────────────────


class RedisStartupTests(unittest.TestCase):
    def test_redis_backend_is_pinged_and_adopted_at_startup(self):
        from app.config import Settings

        limiter = _FakeRedisLimiter()
        settings = Settings(rate_limit_backend="redis", rate_limit_strict=True)
        with patch.object(main_module, "create_rate_limiter", return_value=limiter):
            resolved = _init_rate_limiter(settings)

        self.assertIs(resolved, limiter)
        # The ping is what makes strict mode meaningful.
        self.assertEqual(limiter.pings, 1)

    def test_backend_name_is_normalised_case_and_whitespace(self):
        from app.config import Settings

        limiter = _FakeRedisLimiter()
        settings = Settings(rate_limit_backend="  REDIS ")
        with patch.object(main_module, "create_rate_limiter", return_value=limiter):
            self.assertIs(_init_rate_limiter(settings), limiter)

    def test_memory_backend_short_circuits_without_pinging(self):
        from app.config import Settings

        limiter = _FakeRedisLimiter()
        settings = Settings(rate_limit_backend="memory")
        with patch.object(
            main_module, "create_rate_limiter", side_effect=AssertionError("not called")
        ):
            self.assertIsNot(_init_rate_limiter(settings), limiter)

    def test_a_limiter_without_ping_is_still_adopted(self):
        from app.config import Settings

        class NoPingLimiter:
            def check(self, **kwargs):  # pragma: no cover - never called here
                return None

            def clear(self):  # pragma: no cover - never called here
                return None

        limiter = NoPingLimiter()
        settings = Settings(rate_limit_backend="redis", rate_limit_strict=True)
        with patch.object(main_module, "create_rate_limiter", return_value=limiter):
            self.assertIs(_init_rate_limiter(settings), limiter)


# ── main._rate_limit_key ─────────────────────────────────


class RateLimitKeyTests(unittest.TestCase):
    def setUp(self):
        self.prev_secret = os.environ.get("STUDY_AUTH_SECRET_KEY")
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev_secret is None:
            os.environ.pop("STUDY_AUTH_SECRET_KEY", None)
        else:
            os.environ["STUDY_AUTH_SECRET_KEY"] = self.prev_secret
        get_settings.cache_clear()

    def test_oauth_scope_is_always_keyed_by_ip(self):
        request = _build_request("/api/auth/github", {}, _REMOTE)
        self.assertEqual(_rate_limit_key("oauth", request), f"ip:{_REMOTE}")

    def test_llm_scope_prefers_the_decoded_session_identity(self):
        token = create_jwt_token("Alice", "GitHub")
        request = _build_request(
            "/api/chat/follow-up", {"cookie": f"session={token}"}, _REMOTE
        )
        self.assertEqual(_rate_limit_key("llm", request), "user:github:alice")

    def test_undecodable_session_cookie_falls_back_to_the_peer_ip(self):
        request = _build_request(
            "/api/chat/follow-up", {"cookie": "session=not-a-jwt"}, _REMOTE
        )
        self.assertEqual(_rate_limit_key("llm", request), f"ip:{_REMOTE}")

    def test_session_cookie_without_a_subject_falls_back_to_the_peer_ip(self):
        import jwt as pyjwt

        token = pyjwt.encode({"provider": "github"}, STRONG_SECRET, algorithm="HS256")
        request = _build_request(
            "/api/chat/follow-up", {"cookie": f"session={token}"}, _REMOTE
        )
        self.assertEqual(_rate_limit_key("llm", request), f"ip:{_REMOTE}")

    def test_missing_session_cookie_falls_back_to_the_peer_ip(self):
        request = _build_request("/api/chat/follow-up", {}, _REMOTE)
        self.assertEqual(_rate_limit_key("llm", request), f"ip:{_REMOTE}")

    def test_rules_are_selected_per_scope(self):
        self.assertEqual(_rate_limit_rule("oauth")[0], 20)
        self.assertEqual(_rate_limit_rule("session")[0], 90)
        self.assertEqual(_rate_limit_rule("llm")[0], 30)
        # Any unknown scope falls back to the LLM budget.
        self.assertEqual(_rate_limit_rule("mystery"), _rate_limit_rule("llm"))


# ── main._load_dotenv ────────────────────────────────────


class LoadDotenvTests(unittest.TestCase):
    def test_missing_dotenv_files_is_not_fatal(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertLogs("app.main", level="WARNING") as logs:
                self.assertIsNone(_load_dotenv())

        self.assertTrue(
            any("No .env file found" in line for line in logs.output), logs.output
        )

    def test_existing_dotenv_populates_missing_env_vars_only(self):
        target = Path("/tmp/coverage-dotenv-probe/.env")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "# comment\n\nCOVERAGE_DOTENV_A=from-file\nCOVERAGE_DOTENV_B=also-file\n"
        )
        os.environ["COVERAGE_DOTENV_B"] = "already-set"
        os.environ.pop("COVERAGE_DOTENV_A", None)
        cwd = os.getcwd()
        os.chdir(str(target.parent))
        try:
            _load_dotenv()
            self.assertEqual(os.environ["COVERAGE_DOTENV_A"], "from-file")
            self.assertEqual(os.environ["COVERAGE_DOTENV_B"], "already-set")
        finally:
            os.chdir(cwd)
            os.environ.pop("COVERAGE_DOTENV_A", None)
            os.environ.pop("COVERAGE_DOTENV_B", None)

    def test_malformed_lines_are_skipped(self):
        target = Path("/tmp/coverage-dotenv-malformed/.env")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("no-equals-sign\n\n# comment\nCOVERAGE_DOTENV_C=ok\n")
        os.environ.pop("COVERAGE_DOTENV_C", None)
        cwd = os.getcwd()
        os.chdir(str(target.parent))
        try:
            _load_dotenv()
            self.assertEqual(os.environ["COVERAGE_DOTENV_C"], "ok")
        finally:
            os.chdir(cwd)
            os.environ.pop("COVERAGE_DOTENV_C", None)


# ── main middleware + handlers ───────────────────────────


class AppWiringTests(unittest.TestCase):
    env_keys = [
        "STUDY_AUTH_SECRET_KEY",
        "STUDY_ENABLE_RATE_LIMITING",
        "STUDY_LLM_RATE_LIMIT_REQUESTS",
        "STUDY_CORS_ORIGINS",
        "STUDY_FRONTEND_URL",
        "STUDY_CREDENTIALS_ENCRYPTION_KEY",
        "STUDY_USER_SETTINGS_FILE",
        "STUDY_LLM_SERVICE_USERS_FILE",
        "STUDY_LLM_ASSIGNMENTS_FILE",
        "STUDY_LEARNING_DB_PATH",
    ]

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        os.environ["STUDY_ENABLE_RATE_LIMITING"] = "true"
        os.environ["STUDY_CREDENTIALS_ENCRYPTION_KEY"] = "coverage-encryption-secret"
        os.environ["STUDY_USER_SETTINGS_FILE"] = os.path.join(
            self.tmpdir.name, "user_settings.json"
        )
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = os.path.join(
            self.tmpdir.name, "llm_service_users.json"
        )
        os.environ["STUDY_LLM_ASSIGNMENTS_FILE"] = os.path.join(
            self.tmpdir.name, "llm_assignments.json"
        )
        os.environ["STUDY_LEARNING_DB_PATH"] = os.path.join(self.tmpdir.name, "learning.db")
        get_settings.cache_clear()
        # Keep the shared instance other test modules imported by name; only
        # its recorded hits are reset.
        self.shared_limiter = main_module._rate_limiter
        self.shared_limiter.clear()

    def tearDown(self):
        self.tmpdir.cleanup()
        self.shared_limiter.clear()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def test_health_endpoint_reports_the_configured_service_name(self):
        app = create_app()
        with TestClient(app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["service"], get_settings().app_name)

    def test_rate_limiting_can_be_disabled_entirely(self):
        os.environ["STUDY_ENABLE_RATE_LIMITING"] = "false"
        get_settings.cache_clear()
        app = create_app()
        with TestClient(app) as client:
            client.cookies.set("session", create_jwt_token("alice", "github"))
            statuses = [
                client.get("/api/llm/providers").status_code for _ in range(5)
            ]
        self.assertEqual(statuses, [200] * 5)

    def test_unscoped_paths_are_never_rate_limited(self):
        app = create_app()
        with TestClient(app) as client:
            statuses = [client.get("/health").status_code for _ in range(5)]
        self.assertEqual(statuses, [200] * 5)

    def test_cors_origins_include_the_frontend_url_without_a_trailing_slash(self):
        os.environ["STUDY_FRONTEND_URL"] = "https://app.example.com/"
        os.environ["STUDY_CORS_ORIGINS"] = "http://a.test, ,http://b.test"
        get_settings.cache_clear()
        app = create_app()
        with TestClient(app) as client:
            response = client.get(
                "/health", headers={"Origin": "https://app.example.com"}
            )
        self.assertIn("access-control-allow-origin", response.headers)

    def test_an_empty_frontend_url_is_omitted_from_cors_origins(self):
        os.environ["STUDY_FRONTEND_URL"] = ""
        os.environ["STUDY_CORS_ORIGINS"] = "http://a.test,,http://b.test"
        get_settings.cache_clear()
        app = create_app()
        with TestClient(app) as client:
            # Blank entries are discarded rather than registered as an origin.
            allowed = client.get("/health", headers={"Origin": "http://b.test"})
        self.assertEqual(
            allowed.headers.get("access-control-allow-origin"), "http://b.test"
        )

    def test_policy_exceptions_are_translated_to_json_responses(self):
        app = create_app()

        @app.get("/boom/approval")
        async def _approval():
            raise LLMServiceApprovalRequiredError("needs approval")

        @app.get("/boom/assignment")
        async def _assignment():
            raise StudyAppLLMNotAssignedError("needs an assignment")

        @app.get("/boom/credential")
        async def _credential():
            raise PersonalCredentialRequiredError("needs a credential")

        with TestClient(app) as client:
            approval = client.get("/boom/approval")
            assignment = client.get("/boom/assignment")
            credential = client.get("/boom/credential")

        self.assertEqual(approval.status_code, 403)
        self.assertEqual(approval.json()["detail"]["code"], "llm_service_approval_required")
        self.assertEqual(
            assignment.json()["detail"]["code"], "study_app_llm_not_assigned"
        )
        self.assertEqual(credential.status_code, 400)
        self.assertEqual(
            credential.json()["detail"]["code"], "personal_credential_required"
        )


# ── app.dependencies ────────────────────────────────────


class HostPredicateTests(unittest.TestCase):
    def test_localhost_host_accepts_the_name_and_loopback_addresses(self):
        for host in ("localhost", "LocalHost", "localhost.", "  localhost  "):
            with self.subTest(host=host):
                self.assertTrue(_is_localhost_host(host))
        for host in ("127.0.0.1", "127.0.0.53", "::1", "::FFFF:127.0.0.1"):
            with self.subTest(host=host):
                self.assertTrue(_is_localhost_host(host))

    def test_localhost_host_rejects_public_addresses_and_junk(self):
        for host in ("example.com", "203.0.113.10", "10.0.0.1", "", "  "):
            with self.subTest(host=host):
                self.assertFalse(_is_localhost_host(host))

    def test_loopback_host_requires_a_parseable_loopback_peer(self):
        self.assertFalse(_is_loopback_host(""))
        self.assertFalse(_is_loopback_host("   "))
        self.assertFalse(_is_loopback_host("not-an-ip"))
        self.assertFalse(_is_loopback_host("203.0.113.10"))
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host(" ::1 "))

    def test_local_frontend_config_only_accepts_http_loopback_urls(self):
        self.assertTrue(_is_local_frontend_config("http://localhost:5174"))
        self.assertTrue(_is_local_frontend_config("http://127.0.0.1:5174"))
        self.assertFalse(_is_local_frontend_config("https://localhost:5174"))
        self.assertFalse(_is_local_frontend_config("https://app.example.com"))
        self.assertFalse(_is_local_frontend_config(""))


class AdminIdentityTests(unittest.TestCase):
    def setUp(self):
        self.prev = os.environ.get("STUDY_ADMIN_USERS")
        get_settings.cache_clear()

    def tearDown(self):
        if self.prev is None:
            os.environ.pop("STUDY_ADMIN_USERS", None)
        else:
            os.environ["STUDY_ADMIN_USERS"] = self.prev
        get_settings.cache_clear()

    def test_blank_identity_is_never_admin(self):
        os.environ["STUDY_ADMIN_USERS"] = "google:admin@example.com"
        get_settings.cache_clear()
        self.assertFalse(is_admin_identity(user="", provider="google"))
        self.assertFalse(is_admin_identity(user="   ", provider="google"))

    def test_bare_and_provider_scoped_entries_both_match(self):
        os.environ["STUDY_ADMIN_USERS"] = "ops@example.com, github:maintainer"
        get_settings.cache_clear()
        self.assertTrue(is_admin_identity("OPS@example.com", "google"))
        self.assertTrue(is_admin_identity("maintainer", "github"))
        self.assertFalse(is_admin_identity("maintainer", "google"))

    def test_empty_admin_list_denies_everyone(self):
        os.environ["STUDY_ADMIN_USERS"] = " , ,"
        get_settings.cache_clear()
        self.assertFalse(is_admin_identity("ops@example.com", "google"))


class DevBypassWithSessionTests(unittest.TestCase):
    """The bypass path must still honour a real session when one is presented."""

    def setUp(self):
        self.env_keys = [
            "STUDY_AUTH_SECRET_KEY",
            "STUDY_DEV_AUTH_BYPASS_LOCALHOST",
            "STUDY_ENVIRONMENT",
            "STUDY_FRONTEND_URL",
        ]
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        os.environ["STUDY_DEV_AUTH_BYPASS_LOCALHOST"] = "true"
        os.environ["STUDY_ENVIRONMENT"] = "development"
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        get_settings.cache_clear()

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

    def _get(self, client_host: str, **kwargs) -> httpx.Response:
        async def _asgi(scope, receive, send):
            if scope.get("type") == "http":
                scope = {**scope, "client": (client_host, 50000)}
            await self.app(scope, receive, send)

        async def _request():
            transport = httpx.ASGITransport(app=_asgi)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://localhost:8001", **kwargs
            ) as client:
                return await client.get("/secure")

        return asyncio.run(_request())

    def test_a_valid_session_identity_wins_over_the_bypass_identity(self):
        token = create_jwt_token("alice", "github")
        response = self._get(_LOOPBACK, cookies={"session": token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": "alice", "provider": "github"})

    def test_an_undecodable_session_falls_back_to_the_local_dev_identity(self):
        response = self._get(_LOOPBACK, cookies={"session": "not-a-jwt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": "local-dev", "provider": "local"})


if __name__ == "__main__":
    unittest.main()