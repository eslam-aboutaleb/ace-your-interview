import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import auth


class AuthAdminTests(unittest.TestCase):
    def setUp(self):
        self.prev_admin_users = os.environ.get("STUDY_ADMIN_USERS")
        self.prev_llm_service_users_file = os.environ.get("STUDY_LLM_SERVICE_USERS_FILE")
        os.environ["STUDY_ADMIN_USERS"] = "google:admin@example.com,ops@example.com"

        self.tmpdir = tempfile.TemporaryDirectory()
        self.allowed_file = Path(self.tmpdir.name) / "allowed_users.json"
        self.llm_service_file = Path(self.tmpdir.name) / "llm_service_users.json"
        os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = str(self.llm_service_file)
        get_settings.cache_clear()

        self.allowed_patch = patch("app.services.auth.ALLOWED_USERS_FILE", self.allowed_file)
        self.allowed_patch.start()
        auth._LLM_SERVICE_ACCESS = None

        self.app = FastAPI()
        self.app.include_router(auth.router)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.allowed_patch.stop()
        self.tmpdir.cleanup()
        self.app.dependency_overrides.clear()
        if self.prev_admin_users is None:
            os.environ.pop("STUDY_ADMIN_USERS", None)
        else:
            os.environ["STUDY_ADMIN_USERS"] = self.prev_admin_users
        if self.prev_llm_service_users_file is None:
            os.environ.pop("STUDY_LLM_SERVICE_USERS_FILE", None)
        else:
            os.environ["STUDY_LLM_SERVICE_USERS_FILE"] = self.prev_llm_service_users_file
        auth._LLM_SERVICE_ACCESS = None
        get_settings.cache_clear()

    def test_auth_me_includes_is_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        res = self.client.get("/api/auth/me")
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertTrue(payload["is_admin"])

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "random@example.com",
            "provider": "google",
        }
        res2 = self.client.get("/api/auth/me")
        self.assertEqual(res2.status_code, 200)
        self.assertFalse(res2.json()["is_admin"])

    def test_allowed_users_endpoints_require_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        forbidden = self.client.get("/api/auth/allowed-users")
        self.assertEqual(forbidden.status_code, 403)

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        ok = self.client.get("/api/auth/allowed-users")
        self.assertEqual(ok.status_code, 200)

    def test_llm_service_users_endpoints_require_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        forbidden = self.client.get("/api/auth/llm-service-users")
        self.assertEqual(forbidden.status_code, 403)

        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        ok = self.client.get("/api/auth/llm-service-users")
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(
            sorted(ok.json()["users"]),
            ["google:admin@example.com", "ops@example.com"],
        )

    def test_admin_can_add_remove_llm_service_user(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }

        add = self.client.post(
            "/api/auth/llm-service-users",
            json={"provider": "github", "identifier": "new-user"},
        )
        self.assertEqual(add.status_code, 200)
        self.assertIn("github:new-user", add.json()["users"])

        remove = self.client.delete("/api/auth/llm-service-users/github/new-user")
        self.assertEqual(remove.status_code, 200)
        self.assertNotIn("github:new-user", remove.json()["users"])


if __name__ == "__main__":
    unittest.main()
