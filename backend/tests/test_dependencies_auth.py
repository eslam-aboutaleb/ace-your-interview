import os
import unittest

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth


class RequireAuthBypassTests(unittest.TestCase):
    def setUp(self):
        self.env_keys = [
            "STUDY_DEV_AUTH_BYPASS_LOCALHOST",
            "STUDY_ENVIRONMENT",
            "STUDY_FRONTEND_URL",
        ]
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
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

    def test_local_dev_bypass_allows_loopback_requests(self):
        self._apply_settings(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        client = TestClient(
            self.app,
            base_url="http://localhost:8001",
            client=("127.0.0.1", 50000),
        )
        response = client.get("/secure")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"user": "local-dev", "provider": "local"})

    def test_bypass_is_disabled_outside_dev_environment(self):
        self._apply_settings(
            bypass=True,
            environment="production",
            frontend_url="http://localhost:5174",
        )
        client = TestClient(
            self.app,
            base_url="http://localhost:8001",
            client=("127.0.0.1", 50000),
        )
        response = client.get("/secure")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_bypass_rejects_non_loopback_client(self):
        self._apply_settings(
            bypass=True,
            environment="development",
            frontend_url="http://localhost:5174",
        )
        client = TestClient(
            self.app,
            base_url="http://localhost:8001",
            client=("203.0.113.10", 50000),
        )
        response = client.get("/secure")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")


if __name__ == "__main__":
    unittest.main()
