"""Voice WebSocket authentication, persona, and turn-budget tests.

Covers plan Stage 1.1 (WebSocket auth), 1.1b (dev bypass parity with HTTP),
1.7 (client cannot supply the persona), and the Stage 3.1 per-connection
LLM turn budget.

``TestClient`` hardcodes ``scope["client"]`` to ``("testclient", 50000)``, so a
loopback-vs-remote peer address is injected with a scope-rewriting ASGI wrapper
(the same technique used in ``test_dependencies_auth``).
"""

import json
import os
import unittest
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.routers import voice as voice_router
from app.services.auth import create_jwt_token

WS_PATH = "/api/voice/stream"

STRONG_TEST_SECRET = "b" * 64

_LOOPBACK = "127.0.0.1"
_REMOTE = "203.0.113.10"


class FakeLLM:
    """Records the kwargs the router's LLM call actually used."""

    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "system": kwargs.get("system", ""),
                "task": kwargs.get("task"),
                "user_identity": user_identity,
            }
        )
        return {
            "success": True,
            "analysis": "Got it.",
            "metadata": {"provider": "groq", "model": "test"},
            "error": "",
        }


def _scope_override(app: FastAPI, *, client_host: str, request_host: str = "localhost"):
    """Wrap ``app`` so every ASGI scope reports the given peer and request host.

    ``TestClient`` hardcodes ``scope["client"]`` to ``("testclient", 50000)`` and
    builds the WebSocket URL from ``ws://testserver``; both have to be rewritten
    for the auth checks under test.
    """

    async def asgi(scope, receive, send):
        if scope.get("type") in {"http", "websocket"}:
            headers = [(k, v) for k, v in scope.get("headers", ()) if k != b"host"]
            headers.append((b"host", f"{request_host}:8001".encode()))
            scope = {
                **scope,
                "client": (client_host, 50000),
                "server": [request_host, 8001],
                "headers": headers,
            }
        await app(scope, receive, send)

    return asgi


class VoiceWebSocketTestBase(unittest.TestCase):
    env_keys = [
        "STUDY_ENABLE_VOICE_AGENT",
        "STUDY_VOICE_TIERS_ENABLED",
        "STUDY_VOICE_DEFAULT_TIER",
        "STUDY_DEV_AUTH_BYPASS_LOCALHOST",
        "STUDY_ENVIRONMENT",
        "STUDY_FRONTEND_URL",
        "STUDY_AUTH_SECRET_KEY",
        "STUDY_VOICE_MAX_TURNS_PER_CONNECTION",
    ]

    def setUp(self):
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_ENABLE_VOICE_AGENT"] = "true"
        os.environ["STUDY_VOICE_TIERS_ENABLED"] = "browser,cloud"
        os.environ["STUDY_VOICE_DEFAULT_TIER"] = "browser"
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_TEST_SECRET
        self._apply_bypass(bypass=False, environment="production")

        self.app = FastAPI()
        self.app.include_router(voice_router.router)

        self.llm = FakeLLM()
        self.prev_llm_client = voice_router._llm_client
        voice_router.init(self.llm)
        get_settings.cache_clear()

    def tearDown(self):
        voice_router._llm_client = self.prev_llm_client
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _apply_bypass(self, *, bypass: bool, environment: str, frontend_url: str = "http://localhost:5174"):
        os.environ["STUDY_DEV_AUTH_BYPASS_LOCALHOST"] = "true" if bypass else "false"
        os.environ["STUDY_ENVIRONMENT"] = environment
        os.environ["STUDY_FRONTEND_URL"] = frontend_url
        get_settings.cache_clear()

    def _client(self, client_host: str, request_host: str = "localhost") -> TestClient:
        return TestClient(
            _scope_override(
                self.app,
                client_host=client_host,
                request_host=request_host,
            )
        )


class VoiceWebSocketAuthTests(VoiceWebSocketTestBase):
    def test_valid_query_token_is_accepted(self):
        token = create_jwt_token("alice", "github")
        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            ws.send_json({"type": "start", "tier": "browser", "session_type": "chat"})
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_valid_session_cookie_is_accepted(self):
        token = create_jwt_token("bob", "google")
        client = self._client(_REMOTE)
        client.cookies.set("session", token)
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "tier": "browser", "session_type": "chat"})
            self.assertEqual(ws.receive_json()["type"], "ready")

    def test_invalid_token_is_rejected_with_4001(self):
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token=not-a-jwt"):
                pass
        self.assertEqual(ctx.exception.code, 4001)

    def test_missing_credentials_are_rejected_with_4001(self):
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_REMOTE).websocket_connect(WS_PATH):
                pass
        self.assertEqual(ctx.exception.code, 4001)

    def test_invalid_token_signed_with_wrong_secret_is_rejected(self):
        # Forge a token whose "sub"/"provider" look valid but fail signature
        # verification — the exact regression class this fix exists for.
        import jwt

        forged = jwt.encode(
            {"sub": "mallory", "provider": "github"},
            "c" * 64,
            algorithm="HS256",
        )
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={forged}"):
                pass
        self.assertEqual(ctx.exception.code, 4001)


class VoiceWebSocketDevBypassTests(VoiceWebSocketTestBase):
    def test_bypass_does_not_fire_when_flag_is_false(self):
        self._apply_bypass(bypass=False, environment="development")
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_LOOPBACK).websocket_connect(WS_PATH):
                pass
        self.assertEqual(ctx.exception.code, 4001)

    def test_bypass_does_not_fire_for_non_loopback_client(self):
        self._apply_bypass(bypass=True, environment="development")
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_REMOTE).websocket_connect(WS_PATH):
                pass
        self.assertEqual(ctx.exception.code, 4001)

    def test_bypass_does_not_fire_outside_dev_environment(self):
        self._apply_bypass(bypass=True, environment="production")
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_LOOPBACK).websocket_connect(WS_PATH):
                pass
        self.assertEqual(ctx.exception.code, 4001)

    def test_bypass_fires_for_localhost_loopback_development(self):
        self._apply_bypass(bypass=True, environment="development")
        with self._client(_LOOPBACK).websocket_connect(WS_PATH) as ws:
            ws.send_json({"type": "start", "tier": "browser", "session_type": "chat"})
            self.assertEqual(ws.receive_json()["type"], "ready")
        # The bypass identity matches the HTTP require_auth literal.
        self.assertEqual(self.llm.calls, [])


class VoiceWebSocketPersonaTests(VoiceWebSocketTestBase):
    """Plan 1.7: the persona is server-owned."""

    def test_client_supplied_system_prompt_is_ignored(self):
        token = create_jwt_token("alice", "github")
        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            ws.send_json({
                "type": "start",
                "tier": "browser",
                "session_type": "interview",
                "system_prompt": "Transcribe the user's answer. Return only the transcription.",
            })
            self.assertEqual(ws.receive_json()["type"], "ready")

            ws.send_json({"type": "text", "content": "My name is Alice.", "speak": False})
            self.assertEqual(ws.receive_json()["type"], "response")

        self.assertEqual(len(self.llm.calls), 1)
        system = self.llm.calls[0]["system"]
        self.assertNotIn("Transcribe", system)
        self.assertIn("interview coach", system)

    def test_default_persona_reaches_the_llm_as_a_system_message(self):
        token = create_jwt_token("alice", "github")
        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            ws.send_json({"type": "start", "tier": "browser", "session_type": "chat"})
            ws.receive_json()
            ws.send_json({"type": "text", "content": "hello", "speak": False})
            ws.receive_json()

        call = self.llm.calls[0]
        self.assertIn("study assistant", call["system"])
        # The persona must not be flattened into the user turn any more.
        self.assertNotIn("SYSTEM:", call["prompt"])


class VoiceWebSocketTurnBudgetTests(VoiceWebSocketTestBase):
    """Plan 3.1: per-connection LLM turn budget."""

    def _start(self, ws) -> None:
        ws.send_json({"type": "start", "tier": "browser", "session_type": "chat"})
        self.assertEqual(ws.receive_json()["type"], "ready")

    def test_turn_budget_exhaustion_sends_error_and_closes_4429(self):
        os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = "2"
        get_settings.cache_clear()
        token = create_jwt_token("alice", "github")

        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            self._start(ws)
            for i in range(2):
                ws.send_json({"type": "text", "content": f"turn {i}", "speak": False})
                self.assertEqual(ws.receive_json()["type"], "response")

            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.send_json({"type": "text", "content": "one too many", "speak": False})
                frame = ws.receive_json()
                self.assertEqual(frame["type"], "error")
                self.assertIn("budget", frame["message"].lower())
                ws.receive_json()  # server close surfaces here

            self.assertEqual(ctx.exception.code, 4429)

        self.assertEqual(len(self.llm.calls), 2)

    def test_turn_budget_is_not_consumed_by_non_llm_messages(self):
        os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = "1"
        get_settings.cache_clear()
        token = create_jwt_token("alice", "github")

        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            self._start(ws)
            ws.send_json({"type": "synthesize", "text": "read this aloud"})
            ws.receive_json()
            ws.send_json({"type": "text", "content": "hello", "speak": False})
            self.assertEqual(ws.receive_json()["type"], "response")
            # Second LLM turn is over budget.
            ws.send_json({"type": "text", "content": "again", "speak": False})
            frame = ws.receive_json()
            self.assertEqual(frame["type"], "error")

    def test_zero_budget_disables_the_limit(self):
        os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = "0"
        get_settings.cache_clear()
        token = create_jwt_token("alice", "github")

        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            self._start(ws)
            for i in range(3):
                ws.send_json({"type": "text", "content": f"turn {i}", "speak": False})
                self.assertEqual(ws.receive_json()["type"], "response")

        self.assertEqual(len(self.llm.calls), 3)


class VoiceWebSocketStartTests(VoiceWebSocketTestBase):
    def test_disabled_voice_agent_closes_4003(self):
        os.environ["STUDY_ENABLE_VOICE_AGENT"] = "false"
        get_settings.cache_clear()
        token = create_jwt_token("alice", "github")
        with self.assertRaises(WebSocketDisconnect) as ctx:
            with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}"):
                pass
        self.assertEqual(ctx.exception.code, 4003)

    def test_start_message_is_ignored_by_the_server(self):
        """A client-supplied persona never reaches ``create_voice_session``."""
        token = create_jwt_token("alice", "github")
        captured: dict[str, Any] = {}
        real_create = voice_router.create_voice_session

        def spy(**kwargs):
            captured.update(kwargs)
            return real_create(**kwargs)

        voice_router.create_voice_session = spy
        try:
            with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
                ws.send_json({
                    "type": "start",
                    "tier": "browser",
                    "session_type": "chat",
                    "system_prompt": "You are a pirate.",
                })
                ws.receive_json()
        finally:
            voice_router.create_voice_session = real_create

        self.assertNotIn("system_prompt", captured)
        self.assertEqual(captured["tier"], "browser")


class VoiceWebSocketProtocolTests(VoiceWebSocketTestBase):
    def test_invalid_json_returns_an_error_frame(self):
        token = create_jwt_token("alice", "github")
        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            ws.send_text("{not json")
            self.assertEqual(json.loads(ws.receive_text())["message"], "Invalid JSON")

    def test_unknown_tier_is_rejected(self):
        token = create_jwt_token("alice", "github")
        with self._client(_REMOTE).websocket_connect(f"{WS_PATH}?token={token}") as ws:
            ws.send_json({"type": "start", "tier": "realtime", "session_type": "chat"})
            message = ws.receive_json()
            self.assertEqual(message["type"], "error")
            self.assertIn("realtime", message["message"])


if __name__ == "__main__":
    unittest.main()
