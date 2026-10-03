"""Coverage for ``app.routers.voice`` — REST settings and the WS protocol.

The WebSocket is driven with the scope-rewriting ``TestClient`` pattern from
``test_voice_ws_auth.py`` (this Starlette version has no ``client=`` kwarg), and
``create_voice_session`` is swapped for an in-process fake so no STT/TTS/LLM
provider is ever contacted.
"""

import base64
import json
import os
import unittest
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import voice as voice_router
from app.services.auth import create_jwt_token

WS_PATH = "/api/voice/stream"
STRONG_SECRET = "v" * 64
_LOOPBACK = "127.0.0.1"
_REMOTE = "203.0.113.10"

#: The router rejects clips below this many decoded bytes.
MIN_AUDIO_BYTES = 100


def _long_audio(size: int = MIN_AUDIO_BYTES + 20) -> str:
    return base64.b64encode(b"\x01" * size).decode()


class FakeSession:
    """Stand-in for ``VoiceSession`` recording what the router asked for."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_id = kwargs.get("session_id", "")
        self.conversation_history: list[dict[str, str]] = []
        self.turn_index = 0
        self.cleared = 0
        self.audio_result: dict[str, Any] = {
            "transcript": "the user spoke",
            "response_text": "here is the reply",
            "audio_bytes": b"\x09\x08\x07",
            "stt_latency_ms": 11,
            "llm_latency_ms": 22,
            "tts_latency_ms": 33,
            "total_latency_ms": 66,
        }
        self.llm_text = "text turn reply"
        self.speech_bytes = b"\x0a\x0b"
        self.llm_raises: Exception | None = None

    async def process_audio_turn(self, audio_bytes: bytes, mime_type: str = "audio/webm"):
        self.kwargs["last_audio_len"] = len(audio_bytes)
        self.kwargs["last_mime"] = mime_type
        return self.audio_result

    async def _call_llm(self, text: str) -> str:
        if self.llm_raises is not None:
            raise self.llm_raises
        self.kwargs["last_llm_text"] = text
        return self.llm_text

    async def generate_speech(self, text: str) -> bytes:
        self.kwargs.setdefault("spoken", []).append(text)
        return self.speech_bytes

    def clear_history(self) -> None:
        self.cleared += 1
        self.conversation_history.clear()
        self.turn_index = 0


def _scope_override(app: FastAPI, *, client_host: str, request_host: str = "localhost"):
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


class VoiceRouterTestBase(unittest.TestCase):
    env_keys = [
        "STUDY_ENABLE_VOICE_AGENT",
        "STUDY_VOICE_TIERS_ENABLED",
        "STUDY_VOICE_DEFAULT_TIER",
        "STUDY_VOICE_STT_PROVIDER",
        "STUDY_VOICE_STT_MODEL",
        "STUDY_VOICE_TTS_PROVIDER",
        "STUDY_VOICE_TTS_VOICE",
        "STUDY_VOICE_MAX_TURNS_PER_CONNECTION",
        "STUDY_DEV_AUTH_BYPASS_LOCALHOST",
        "STUDY_ENVIRONMENT",
        "STUDY_FRONTEND_URL",
        "STUDY_AUTH_SECRET_KEY",
        "STUDY_ADMIN_USERS",
    ]

    def setUp(self):
        self.prev_env = {key: os.environ.get(key) for key in self.env_keys}
        os.environ["STUDY_ENABLE_VOICE_AGENT"] = "true"
        os.environ["STUDY_VOICE_TIERS_ENABLED"] = "browser,cloud"
        os.environ["STUDY_VOICE_DEFAULT_TIER"] = "browser"
        os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = "40"
        os.environ["STUDY_DEV_AUTH_BYPASS_LOCALHOST"] = "false"
        os.environ["STUDY_ENVIRONMENT"] = "production"
        os.environ["STUDY_FRONTEND_URL"] = "http://localhost:5174"
        os.environ["STUDY_AUTH_SECRET_KEY"] = STRONG_SECRET
        os.environ["STUDY_ADMIN_USERS"] = "google:admin@example.com"
        get_settings.cache_clear()

        self.app = FastAPI()
        self.app.include_router(voice_router.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "admin@example.com",
            "provider": "google",
        }
        self.client = TestClient(self.app)

        self.sessions: list[FakeSession] = []
        self._prev_create = voice_router.create_voice_session
        voice_router.create_voice_session = self._make_session
        self._prev_llm = voice_router._llm_client
        voice_router._llm_client = object()

    def _make_session(self, **kwargs) -> FakeSession:
        session = FakeSession(**kwargs)
        self.sessions.append(session)
        return session

    def tearDown(self):
        voice_router.create_voice_session = self._prev_create
        voice_router._llm_client = self._prev_llm
        self.app.dependency_overrides.clear()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _ws_client(self) -> TestClient:
        return TestClient(
            _scope_override(self.app, client_host=_REMOTE, request_host="localhost")
        )

    def _open(self, client=None, *, max_turns=None):
        if max_turns is not None:
            os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = str(max_turns)
            get_settings.cache_clear()
        token = create_jwt_token("alice", "github")
        return (client or self._ws_client()).websocket_connect(f"{WS_PATH}?token={token}")

    def _start(self, ws, **extra):
        payload = {"type": "start", "tier": "browser", "session_type": "chat"}
        payload.update(extra)
        ws.send_json(payload)
        ready = ws.receive_json()
        self.assertEqual(ready["type"], "ready")
        return ready


class VoiceSettingsUpdateTests(VoiceRouterTestBase):
    def test_empty_body_is_rejected(self):
        response = self.client.put("/api/voice/settings", json={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "No fields to update")

    def test_settings_update_writes_the_mapped_env_vars_and_lowercases_booleans(self):
        response = self.client.put(
            "/api/voice/settings",
            json={
                "enable_voice_agent": True,
                "voice_tiers_enabled": "browser,cloud",
                "voice_default_tier": "cloud",
                "voice_stt_provider": "openai",
                "voice_stt_model": "whisper-1",
                "voice_tts_provider": "openai",
                "voice_tts_voice": "alloy",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["applied"]["enable_voice_agent"], "true")
        self.assertEqual(payload["applied"]["voice_default_tier"], "cloud")
        self.assertEqual(payload["applied"]["voice_tts_voice"], "alloy")

        self.assertEqual(os.environ["STUDY_ENABLE_VOICE_AGENT"], "true")
        self.assertEqual(os.environ["STUDY_VOICE_TIERS_ENABLED"], "browser,cloud")
        self.assertEqual(os.environ["STUDY_VOICE_STT_PROVIDER"], "openai")
        self.assertEqual(os.environ["STUDY_VOICE_STT_MODEL"], "whisper-1")
        self.assertEqual(os.environ["STUDY_VOICE_TTS_PROVIDER"], "openai")

        # The settings cache is dropped so the new values take effect.
        get_settings.cache_clear()
        self.assertTrue(get_settings().enable_voice_agent)
        self.assertEqual(get_settings().voice_default_tier, "cloud")

    def test_partial_update_leaves_untouched_fields_at_their_current_value(self):
        self.client.put(
            "/api/voice/settings", json={"voice_default_tier": "cloud"}
        )
        response = self.client.put(
            "/api/voice/settings", json={"voice_stt_model": "whisper-large-v3-turbo"}
        )
        self.assertEqual(list(response.json()["applied"]), ["voice_stt_model"])
        self.assertEqual(os.environ["STUDY_VOICE_DEFAULT_TIER"], "cloud")

    def test_boolean_false_is_applied_as_the_string_false(self):
        response = self.client.put(
            "/api/voice/settings", json={"enable_voice_agent": False}
        )
        self.assertEqual(response.json()["applied"]["enable_voice_agent"], "false")
        self.assertEqual(os.environ["STUDY_ENABLE_VOICE_AGENT"], "false")

    def test_settings_update_requires_admin(self):
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "member@example.com",
            "provider": "google",
        }
        response = self.client.put(
            "/api/voice/settings", json={"voice_default_tier": "cloud"}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Admin access required")


class VoiceServiceInitialisationTests(VoiceRouterTestBase):
    def test_uninitialised_service_reports_an_error_and_closes_4503(self):
        voice_router._llm_client = None
        with self._open() as ws:
            frame = ws.receive_json()
            self.assertEqual(frame, {"type": "error", "message": "Service not initialised"})
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 4503)


class VoiceStartMessageTests(VoiceRouterTestBase):
    def test_invalid_llm_config_is_discarded_and_the_session_still_starts(self):
        with self._open() as ws:
            ready = self._start(ws, llm_config={"temperature": "extremely hot"})
            self.assertTrue(ready["session_id"])

        session = self.sessions[-1]
        self.assertIsNone(session.kwargs["llm_config"])

    def test_valid_llm_config_is_forwarded_to_the_session_factory(self):
        with self._open() as ws:
            self._start(ws, llm_config={"provider": "openai", "model": "gpt-4o-mini"})

        forwarded = self.sessions[-1].kwargs["llm_config"]
        self.assertIsNotNone(forwarded)
        self.assertEqual(forwarded.provider, "openai")

    def test_start_honours_a_client_supplied_session_id(self):
        with self._open() as ws:
            ready = self._start(ws, session_id="sess-42")
        self.assertEqual(ready["session_id"], "sess-42")
        self.assertEqual(ready["tier"], "browser")

    def test_start_defaults_to_the_configured_tier_when_none_is_sent(self):
        with self._open() as ws:
            ws.send_json({"type": "start"})
            ready = ws.receive_json()
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["tier"], "browser")


class VoiceAudioTurnTests(VoiceRouterTestBase):
    def test_audio_before_start_is_rejected(self):
        with self._open() as ws:
            ws.send_json({"type": "audio", "data": _long_audio()})
            frame = ws.receive_json()
        self.assertEqual(frame["type"], "error")
        self.assertIn("Session not started", frame["message"])

    def test_undecodable_base64_audio_is_rejected(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json({"type": "audio", "data": "abc"})
            frame = ws.receive_json()
        self.assertEqual(frame["message"], "Invalid base64 audio")

    def test_audio_below_the_minimum_length_is_rejected(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json(
                {"type": "audio", "data": base64.b64encode(b"\x01" * 10).decode()}
            )
            frame = ws.receive_json()
        self.assertEqual(frame["message"], "Audio too short")

    def test_audio_turn_emits_transcript_then_response_with_encoded_audio(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json(
                {"type": "audio", "data": _long_audio(), "mime": "audio/wav"}
            )
            transcript = ws.receive_json()
            response = ws.receive_json()

        self.assertEqual(
            transcript, {"type": "transcript", "text": "the user spoke", "final": True}
        )
        self.assertEqual(response["type"], "response")
        self.assertEqual(response["text"], "here is the reply")
        self.assertEqual(
            base64.b64decode(response["audio"]), b"\x09\x08\x07"
        )
        self.assertEqual(
            response["latency"],
            {"stt_ms": 11, "llm_ms": 22, "tts_ms": 33, "total_ms": 66},
        )
        session = self.sessions[-1]
        self.assertEqual(session.kwargs["last_audio_len"], MIN_AUDIO_BYTES + 20)
        self.assertEqual(session.kwargs["last_mime"], "audio/wav")

    def test_audio_turn_with_an_empty_transcript_skips_the_transcript_frame(self):
        with self._open() as ws:
            self._start(ws)
            self.sessions[-1].audio_result = {
                "transcript": "",
                "response_text": "ok",
                "audio_bytes": b"",
                "stt_latency_ms": 1,
                "llm_latency_ms": 2,
                "tts_latency_ms": 3,
                "total_latency_ms": 6,
            }
            ws.send_json({"type": "audio", "data": _long_audio()})
            response = ws.receive_json()

        self.assertEqual(response["type"], "response")
        self.assertEqual(response["audio"], "")

    def test_audio_turn_error_is_forwarded_to_the_client(self):
        with self._open() as ws:
            self._start(ws)
            self.sessions[-1].audio_result = {"error": "Speech recognition failed: STT down"}
            ws.send_json({"type": "audio", "data": _long_audio()})
            frame = ws.receive_json()
            # The connection survives the error frame.
            self._start(ws)

        self.assertEqual(frame["type"], "error")
        self.assertIn("STT down", frame["message"])

    def test_audio_turn_budget_exhaustion_closes_the_connection(self):
        with self._open(max_turns=1) as ws:
            self._start(ws)
            ws.send_json({"type": "audio", "data": _long_audio()})
            self.assertEqual(ws.receive_json()["type"], "transcript")
            self.assertEqual(ws.receive_json()["type"], "response")

            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.send_json({"type": "audio", "data": _long_audio()})
                frame = ws.receive_json()
                self.assertEqual(frame["type"], "error")
                self.assertIn("budget", frame["message"].lower())
                ws.receive_json()

            self.assertEqual(ctx.exception.code, 4429)


class VoiceTextTurnTests(VoiceRouterTestBase):
    def test_text_before_start_is_rejected(self):
        with self._open() as ws:
            ws.send_json({"type": "text", "content": "hello"})
            frame = ws.receive_json()
        self.assertEqual(frame["type"], "error")
        self.assertIn("Session not started", frame["message"])

    def test_blank_text_is_ignored_without_spending_a_turn(self):
        with self._open(max_turns=1) as ws:
            self._start(ws)
            ws.send_json({"type": "text", "content": "   "})
            ws.send_json({"type": "text", "content": "real question", "speak": False})
            frame = ws.receive_json()
        self.assertEqual(frame["type"], "response")

    def test_text_turn_without_speak_returns_no_audio(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json({"type": "text", "content": "hello", "speak": False})
            frame = ws.receive_json()

        self.assertEqual(frame["type"], "response")
        self.assertEqual(frame["text"], "text turn reply")
        self.assertEqual(frame["audio"], "")
        self.assertEqual(frame["latency"]["stt_ms"], 0)
        self.assertEqual(frame["latency"]["tts_ms"], 0)
        self.assertEqual(self.sessions[-1].kwargs.get("spoken"), None)

    def test_text_turn_with_speak_returns_encoded_audio(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json({"type": "text", "content": "hello", "speak": True})
            frame = ws.receive_json()

        self.assertEqual(base64.b64decode(frame["audio"]), b"\x0a\x0b")
        self.assertEqual(self.sessions[-1].kwargs["spoken"], ["text turn reply"])
        self.assertEqual(
            self.sessions[-1].conversation_history,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "text turn reply"},
            ],
        )
        self.assertEqual(self.sessions[-1].turn_index, 1)

    def test_unexpected_llm_failure_closes_with_4500(self):
        with self._open() as ws:
            self._start(ws)
            self.sessions[-1].llm_raises = RuntimeError("llm exploded")
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.send_json({"type": "text", "content": "hello", "speak": False})
                frame = ws.receive_json()
                self.assertEqual(frame, {"type": "error", "message": "Internal server error"})
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 4500)

    def test_text_turn_with_a_failing_tts_returns_no_audio_but_still_replies(self):
        with self._open() as ws:
            self._start(ws)
            self.sessions[-1].speech_bytes = b""
            ws.send_json({"type": "text", "content": "hello", "speak": True})
            frame = ws.receive_json()

        self.assertEqual(frame["type"], "response")
        self.assertEqual(frame["text"], "text turn reply")
        self.assertEqual(frame["audio"], "")
        # Latency is still reported even though synthesis produced nothing.
        self.assertEqual(frame["latency"]["stt_ms"], 0)
        self.assertGreaterEqual(frame["latency"]["tts_ms"], 0)
        self.assertEqual(
            frame["latency"]["total_ms"],
            frame["latency"]["llm_ms"] + frame["latency"]["tts_ms"],
        )


class VoiceSynthesizeTests(VoiceRouterTestBase):
    def test_synthesize_before_start_is_rejected(self):
        with self._open() as ws:
            ws.send_json({"type": "synthesize", "text": "read this"})
            frame = ws.receive_json()
        self.assertEqual(frame["message"], "Session not started.")

    def test_synthesize_with_blank_text_is_ignored(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json({"type": "synthesize", "text": "  "})
            ws.send_json({"type": "synthesize", "text": "read this"})
            frame = ws.receive_json()
        self.assertEqual(frame["type"], "audio")
        self.assertEqual(self.sessions[-1].kwargs["spoken"], ["read this"])

    def test_synthesize_returns_an_audio_frame_with_the_spoken_text(self):
        with self._open() as ws:
            self._start(ws)
            ws.send_json({"type": "synthesize", "text": "read this"})
            frame = ws.receive_json()

        self.assertEqual(frame["type"], "audio")
        self.assertEqual(frame["text"], "read this")
        self.assertEqual(base64.b64decode(frame["audio"]), b"\x0a\x0b")

    def test_synthesize_reports_an_empty_payload_when_tts_returns_nothing(self):
        with self._open() as ws:
            self._start(ws)
            self.sessions[-1].speech_bytes = b""
            ws.send_json({"type": "synthesize", "text": "silence"})
            frame = ws.receive_json()
        self.assertEqual(frame["audio"], "")


class VoiceStopAndUnknownTypeTests(VoiceRouterTestBase):
    def test_stop_clears_the_session_so_later_turns_are_rejected(self):
        with self._open() as ws:
            self._start(ws)
            session = self.sessions[-1]
            ws.send_json({"type": "text", "content": "hello", "speak": False})
            ws.receive_json()

            ws.send_json({"type": "stop"})
            stopped = ws.receive_json()
            self.assertEqual(stopped, {"type": "stopped", "session_id": session.session_id})
            self.assertEqual(session.cleared, 1)

            ws.send_json({"type": "text", "content": "again", "speak": False})
            frame = ws.receive_json()

        self.assertIn("Session not started", frame["message"])

    def test_stop_without_a_session_is_still_acknowledged(self):
        with self._open() as ws:
            ws.send_json({"type": "stop"})
            stopped = ws.receive_json()
        self.assertEqual(stopped["type"], "stopped")
        self.assertEqual(stopped["session_id"], "")

    def test_unknown_message_type_is_reported_verbatim(self):
        with self._open() as ws:
            ws.send_text(json.dumps({"type": "teleport", "extra": 1}))
            frame = ws.receive_json()
            ws.send_text(json.dumps({"payload": "no type at all"}))
            second = ws.receive_json()

        self.assertEqual(frame, {"type": "error", "message": "Unknown message type: teleport"})
        self.assertEqual(second["message"], "Unknown message type: ")

    def test_invalid_json_is_reported_and_the_socket_survives(self):
        with self._open() as ws:
            ws.send_text("{not json")
            self.assertEqual(
                ws.receive_json(), {"type": "error", "message": "Invalid JSON"}
            )
            self._start(ws)


if __name__ == "__main__":
    unittest.main()