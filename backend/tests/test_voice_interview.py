"""Voice interview E2E — scripted STT/TTS, no external services.

Validation target (from the plan): a voice interview session
is created, the first question is TTS'd, a spoken answer is
transcribed, evaluated, and the rubric is returned. The
per-connection turn budget applies.
"""

import asyncio
import base64
import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.routers import interview_sessions, voice
from app.services.auth import create_jwt_token
from app.services.interview_store import InterviewStore
from app.services.interview_generator import InterviewGenerator
from app.services.learning_store import LearningStore
from app.services.voice_session import (
    VoiceInterviewSession,
    VoiceSession,
)


SCRIPTED_TRANSCRIPT = "I would use a token bucket in Redis with a refill rate."


class ScriptedSTT:
    """Offline STT: returns a fixed transcript."""

    def __init__(self, transcript: str = SCRIPTED_TRANSCRIPT):
        self.transcript = transcript
        self.calls = 0

    async def transcribe(self, audio_bytes, *, mime_type="audio/webm"):
        self.calls += 1
        return self.transcript


class ScriptedTTS:
    """Offline TTS: returns fake audio and records what was spoken."""

    def __init__(self):
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return b"\x00\x01\x02"

    async def synthesize_stream(self, text: str):
        yield b"\x00\x01\x02"


class ScriptedInterviewLLM:
    """Question generation + answer evaluation, discriminated by prompt."""

    def __init__(self):
        self.question_count = 0
        self.eval_count = 0

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        if "rubric" in prompt.lower():
            self.eval_count += 1
            return {
                "success": True,
                "analysis": (
                    '{"rubric":{"technical_accuracy":4,"reasoning_depth":4,'
                    '"communication_clarity":4,"completeness":4,'
                    '"confidence_signal":4,"overall":80},'
                    '"strengths":["Clear structure"],'
                    '"improvements":["Add a metric"],'
                    '"follow_up_note":"Quantify the outcome next time."}'
                ),
                "metadata": {},
                "error_code": "",
                "finish_reason": "stop",
                "usage": {},
            }
        self.question_count += 1
        question = f"Design a rate limiter for a bursty API, question {self.question_count}."
        return {
            "success": True,
            "analysis": json.dumps(
                {
                    "question": question,
                    "competency_focus": "system design",
                    "expected_signals": ["constraints", "tradeoffs"],
                }
            ),
            "metadata": {},
            "error_code": "",
            "finish_reason": "stop",
            "usage": {},
        }


class FakeParser:
    """Minimal DocParser stand-in for report building."""

    def list_topics(self, track=None):
        return []


def _run(coro):
    return asyncio.run(coro)


class VoiceInterviewSessionTests(unittest.TestCase):
    """Service-level E2E with scripted providers."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.llm = ScriptedInterviewLLM()
        self.stt = ScriptedSTT()
        self.tts = ScriptedTTS()
        self.store = InterviewStore(self.db_path)
        self.generator = InterviewGenerator(self.llm, FakeParser())
        self.session_row = self.store.create_session(
            user_id="u1",
            track="system-design",
            level="mid",
            interview_type="coding",
            turn_count=2,
            target_role="backend engineer",
            job_description_text="",
            resume_summary_text="",
            focus_areas=["system design"],
        )
        self.voice_session = VoiceSession(
            llm_client=self.llm,
            stt=self.stt,
            tts=self.tts,
            session_type="interview",
            session_id=self.session_row["session_id"],
            user_identity={"user": "u1", "provider": "local"},
        )
        self.interview = VoiceInterviewSession(
            voice_session=self.voice_session,
            store=self.store,
            generator=self.generator,
            session=self.session_row,
            user_id="u1",
            user_identity={"user": "u1", "provider": "local"},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_start_generates_and_speaks_the_first_question(self):
        started = _run(self.interview.start())
        self.assertTrue(started["question"].startswith("Design a rate limiter"))
        self.assertEqual(started["competency_focus"], "system design")
        self.assertEqual(started["expected_signals"], ["constraints", "tradeoffs"])
        # The question was spoken aloud and returned as audio.
        self.assertIn(started["question"], self.tts.spoken)
        self.assertEqual(started["audio_bytes"], b"\x00\x01\x02")
        # The question is persisted as the current question.
        context = self.store.get_session_context(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(context["current_question"], started["question"])

    def test_answer_is_transcribed_evaluated_and_recorded(self):
        started = _run(self.interview.start())
        result = _run(self.interview.answer(b"\x00" * 200))
        self.assertEqual(result["transcript"], SCRIPTED_TRANSCRIPT)
        self.assertEqual(result["rubric"]["overall"], 80)
        self.assertEqual(result["rubric"]["reasoning_depth"], 4)
        self.assertFalse(result["degraded"])
        self.assertFalse(result["completed"])
        # Feedback was spoken aloud.
        self.assertTrue(self.tts.spoken[-1].startswith("Overall score: 80"))
        # The turn was recorded with the transcript as the answer,
        # against the question that was asked.
        turns = self.store.get_turns(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["user_answer"], SCRIPTED_TRANSCRIPT)
        self.assertEqual(turns[0]["question"], started["question"])

    def test_turn_count_cap_completes_and_builds_report(self):
        _run(self.interview.start())
        first = _run(self.interview.answer(b"\x00" * 200))
        self.assertFalse(first["completed"])
        second = _run(self.interview.answer(b"\x00" * 200))
        self.assertTrue(second["completed"])
        self.assertIsNotNone(second["report"])
        self.assertEqual(second["report"]["overall_score"], 80.0)
        self.assertEqual(second["report"]["completed_turns"], 2)
        # The report is persisted.
        session = self.store.get_session(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertTrue(session["report_ready"])

    def test_answer_text_fallback(self):
        _run(self.interview.start())
        result = _run(self.interview.answer_text("Token bucket in Redis."))
        self.assertEqual(result["transcript"], "Token bucket in Redis.")
        self.assertEqual(result["rubric"]["overall"], 80)
        # STT was never called for a typed answer.
        self.assertEqual(self.stt.calls, 0)

    def test_empty_transcript_short_circuits(self):
        self.stt.transcript = ""
        _run(self.interview.start())
        result = _run(self.interview.answer(b"\x00" * 200))
        self.assertEqual(result["transcript"], "")
        self.assertIsNone(result["rubric"])
        # No turn was recorded.
        turns = self.store.get_turns(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(turns, [])


class VoiceInterviewWebSocketTests(unittest.TestCase):
    """WebSocket E2E: interview_start → interview_ready → answers."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.prev_env = {
            key: os.environ.get(key)
            for key in (
                "STUDY_ENABLE_VOICE_AGENT",
                "STUDY_ENABLE_VOICE_INTERVIEW_V1",
                "STUDY_VOICE_TIERS_ENABLED",
                "STUDY_VOICE_DEFAULT_TIER",
                "STUDY_AUTH_SECRET_KEY",
                "STUDY_LEARNING_DB_PATH",
            )
        }
        os.environ["STUDY_ENABLE_VOICE_AGENT"] = "true"
        os.environ["STUDY_ENABLE_VOICE_INTERVIEW_V1"] = "true"
        os.environ["STUDY_VOICE_TIERS_ENABLED"] = "browser,cloud"
        os.environ["STUDY_VOICE_DEFAULT_TIER"] = "browser"
        os.environ["STUDY_AUTH_SECRET_KEY"] = "a" * 64
        get_settings.cache_clear()

        self.llm = ScriptedInterviewLLM()
        self.stt = ScriptedSTT()
        self.tts = ScriptedTTS()

        interview_sessions.init(
            self.llm,
            FakeParser(),
            LearningStore(self.db_path),
            self.db_path,
        )
        voice.init(self.llm, None)

        # Route voice sessions to the scripted providers so the
        # E2E needs no external STT/TTS service.
        def _scripted_voice_session(
            llm_client,
            tier,
            session_type="chat",
            session_id="",
            user_identity=None,
            llm_config=None,
            system_prompt=None,
        ):
            return VoiceSession(
                llm_client=llm_client,
                stt=self.stt,
                tts=self.tts,
                session_type=session_type,
                session_id=session_id,
                user_identity=user_identity,
                llm_config=llm_config,
            )

        self._patcher = mock.patch(
            "app.routers.voice.create_voice_session",
            side_effect=_scripted_voice_session,
        )
        self._patcher.start()

        self.app = FastAPI()
        self.app.include_router(voice.router)
        self.client = TestClient(self.app)
        self.token = create_jwt_token("voice-user", "local")

    def tearDown(self):
        self._patcher.stop()
        self.tempdir.cleanup()
        for key, value in self.prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()

    def _connect(self):
        return self.client.websocket_connect(
            f"/api/voice/stream?token={self.token}"
        )

    def test_interview_start_reads_the_first_question(self):
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "interview_start",
                    "tier": "browser",
                    "config": {
                        "track": "system_design",
                        "level": "mid",
                        "interview_type": "coding",
                        "turn_count": 2,
                        "target_role": "backend engineer",
                        "focus_areas": ["system design"],
                    },
                }
            )
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "interview_ready")
            self.assertTrue(ready["question"].startswith("Design a rate limiter"))
            self.assertEqual(ready["turn_count"], 2)
            self.assertTrue(ready["audio"])
            self.assertEqual(
                base64.b64decode(ready["audio"]),
                b"\x00\x01\x02",
            )
            # The question was spoken by the TTS provider.
            self.assertIn(ready["question"], self.tts.spoken)

    def test_spoken_answer_is_evaluated_and_returned(self):
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "interview_start",
                    "tier": "browser",
                    "config": {"track": "system_design", "turn_count": 2},
                }
            )
            ws.receive_json()  # interview_ready

            ws.send_json(
                {
                    "type": "audio",
                    "data": base64.b64encode(b"\x00" * 200).decode(),
                    "mime": "audio/webm",
                }
            )
            transcript = ws.receive_json()
            self.assertEqual(transcript["type"], "transcript")
            self.assertEqual(transcript["text"], SCRIPTED_TRANSCRIPT)
            self.assertTrue(transcript["final"])

            response = ws.receive_json()
            self.assertEqual(response["type"], "interview_response")
            self.assertEqual(response["rubric"]["overall"], 80)
            self.assertFalse(response["degraded"])
            self.assertFalse(response["completed"])
            self.assertIn("Overall score: 80", response["text"])

    def test_interview_completes_at_turn_count(self):
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "interview_start",
                    "tier": "browser",
                    "config": {"track": "system_design", "turn_count": 1},
                }
            )
            ws.receive_json()  # interview_ready

            ws.send_json(
                {
                    "type": "audio",
                    "data": base64.b64encode(b"\x00" * 200).decode(),
                    "mime": "audio/webm",
                }
            )
            ws.receive_json()  # transcript
            response = ws.receive_json()
            self.assertTrue(response["completed"])
            completed = ws.receive_json()
            self.assertEqual(completed["type"], "interview_completed")
            self.assertEqual(completed["report"]["overall_score"], 80.0)
            self.assertEqual(completed["report"]["completed_turns"], 1)

    def test_interview_start_is_gated_by_the_rollout_flag(self):
        os.environ["STUDY_ENABLE_VOICE_INTERVIEW_V1"] = "false"
        get_settings.cache_clear()
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "interview_start",
                    "tier": "browser",
                    "config": {"track": "system_design"},
                }
            )
            error = ws.receive_json()
            self.assertEqual(error["type"], "error")
            self.assertIn("disabled", error["message"])

    def test_chat_start_still_works(self):
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "start",
                    "tier": "browser",
                    "session_type": "chat",
                }
            )
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "ready")


if __name__ == "__main__":
    unittest.main()
