"""Voice enhancement tests — response-time, barge-in interrupt, JD→plan.

Covers:
- response-time round-trip through ``record_turn``
- ``interrupt`` cancels an in-flight turn (service + WebSocket level)
- an interrupted turn that never reached the LLM does not consume
  an LLM turn of the per-connection budget
- JD→topic mapping (high-confidence match, unmapped gap, no-JD case)
- ``POST /api/learning/plan-from-jd`` endpoint contract
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
from app.dependencies import require_auth
from app.routers import interview_sessions, learning, voice
from app.routers.learning import map_skills_to_topic_ids
from app.services.auth import create_jwt_token
from app.services.interview_generator import InterviewGenerator
from app.services.interview_store import InterviewStore
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore
from app.services.voice_session import VoiceInterviewSession, VoiceSession


SCRIPTED_TRANSCRIPT = "I would use a token bucket in Redis with a refill rate."


class ScriptedSTT:
    """Offline STT: returns a fixed transcript."""

    def __init__(self, transcript: str = SCRIPTED_TRANSCRIPT):
        self.transcript = transcript
        self.calls = 0

    async def transcribe(self, audio_bytes, *, mime_type="audio/webm"):
        self.calls += 1
        return self.transcript


class BlockingSTT:
    """STT whose first call blocks until the turn is cancelled."""

    def __init__(self, transcript: str = SCRIPTED_TRANSCRIPT):
        self.transcript = transcript
        self.calls = 0

    async def transcribe(self, audio_bytes, *, mime_type="audio/webm"):
        self.calls += 1
        if self.calls == 1:
            # The first turn never completes on its own;
            # only a client interrupt cuts it short.
            await asyncio.sleep(3600)
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

    async def completion(
        self, prompt, llm_config=None, user_identity=None, **kwargs
    ):
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
        question = (
            f"Design a rate limiter for a bursty API, "
            f"question {self.question_count}."
        )
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


class FakeTopic:
    """Minimal curriculum topic for JD→topic mapping tests."""

    def __init__(self, topic_id: str, title: str):
        self.id = topic_id
        self.title = title


def _run(coro):
    return asyncio.run(coro)


class VoiceInterviewResponseTimeTests(unittest.TestCase):
    """The spoken answer's duration round-trips into the turn."""

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
        self.interview = VoiceInterviewSession(
            voice_session=VoiceSession(
                llm_client=self.llm,
                stt=self.stt,
                tts=self.tts,
                session_type="interview",
                session_id=self.session_row["session_id"],
                user_identity={"user": "u1", "provider": "local"},
            ),
            store=self.store,
            generator=self.generator,
            session=self.session_row,
            user_id="u1",
            user_identity={"user": "u1", "provider": "local"},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_answer_records_response_time_and_latency_breakdown(self):
        _run(self.interview.start())
        result = _run(
            self.interview.answer(b"\x00" * 200, response_time_ms=4242)
        )
        # The result carries the client-measured duration and the
        # per-stage latency breakdown.
        self.assertEqual(result["response_time_ms"], 4242)
        for stage in ("stt_ms", "llm_ms", "tts_ms", "total_ms"):
            self.assertIn(stage, result["latency"])
        self.assertGreaterEqual(result["latency"]["total_ms"], 0)
        # The duration round-trips through record_turn into the store.
        turns = self.store.get_turns(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["response_time_ms"], 4242)

    def test_answer_defaults_response_time_to_zero(self):
        _run(self.interview.start())
        result = _run(self.interview.answer(b"\x00" * 200))
        self.assertEqual(result["response_time_ms"], 0)
        turns = self.store.get_turns(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(turns[0]["response_time_ms"], 0)


class VoiceInterviewInterruptTests(unittest.TestCase):
    """A client interrupt cuts the in-flight answer short."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.llm = ScriptedInterviewLLM()
        self.stt = BlockingSTT()
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
        self.interview = VoiceInterviewSession(
            voice_session=VoiceSession(
                llm_client=self.llm,
                stt=self.stt,
                tts=self.tts,
                session_type="interview",
                session_id=self.session_row["session_id"],
                user_identity={"user": "u1", "provider": "local"},
            ),
            store=self.store,
            generator=self.generator,
            session=self.session_row,
            user_id="u1",
            user_identity={"user": "u1", "provider": "local"},
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_interrupt_cancels_in_flight_answer(self):
        _run(self.interview.start())

        async def _scenario():
            task = asyncio.ensure_future(self.interview.answer(b"\x00" * 200))
            # Let the turn start and block inside STT.
            await asyncio.sleep(0.05)
            self.interview.interrupt()
            return await task

        result = _run(_scenario())
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["transcript"], "")
        self.assertIsNone(result["rubric"])
        # The interrupted turn was never recorded.
        turns = self.store.get_turns(
            user_id="u1", session_id=self.session_row["session_id"]
        )
        self.assertEqual(turns, [])
        # The evaluation LLM was never reached.
        self.assertEqual(self.llm.eval_count, 0)

    def test_interrupt_without_in_flight_turn_is_a_noop(self):
        self.interview.interrupt()
        self.assertFalse(self.interview.llm_reached)


class VoiceInterruptWebSocketTests(unittest.TestCase):
    """WebSocket E2E: an ``interrupt`` message cancels the in-flight turn."""

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
                "STUDY_VOICE_MAX_TURNS_PER_CONNECTION",
                "STUDY_AUTH_SECRET_KEY",
                "STUDY_LEARNING_DB_PATH",
            )
        }
        os.environ["STUDY_ENABLE_VOICE_AGENT"] = "true"
        os.environ["STUDY_ENABLE_VOICE_INTERVIEW_V1"] = "true"
        os.environ["STUDY_VOICE_TIERS_ENABLED"] = "browser,cloud"
        os.environ["STUDY_VOICE_DEFAULT_TIER"] = "browser"
        # Two LLM turns per connection: the interview start
        # spends one (pre-existing behaviour), so a single
        # answer turn fits — and an interrupted answer must
        # be refunded for the second answer to fit.
        os.environ["STUDY_VOICE_MAX_TURNS_PER_CONNECTION"] = "2"
        os.environ["STUDY_AUTH_SECRET_KEY"] = "a" * 64
        os.environ["STUDY_LEARNING_DB_PATH"] = self.db_path
        get_settings.cache_clear()

        self.llm = ScriptedInterviewLLM()
        self.stt = BlockingSTT()
        self.tts = ScriptedTTS()

        interview_sessions.init(
            self.llm,
            FakeParser(),
            LearningStore(self.db_path),
            self.db_path,
        )
        voice.init(self.llm, None)

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

    def test_interrupt_message_cancels_turn_and_refunds_budget(self):
        with self._connect() as ws:
            ws.send_json(
                {
                    "type": "interview_start",
                    "tier": "browser",
                    "config": {"track": "system_design", "turn_count": 2},
                }
            )
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "interview_ready")
            session_id = ready["session_id"]

            # Turn 1 blocks inside STT; the interrupt cuts it short
            # before the evaluation LLM is reached.
            ws.send_json(
                {
                    "type": "audio",
                    "data": base64.b64encode(b"\x00" * 200).decode(),
                    "mime": "audio/webm",
                    "response_time_ms": 1500,
                }
            )
            ws.send_json({"type": "interrupt"})

            # Turn 2 still fits the budget (the interrupted turn was
            # refunded) and completes normally.
            ws.send_json(
                {
                    "type": "audio",
                    "data": base64.b64encode(b"\x00" * 200).decode(),
                    "mime": "audio/webm",
                    "response_time_ms": 900,
                }
            )
            transcript = ws.receive_json()
            self.assertEqual(transcript["type"], "transcript")
            self.assertEqual(transcript["text"], SCRIPTED_TRANSCRIPT)
            response = ws.receive_json()
            self.assertEqual(response["type"], "interview_response")
            self.assertFalse(response["degraded"])
            self.assertEqual(response["response_time_ms"], 900)
            for stage in ("stt_ms", "llm_ms", "tts_ms", "total_ms"):
                self.assertIn(stage, response["latency"])

            # Only the completed turn was recorded.
            turns = interview_sessions._store.get_turns(
                user_id="voice-user", session_id=session_id
            )
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["response_time_ms"], 900)


class JdTopicMappingTests(unittest.TestCase):
    """JD gaps map to curriculum topics on high-confidence matches only."""

    TOPICS = [
        FakeTopic("t-redis", "Redis Caching"),
        FakeTopic("t-sql", "SQL Databases"),
        FakeTopic("t-http", "HTTP and REST"),
    ]

    def test_exact_match_maps_to_topic_id(self):
        mapped, unmapped = map_skills_to_topic_ids(
            ["Redis Caching"], self.TOPICS
        )
        self.assertEqual(mapped, {"Redis Caching": "t-redis"})
        self.assertEqual(unmapped, [])

    def test_contained_match_maps_when_long_enough(self):
        # "Redis" (5 chars) is contained in "Redis Caching".
        mapped, unmapped = map_skills_to_topic_ids(["Redis"], self.TOPICS)
        self.assertEqual(mapped, {"Redis": "t-redis"})
        self.assertEqual(unmapped, [])

    def test_short_substrings_do_not_match(self):
        # "SQL" normalises to 3 characters — below the floor that
        # keeps "sql" inside "mysql" from false-positiving.
        mapped, unmapped = map_skills_to_topic_ids(["SQL"], self.TOPICS)
        self.assertEqual(mapped, {})
        self.assertEqual(unmapped, ["SQL"])

    def test_unmapped_gap_stays_a_plain_string(self):
        mapped, unmapped = map_skills_to_topic_ids(
            ["Quantum Computing"], self.TOPICS
        )
        self.assertEqual(mapped, {})
        self.assertEqual(unmapped, ["Quantum Computing"])

    def test_no_jd_case_maps_nothing(self):
        mapped, unmapped = map_skills_to_topic_ids([], self.TOPICS)
        self.assertEqual(mapped, {})
        self.assertEqual(unmapped, [])

    def test_duplicates_and_blanks_are_deduplicated(self):
        mapped, unmapped = map_skills_to_topic_ids(
            ["Redis", "redis", "  ", "Redis"], self.TOPICS
        )
        self.assertEqual(mapped, {"Redis": "t-redis"})
        self.assertEqual(unmapped, [])


class FakeJDAnalyzer:
    def __init__(self, gaps, focus_areas):
        self.gaps = gaps
        self.focus_areas = focus_areas
        self.calls: list[dict] = []

    async def analyze(
        self, *, jd_text, profile, llm_config=None, user_identity=None
    ):
        self.calls.append(
            {
                "jd_text": jd_text,
                "profile": profile,
                "llm_config": llm_config,
                "user_identity": user_identity,
            }
        )
        return {
            "matched": ["Python"],
            "gaps": self.gaps,
            "weak_spots": [],
            "likely_followups": [],
            "recommended_focus_areas": self.focus_areas,
        }


class FakeTopicParser:
    def __init__(self, topics):
        self.topics = topics

    def list_topics(self, track=None):
        return self.topics


class FakeResumeStore:
    def __init__(self, profile=None):
        self.profile = profile

    def get(self, user_id):
        return self.profile


class PlanFromJDTests(unittest.TestCase):
    """POST /api/learning/plan-from-jd contract."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.learning_store = LearningStore(self.db_path)
        self.planner_store = LearningPlannerStore(self.db_path)
        learning.init(self.learning_store, self.planner_store)

        self.app = FastAPI()
        self.app.include_router(learning.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "jd-user",
            "provider": "local",
        }
        self.client = TestClient(self.app)

        self._prev_services = interview_sessions.get_jd_analysis_services
        self.topics = [
            FakeTopic("t-redis", "Redis Caching"),
            FakeTopic("t-sql", "SQL Databases"),
        ]
        self.analyzer = FakeJDAnalyzer(
            gaps=["Redis Caching"],
            focus_areas=["Database Indexing"],
        )
        self._patch_services(
            self.analyzer,
            FakeTopicParser(self.topics),
            FakeResumeStore({"user_id": "jd-user", "skills": ["Python"]}),
        )

    def tearDown(self):
        interview_sessions.get_jd_analysis_services = self._prev_services
        self.tempdir.cleanup()

    def _patch_services(self, analyzer, parser, resume_store):
        interview_sessions.get_jd_analysis_services = lambda: (
            analyzer,
            parser,
            resume_store,
        )

    def test_plan_from_jd_maps_gaps_and_persists_focus_topics(self):
        res = self.client.post(
            "/api/learning/plan-from-jd",
            json={"jd_text": "Needs Redis Caching experience.", "days": 2, "daily_items": 2},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        # High-confidence gap matched a curriculum topic.
        self.assertEqual(body["mapped_topic_ids"], ["t-redis"])
        # No confident match — stays a plain focus-area string.
        self.assertEqual(body["unmapped_focus_areas"], ["Database Indexing"])
        self.assertEqual(body["focus_topic_ids"], ["t-redis"])
        # The study plan was built from the updated profile.
        self.assertEqual(body["days"], 2)
        self.assertEqual(len(body["days_plan"]), 2)
        # The mapped topic was persisted as a learner focus topic.
        profile = self.client.get("/api/learning/profile")
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["focus_topic_ids"], ["t-redis"])
        # The JD was analysed against the stored resume profile.
        self.assertEqual(len(self.analyzer.calls), 1)
        self.assertEqual(self.analyzer.calls[0]["jd_text"], "Needs Redis Caching experience.")
        self.assertEqual(self.analyzer.calls[0]["profile"]["skills"], ["Python"])

    def test_plan_from_jd_merges_with_existing_focus_topics(self):
        self.client.put(
            "/api/learning/profile",
            json={"focus_topic_ids": ["t-http"]},
        )
        res = self.client.post(
            "/api/learning/plan-from-jd",
            json={"jd_text": "Needs Redis Caching experience."},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["focus_topic_ids"], ["t-http", "t-redis"])

    def test_plan_from_jd_rejects_empty_jd(self):
        res = self.client.post(
            "/api/learning/plan-from-jd",
            json={"jd_text": ""},
        )
        self.assertEqual(res.status_code, 422)

    def test_plan_from_jd_disabled_with_interview_plus(self):
        os.environ["STUDY_ENABLE_INTERVIEW_PLUS_V1"] = "false"
        get_settings.cache_clear()
        try:
            res = self.client.post(
                "/api/learning/plan-from-jd",
                json={"jd_text": "Needs Redis."},
            )
            self.assertEqual(res.status_code, 404)
        finally:
            os.environ.pop("STUDY_ENABLE_INTERVIEW_PLUS_V1", None)
            get_settings.cache_clear()

    def test_plan_from_jd_unavailable_services(self):
        interview_sessions.get_jd_analysis_services = lambda: None
        res = self.client.post(
            "/api/learning/plan-from-jd",
            json={"jd_text": "Needs Redis."},
        )
        self.assertEqual(res.status_code, 503)


if __name__ == "__main__":
    unittest.main()
