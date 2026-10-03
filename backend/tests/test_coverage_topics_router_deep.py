"""Coverage for app/routers/topics.py.

Pins the router's contract edges that the happy-path tests skip: the service
gates (503), the 404/422 rejections on every topic-scoped route, the list
filters, the video feature routes, both NDJSON streams' cached / dict-payload /
missing-done / policy-error frames, and the pure helpers behind them
(``_llm_error_code``, ``_profile_is_stale``, ``_passes_topic_filters``).

Every route is exercised through an in-process ``TestClient`` with a real
``DocParser``/``LearningStore`` rooted in a temp directory; the LLM is a fake
that replays canned JSON, so nothing here can open a socket.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import optional_auth, require_auth
from app.routers import topics
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore
from app.services.llm_client import BUDGET_EXCEEDED_CODE
from app.services.llm_policy import (
    APPROVAL_REQUIRED_CODE,
    LLMServiceApprovalRequiredError,
)

PROBLEM_SOLVING_ID = "00-problem-solving-and-algorithms"


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if '"requires_programming"' in prompt:
            payload = {
                "requires_programming": True,
                "language_options": ["python", "go", "typescript"],
            }
        else:
            payload = {
                "title": "Custom Topic",
                "description": "Custom desc",
                "track": "backend",
                "levels": ["junior", "mid", "senior"],
                "sections": [
                    {"heading": f"Section {i}", "content": "Content for section."}
                    for i in range(1, 101)
                ],
            }
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


class StubVideoRecommender:
    def __init__(self, status=None, result=None):
        self._status = status or {"enabled": True, "status": "enabled", "reason": ""}
        self._result = result or {"enabled": True, "status": "enabled", "videos": []}
        self.recommend_calls: list[dict] = []

    async def get_status_async(self) -> dict:
        return dict(self._status)

    async def recommend(self, **kwargs) -> dict:
        self.recommend_calls.append(kwargs)
        return dict(self._result)


class StubMCP:
    def __init__(self):
        self.calls: list[dict] = []

    async def gather_context(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "The team standardises on Python and Go."


class RecordingStore(LearningStore):
    """A real store plus a call log, so router delegation can be asserted."""

    def __init__(self, path):
        super().__init__(path)
        self.calls: list[tuple[str, dict]] = []

    def run_async(self, fn, *args, **kwargs):
        name = getattr(fn, "__name__", repr(fn))
        self.calls.append((name, kwargs))
        return super().run_async(fn, *args, **kwargs)

    def call_names(self) -> list[str]:
        return [name for name, _ in self.calls]


class TopicsRouterDeepTests(unittest.TestCase):
    def setUp(self):
        self._prev = (
            topics._parser,
            topics._llm_client,
            topics._learning_store,
            topics._mcp_gateway,
            topics._video_recommender,
        )
        self.tempdir = tempfile.TemporaryDirectory()
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as handle:
            handle.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static description about caching.

## Static Section

Section body.
"""
            )
        self.parser = DocParser(docs_path=self.tempdir.name)
        self.store = RecordingStore(os.path.join(self.tempdir.name, "learning.db"))
        self.llm = FakeLLM()
        self.videos = StubVideoRecommender()
        topics.init(self.parser, self.llm, self.store, mcp_gateway=None)
        topics._video_recommender = self.videos

        # Two clients over the same router: one authenticated, one anonymous.
        # ``GET /api/topics`` and ``GET /api/topics/{id}`` use ``optional_auth``,
        # so the anonymous paths need a client that does not override it.
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "alice",
            "provider": "local",
        }
        self.app.dependency_overrides[optional_auth] = lambda: {
            "user": "alice",
            "provider": "local",
        }
        self.app.include_router(topics.router)
        self.app.include_router(topics.features_router)
        self.client = TestClient(self.app)

        self.anon_app = FastAPI()
        self.anon_app.include_router(topics.router)
        self.anon_client = TestClient(self.anon_app)

    def tearDown(self):
        (
            topics._parser,
            topics._llm_client,
            topics._learning_store,
            topics._mcp_gateway,
            topics._video_recommender,
        ) = self._prev
        self.tempdir.cleanup()

    def _stream_events(self, path: str, payload: dict) -> list[dict]:
        with self.client.stream("POST", path, json=payload) as res:
            self.assertEqual(res.status_code, 200)
            return [json.loads(line) for line in res.iter_lines() if line]

    # ---------------------------------------------------------------- helpers

    def test_llm_error_code_reads_the_attribute_first(self):
        class Coded(RuntimeError):
            error_code = "llm_truncated"

        self.assertEqual(topics._llm_error_code(Coded("boom")), "llm_truncated")

    def test_llm_error_code_falls_back_to_the_message(self):
        # Only the literal code string is matched; a prose "budget exceeded"
        # in the message is not recognised.
        self.assertEqual(
            topics._llm_error_code(RuntimeError(f"call failed: {BUDGET_EXCEEDED_CODE}")),
            BUDGET_EXCEEDED_CODE,
        )
        self.assertEqual(topics._llm_error_code(RuntimeError("budget exceeded")), "")

    def test_llm_error_code_is_empty_for_an_unrelated_failure(self):
        self.assertEqual(topics._llm_error_code(RuntimeError("socket closed")), "")

    def test_the_generation_error_event_distinguishes_budget_from_hard_failures(self):
        self.assertEqual(
            topics._generation_error_event(
                RuntimeError(BUDGET_EXCEEDED_CODE),
                message="Custom topic generation failed.",
            ),
            {
                "type": "error",
                "code": BUDGET_EXCEEDED_CODE,
                "message": "LLM call budget exceeded. Retry shortly.",
            },
        )
        self.assertEqual(
            topics._generation_error_event(
                RuntimeError("socket closed"), message="Custom topic generation failed."
            ),
            {
                "type": "error",
                "code": "generation_failed",
                "message": "Custom topic generation failed.",
            },
        )

    def test_a_budget_failure_maps_to_429_and_anything_else_does_not(self):
        class Coded(RuntimeError):
            error_code = BUDGET_EXCEEDED_CODE

        mapped = topics._budget_http_exception(Coded("boom"))
        self.assertEqual(mapped.status_code, 429)
        self.assertEqual(mapped.detail["code"], BUDGET_EXCEEDED_CODE)
        self.assertIsNone(topics._budget_http_exception(RuntimeError("boom")))

    def test_a_profile_older_than_seven_days_is_stale(self):
        fresh = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        self.assertFalse(topics._profile_is_stale(fresh))
        self.assertTrue(topics._profile_is_stale(old))

    def test_a_naive_profile_timestamp_is_read_as_utc(self):
        naive = (datetime.now(timezone.utc) - timedelta(days=1)).replace(tzinfo=None)
        self.assertFalse(topics._profile_is_stale(naive.isoformat()))

    def test_an_unparsable_profile_timestamp_is_treated_as_stale(self):
        self.assertTrue(topics._profile_is_stale("not-a-timestamp"))

    def test_topic_filters_reject_on_track_level_and_query(self):
        base = {
            "title": "Kafka internals",
            "description": "Partitions and replication",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
        }
        self.assertTrue(topics._passes_topic_filters(**base, track_filter=None, level_filter=None, query_filter=None))
        self.assertTrue(
            topics._passes_topic_filters(**base, track_filter="backend", level_filter="mid", query_filter="kafka")
        )
        self.assertFalse(topics._passes_topic_filters(**base, track_filter="frontend", level_filter=None, query_filter=None))
        self.assertFalse(topics._passes_topic_filters(**base, track_filter=None, level_filter="lead", query_filter=None))
        self.assertFalse(topics._passes_topic_filters(**base, track_filter=None, level_filter=None, query_filter="redis"))
        # A blank query is treated as "no filter" rather than as a literal "".
        self.assertTrue(topics._passes_topic_filters(**base, track_filter=None, level_filter=None, query_filter="   "))

    # --------------------------------------------------------- service gates

    def test_every_topic_route_refuses_to_serve_before_init(self):
        topics._parser = None
        for method, path, payload in (
            ("get", "/api/topics", None),
            ("get", "/api/topics/01-static", None),
            ("get", "/api/topics/01-static/videos", None),
            ("get", "/api/topics/01-static/preferences", None),
            ("put", "/api/topics/01-static/preferences", {"response_detail": "concise"}),
        ):
            with self.subTest(path=path):
                if method == "get":
                    res = self.client.get(path)
                else:
                    res = self.client.put(path, json=payload)
                self.assertEqual(res.status_code, 503, path)
                self.assertEqual(res.json()["detail"], "Service not initialised")

    def test_the_video_routes_refuse_to_serve_without_a_recommender(self):
        topics._video_recommender = None
        res = self.client.get("/api/features/topic-videos/status")
        self.assertEqual(res.status_code, 503)
        self.assertEqual(res.json()["detail"], "Video service not initialised")

    # ------------------------------------------------------------ list/get

    def test_anonymous_listing_omits_the_problem_solving_synthetic_topic(self):
        res = self.anon_client.get("/api/topics")
        self.assertEqual(res.status_code, 200)
        ids = [item["id"] for item in res.json()]
        self.assertIn("01-static", ids)
        self.assertNotIn(PROBLEM_SOLVING_ID, ids)
        self.assertEqual(res.headers["X-Offset"], "0")
        self.assertNotIn("X-Limit", res.headers)

    def test_an_authenticated_listing_adds_the_problem_solving_topic_and_paginates(self):
        res = self.client.get("/api/topics")
        self.assertEqual(res.status_code, 200)
        # The synthetic built-in sorts ahead of the static handbook topic and
        # advertises the default question budget while nothing is cached.
        body = res.json()
        self.assertEqual(
            [item["id"] for item in body], [PROBLEM_SOLVING_ID, "01-static"]
        )
        self.assertEqual(body[0]["estimated_questions"], 240)
        self.assertEqual(body[0]["section_count"], 0)
        self.assertNotIn("X-Limit", res.headers)

        res = self.client.get("/api/topics", params={"limit": 1, "offset": 1})
        self.assertEqual(res.headers["X-Limit"], "1")
        self.assertEqual(res.headers["X-Offset"], "1")
        self.assertEqual(int(res.headers["X-Total-Count"]), 2)
        self.assertEqual([item["id"] for item in res.json()], ["01-static"])

    def test_a_cached_problem_solving_curriculum_reports_its_real_size(self):
        _seed_dynamic_curriculum(self.store, PROBLEM_SOLVING_ID, "python")
        res = self.client.get("/api/topics")
        summary = next(
            item for item in res.json() if item["id"] == PROBLEM_SOLVING_ID
        )
        self.assertEqual(summary["section_count"], 1)
        self.assertEqual(summary["estimated_questions"], 5)

    def test_a_query_that_excludes_the_problem_solving_topic_skips_its_lookup(self):
        res = self.client.get("/api/topics", params={"q": "static"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual([item["id"] for item in res.json()], ["01-static"])
        self.assertNotIn("resolve_topic_ai_settings", self.store.call_names())

    def test_a_track_filter_that_excludes_the_problem_solving_topic_skips_its_lookup(self):
        res = self.client.get("/api/topics", params={"track": "frontend"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), [])
        self.assertNotIn("resolve_topic_ai_settings", self.store.call_names())

    def test_a_level_filter_keeps_the_problem_solving_topic(self):
        res = self.client.get("/api/topics", params={"level": "mid"})
        self.assertEqual(res.status_code, 200)
        self.assertIn(PROBLEM_SOLVING_ID, [item["id"] for item in res.json()])

    def test_an_out_of_range_level_filter_is_rejected_by_the_query_pattern(self):
        res = self.client.get("/api/topics", params={"level": "lead"})
        self.assertEqual(res.status_code, 422)

    def test_the_problem_solving_topic_is_404_for_an_anonymous_reader(self):
        res = self.anon_client.get(f"/api/topics/{PROBLEM_SOLVING_ID}")
        self.assertEqual(res.status_code, 404)
        self.assertIn("not found", res.json()["detail"])

    def test_an_unknown_topic_is_404_for_an_anonymous_reader(self):
        res = self.anon_client.get("/api/topics/99-does-not-exist")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.json()["detail"], "Topic '99-does-not-exist' not found")

    def test_an_unknown_topic_is_404_for_an_authenticated_reader_too(self):
        res = self.client.get("/api/topics/99-does-not-exist")
        self.assertEqual(res.status_code, 404)

    def test_a_static_topic_is_served_without_a_language_profile_for_anonymous_readers(self):
        res = self.anon_client.get("/api/topics/01-static")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["title"], "Static Topic")
        self.assertTrue(body["content_ready"])
        self.assertEqual(body["language_options"], [])
        self.assertNotIn("get_topic_language_profile", self.store.call_names())

    def test_a_static_topic_gets_a_language_profile_for_authenticated_readers(self):
        topics._mcp_gateway = StubMCP()
        res = self.client.get("/api/topics/01-static")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["language_options"], ["python", "go", "typescript"])
        self.assertTrue(res.json()["requires_programming"])
        # A cold profile runs the advisor, which gathers external context first.
        self.assertEqual(topics._mcp_gateway.calls[0]["flow"], "questions")
        self.assertEqual(topics._mcp_gateway.calls[0]["topic_id"], "01-static")
        self.assertIn("Python and Go", self.llm.prompts[-1])
        # The upserted profile is cached, so a second read skips the advisor.
        before = self.llm.calls
        self.client.get("/api/topics/01-static")
        self.assertEqual(self.llm.calls, before)

    def test_the_problem_solving_topic_uses_the_builtin_profile(self):
        res = self.client.get(f"/api/topics/{PROBLEM_SOLVING_ID}")
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertTrue(body["requires_programming"])
        self.assertIn("python", body["language_options"])
        # The builtin profile needs no advisor call at all.
        self.assertEqual(self.llm.calls, 0)
        self.assertIn("upsert_topic_language_profile", self.store.call_names())

    def test_a_content_ready_flag_is_cleared_when_the_topic_has_no_sections(self):
        empty = self.parser.get_topic("01-static")
        topics._parser = _StubParser(empty.model_copy(update={"sections": []}))
        res = self.client.get("/api/topics/01-static")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["content_ready"])

    # ------------------------------------------------------------- videos

    def test_the_video_metrics_route_merges_status_and_usage(self):
        self.videos._status = {
            "enabled": False,
            "status": "disabled_quota_exhausted",
            "disabled_until": "2030-01-01T00:00:00Z",
            "reason": "quota",
        }
        res = self.client.get("/api/features/topic-videos/metrics")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "disabled_quota_exhausted")
        self.assertTrue(body["hidden_now"])
        self.assertEqual(body["disabled_until"], "2030-01-01T00:00:00Z")
        self.assertEqual(body["reason"], "quota")

    def test_the_video_metrics_route_defaults_a_missing_status(self):
        self.videos._status = {}
        res = self.client.get("/api/features/topic-videos/metrics")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "disabled_config")
        self.assertFalse(body["hidden_now"])

    def test_a_video_event_is_recorded_with_its_metadata(self):
        res = self.client.post(
            "/api/features/topic-videos/events",
            json={
                "event_name": "video_click",
                "topic_id": "01-static",
                "section_index": 2,
                "section_heading": "Static Section",
                "video_id": "vid-1",
                "metadata": {"position": 0},
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"ok": True})
        self.assertIn("record_feature_event", self.store.call_names())

    def test_an_unknown_video_event_name_is_rejected(self):
        res = self.client.post(
            "/api/features/topic-videos/events", json={"event_name": "teleport"}
        )
        self.assertEqual(res.status_code, 422)

    def test_section_videos_are_404_for_an_unknown_topic(self):
        res = self.client.get("/api/topics/99-nope/videos")
        self.assertEqual(res.status_code, 404)

    def test_section_videos_are_422_when_the_index_is_out_of_range(self):
        res = self.client.get("/api/topics/01-static/videos", params={"section_index": 99})
        self.assertEqual(res.status_code, 422)
        self.assertEqual(res.json()["detail"], "section_index is out of range for this topic")

    def test_section_videos_pass_the_topic_language_and_record_a_view(self):
        res = self.client.get("/api/topics/01-static/videos")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["topic_id"], "01-static")
        self.assertEqual(body["section_heading"], "Static Section")
        self.assertEqual(self.videos.recommend_calls[0]["preferred_language"], "")
        self.assertIn("record_feature_event", self.store.call_names())

    def test_an_explicit_preferred_language_overrides_the_topic_default(self):
        self.client.get(
            "/api/topics/01-static/videos", params={"preferred_language": "  GoLang  "}
        )
        self.assertEqual(self.videos.recommend_calls[0]["preferred_language"], "golang")

    def test_a_disabled_recommendation_records_no_view_event(self):
        self.videos._result = {
            "enabled": False,
            "status": "disabled_config",
            "reason": "off",
        }
        res = self.client.get("/api/topics/01-static/videos")
        self.assertEqual(res.status_code, 200)
        self.assertFalse(res.json()["enabled"])
        self.assertNotIn("record_feature_event", self.store.call_names())

    # --------------------------------------------------------- preferences

    def test_preferences_are_404_for_an_unknown_topic(self):
        self.assertEqual(self.client.get("/api/topics/99-nope/preferences").status_code, 404)

    def test_updating_preferences_is_404_for_an_unknown_topic(self):
        res = self.client.put("/api/topics/99-nope/preferences", json={"response_detail": "concise"})
        self.assertEqual(res.status_code, 404)

    def test_preferences_round_trip(self):
        res = self.client.get("/api/topics/01-static/preferences")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["requires_programming"], True)

        res = self.client.put(
            "/api/topics/01-static/preferences",
            json={"response_detail": "concise", "preferred_language": "go"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["preferred_language"], "go")
        self.assertEqual(res.json()["response_detail"], "concise")

        res = self.client.get("/api/topics/01-static/preferences")
        self.assertEqual(res.json()["preferred_language"], "go")
        self.assertEqual(res.json()["response_detail"], "concise")

    def test_a_language_outside_the_profile_is_rejected(self):
        res = self.client.put(
            "/api/topics/01-static/preferences", json={"preferred_language": "brainfuck"}
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("preferred_language must be one of", res.json()["detail"])

    def test_a_blank_language_clears_the_preference_without_rejection(self):
        res = self.client.put("/api/topics/01-static/preferences", json={"preferred_language": "   "})
        self.assertEqual(res.status_code, 200)

    def test_omitting_the_language_skips_validation_entirely(self):
        res = self.client.put(
            "/api/topics/01-static/preferences", json={"response_detail": "very_detailed"}
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["response_detail"], "very_detailed")
        upserts = [kwargs for name, kwargs in self.store.calls if name == "upsert_topic_preferences"]
        self.assertIsNone(upserts[-1]["preferred_language"])

    # ------------------------------------------- problem solving content stream

    def test_the_content_stream_is_404_for_an_ordinary_topic(self):
        res = self.client.post(
            "/api/topics/01-static/content/generate/stream",
            json={"preferred_language": "python"},
        )
        self.assertEqual(res.status_code, 404)

    def test_the_content_stream_is_422_for_an_unsupported_language(self):
        res = self.client.post(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            json={"preferred_language": "cobol"},
        )
        self.assertEqual(res.status_code, 422)
        self.assertIn("preferred_language must be one of", res.json()["detail"])

    def test_the_content_stream_is_422_when_the_language_is_too_short(self):
        res = self.client.post(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            json={"preferred_language": "p"},
        )
        self.assertEqual(res.status_code, 422)

    def test_the_content_stream_reuses_a_cached_curriculum(self):
        _seed_dynamic_curriculum(self.store, PROBLEM_SOLVING_ID, "python")
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[0]["preferred_language"], "python")
        progress = [e for e in events if e["type"] == "progress"]
        self.assertEqual(progress[-1]["stage"], "ready")
        self.assertEqual(progress[-1]["message"], "Using cached roadmap for selected language.")
        done = events[-1]
        self.assertEqual(done["type"], "done")
        self.assertTrue(done["topic"]["is_dynamic_topic"])
        # The cached path must not touch the generator or the LLM at all.
        self.assertEqual(self.llm.calls, 0)

    def test_force_regenerate_deletes_the_cached_curriculum_first(self):
        _seed_dynamic_curriculum(self.store, PROBLEM_SOLVING_ID, "python")
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python", "force_regenerate": True},
        )
        names = self.store.call_names()
        self.assertIn("delete_dynamic_topic_curriculum", names)
        self.assertLess(
            names.index("delete_dynamic_topic_curriculum"),
            names.index("upsert_dynamic_topic_curriculum"),
        )
        self.assertEqual(events[-1]["type"], "done")

    def test_an_unknown_frame_type_is_ignored_by_the_content_stream(self):
        _patch_problem_solving_stream(
            self,
            [
                {"type": "ping", "index": 1},
                {},
                {
                    "type": "done",
                    "topic": {
                        "id": PROBLEM_SOLVING_ID,
                        "title": "T",
                        "description": "d",
                        "track": "system_design",
                        "levels": ["mid"],
                        "sections": [{"heading": "h", "content": "c"}],
                        "raw_content": "raw",
                    },
                },
            ],
        )
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual([e["type"] for e in events], ["start", "done"])

    def test_a_dict_done_payload_is_rehydrated_into_a_topic_detail(self):
        _patch_problem_solving_stream(self, [{"type": "done", "topic": {"id": PROBLEM_SOLVING_ID, "title": "Dict Topic", "description": "d", "track": "system_design", "levels": ["mid"], "sections": [{"heading": "h", "content": "c"}], "raw_content": "raw"}}])
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[-1]["topic"]["title"], "Dict Topic")
        upserts = [kwargs for name, kwargs in self.store.calls if name == "upsert_dynamic_topic_curriculum"]
        self.assertEqual(upserts[-1]["title"], "Dict Topic")

    def test_an_unusable_done_payload_is_reported_as_a_stream_error(self):
        _patch_problem_solving_stream(self, [{"type": "done", "topic": "not-a-topic"}])
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "Dynamic topic generation failed.")

    def test_progress_and_section_frames_are_forwarded_with_defaults(self):
        _patch_problem_solving_stream(
            self,
            [
                {"type": "progress"},
                {"type": "section"},
                {
                    "type": "done",
                    "topic": {
                        "id": PROBLEM_SOLVING_ID,
                        "title": "T",
                        "description": "d",
                        "track": "system_design",
                        "levels": ["mid"],
                        "sections": [{"heading": "h", "content": "c"}],
                        "raw_content": "raw",
                    },
                },
            ],
        )
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        progress = [e for e in events if e["type"] == "progress" and e["stage"] == "batching"][0]
        self.assertEqual(
            progress["message"],
            "Generating language-specific problem solving roadmap...",
        )
        self.assertEqual(progress["batch_index"], 0)
        self.assertEqual(progress["batch_count"], 0)
        self.assertGreater(progress["target_sections"], 0)
        section = [e for e in events if e["type"] == "section"][0]
        self.assertEqual(section["index"], 0)
        self.assertEqual(section["heading"], "")
        self.assertGreater(section["total_sections"], 0)

    def test_a_policy_block_is_reported_with_its_own_code(self):
        async def _blocked(self, **kwargs):
            raise LLMServiceApprovalRequiredError("Approval required")
            yield  # pragma: no cover - generator marker

        _patch_problem_solving_stream(self, None, _blocked)
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], APPROVAL_REQUIRED_CODE)

    def test_a_budget_rejection_in_the_content_stream_keeps_its_own_code(self):
        class BudgetExceeded(RuntimeError):
            error_code = BUDGET_EXCEEDED_CODE

        async def _exhausted(self, **kwargs):
            raise BudgetExceeded("budget exceeded")
            yield  # pragma: no cover - generator marker

        _patch_problem_solving_stream(self, None, _exhausted)
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[-1]["code"], BUDGET_EXCEEDED_CODE)
        self.assertEqual(events[-1]["message"], "LLM call budget exceeded. Retry shortly.")

    # ------------------------------------------------------ custom topic stream

    def test_an_unusable_custom_done_payload_is_reported_as_a_stream_error(self):
        import app.services.custom_topic_generator as generator_module

        async def _bad_done(self, **kwargs):
            yield {"type": "done", "topic": ["not", "a", "topic"]}

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _bad_done
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "generation_failed")
        self.assertEqual(events[-1]["message"], "Custom topic generation failed.")

    def test_a_dict_custom_done_payload_is_rehydrated_and_persisted(self):
        import app.services.custom_topic_generator as generator_module

        payload = {
            "id": "custom-rust",
            "title": "Rust Ownership Interview Roadmap",
            "description": "d",
            "track": "backend",
            "levels": ["junior"],
            "sections": [{"heading": "Borrowing", "content": "Ownership rules."}],
            "raw_content": "raw",
        }

        async def _dict_done(self, **kwargs):
            yield {"type": "done", "topic": payload}

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _dict_done
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual(events[-1]["topic"]["title"], "Rust Ownership Interview Roadmap")
        upserts = [kwargs for name, kwargs in self.store.calls if name == "upsert_custom_topic"]
        self.assertEqual(upserts[-1]["topic_id"], "custom-rust")

    def test_an_unknown_frame_type_is_ignored_by_the_custom_stream(self):
        import app.services.custom_topic_generator as generator_module

        async def _scripted(self, **kwargs):
            yield {"type": "ping"}
            yield {
                "type": "done",
                "topic": {
                    "id": "custom-rust",
                    "title": "T",
                    "description": "d",
                    "track": "backend",
                    "levels": ["junior"],
                    "sections": [{"heading": "h", "content": "c"}],
                    "raw_content": "raw",
                },
            }

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _scripted
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual([e["type"] for e in events], ["start", "done"])

    def test_a_policy_block_in_the_custom_stream_keeps_its_own_code(self):
        import app.services.custom_topic_generator as generator_module

        async def _blocked(self, **kwargs):
            raise LLMServiceApprovalRequiredError("Approval required")
            yield  # pragma: no cover - generator marker

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _blocked
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual(events[-1]["code"], APPROVAL_REQUIRED_CODE)
        self.assertIn("approval", events[-1]["message"].lower())

    def test_the_video_status_route_passes_the_gate_through_verbatim(self):
        self.videos._status = {
            "enabled": False,
            "status": "disabled_config",
            "disabled_until": "",
            "reason": "flag off",
        }
        res = self.client.get("/api/features/topic-videos/status")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.json(),
            {
                "enabled": False,
                "status": "disabled_config",
                "disabled_until": "",
                "reason": "flag off",
            },
        )

    def test_a_user_scoped_custom_topic_is_resolved_and_flagged_dynamic(self):
        self.store.upsert_custom_topic(
            user_id="alice",
            topic_id="custom-rust",
            source_topic="Rust ownership",
            title="Rust Ownership Roadmap",
            description="Ownership and borrowing.",
            track="backend",
            levels=["junior", "mid", "senior"],
            sections=[{"heading": "Borrowing", "content": "Ownership rules in detail."}],
            raw_content="raw",
        )
        res = self.client.get("/api/topics/custom-rust")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["title"], "Rust Ownership Roadmap")
        self.assertEqual(res.json()["sections"][0]["heading"], "Borrowing")
        self.assertIn("get_custom_topic", self.store.call_names())

    def test_a_custom_topic_owned_by_someone_else_is_404(self):
        self.store.upsert_custom_topic(
            user_id="bob",
            topic_id="custom-bobs",
            source_topic="Bob topic",
            title="Bob",
            description="Bob's topic.",
            track="backend",
            levels=["junior"],
            sections=[{"heading": "h", "content": "c"}],
            raw_content="raw",
        )
        self.assertEqual(self.client.get("/api/topics/custom-bobs").status_code, 404)

    def test_a_topic_detail_built_in_instance_in_the_done_frame_is_used_directly(self):
        from app.schemas.models import TopicDetail

        detail = TopicDetail(
            id=PROBLEM_SOLVING_ID,
            title="Instance Topic",
            description="d",
            track="system_design",
            levels=["mid"],
            sections=[{"heading": "h", "content": "c"}],
            raw_content="raw",
        )
        _patch_problem_solving_stream(
            self,
            [
                {"type": "done", "topic": detail},
                # A frame after the done keeps the stream loop iterating.
                {"type": "section", "index": 99, "total_sections": 1},
            ],
        )
        events = self._stream_events(
            f"/api/topics/{PROBLEM_SOLVING_ID}/content/generate/stream",
            {"preferred_language": "python"},
        )
        self.assertEqual(events[-1]["topic"]["title"], "Instance Topic")
        upserts = [
            kwargs
            for name, kwargs in self.store.calls
            if name == "upsert_dynamic_topic_curriculum"
        ]
        self.assertEqual(upserts[-1]["title"], "Instance Topic")

    def test_creating_a_custom_topic_persists_and_returns_the_generated_detail(self):
        res = self.client.post("/api/topics/custom", json={"topic": "  Rust ownership  "})
        self.assertEqual(res.status_code, 200, res.text)
        body = res.json()
        self.assertEqual(body["id"], "custom-rust-ownership")
        self.assertTrue(body["sections"])
        upserts = [kwargs for name, kwargs in self.store.calls if name == "upsert_custom_topic"]
        self.assertEqual(upserts[-1]["source_topic"], "Rust ownership")

    def test_the_custom_stream_forwards_progress_and_section_frames(self):
        import app.services.custom_topic_generator as generator_module

        async def _scripted(self, **kwargs):
            yield {
                "type": "progress",
                "message": "Building sections 1-8 of 50.",
                "batch_index": 1,
                "batch_count": 7,
                "generated_sections": 0,
                "target_sections": 50,
            }
            yield {"type": "progress"}
            yield {
                "type": "section",
                "index": 1,
                "total_sections": 50,
                "heading": "Ownership",
                "content": "Borrowing rules.",
            }
            yield {
                "type": "done",
                "topic": {
                    "id": "custom-rust",
                    "title": "Rust Ownership Roadmap",
                    "description": "d",
                    "track": "backend",
                    "levels": ["junior"],
                    "sections": [{"heading": "Ownership", "content": "Borrowing rules."}],
                    "raw_content": "raw",
                },
            }

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _scripted
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original

        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[0]["topic"], "Rust ownership")
        self.assertEqual(events[0]["target_sections"], 0)
        progress = [e for e in events if e["type"] == "progress"]
        self.assertEqual(progress[0]["batch_index"], 1)
        self.assertEqual(progress[0]["batch_count"], 7)
        self.assertEqual(
            progress[1]["message"],
            "Analyzing custom topic and building roadmap...",
        )
        self.assertEqual(progress[1]["batch_count"], 0)
        section = [e for e in events if e["type"] == "section"][0]
        self.assertEqual(section["heading"], "Ownership")
        self.assertEqual(section["index"], 1)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["topic"]["title"], "Rust Ownership Roadmap")

    def test_an_explicit_custom_target_is_echoed_on_the_start_frame(self):
        import app.services.custom_topic_generator as generator_module

        async def _scripted(self, **kwargs):
            yield {"type": "section", "index": 1, "total_sections": 60}
            yield {
                "type": "done",
                "topic": {
                    "id": "custom-rust",
                    "title": "T",
                    "description": "d",
                    "track": "backend",
                    "levels": ["junior"],
                    "sections": [{"heading": "h", "content": "c"}],
                    "raw_content": "raw",
                },
            }

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _scripted
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership", "target_sections": 60}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual(events[0]["target_sections"], 60)
        section = [e for e in events if e["type"] == "section"][0]
        self.assertEqual(section["content"], "")
        self.assertEqual(section["total_sections"], 60)

    def test_a_non_budget_creation_failure_is_not_rewritten_as_a_429(self):
        import app.services.custom_topic_generator as generator_module

        async def _socket_dead(self, **kwargs):
            raise ConnectionError("socket closed")

        original = generator_module.CustomTopicGenerator.generate_topic
        generator_module.CustomTopicGenerator.generate_topic = _socket_dead
        try:
            with self.assertRaises(ConnectionError):
                self.client.post("/api/topics/custom", json={"topic": "Rust ownership"})
        finally:
            generator_module.CustomTopicGenerator.generate_topic = original

    def test_a_custom_topic_instance_in_the_done_frame_is_used_directly(self):
        import app.services.custom_topic_generator as generator_module
        from app.schemas.models import TopicDetail

        detail = TopicDetail(
            id="custom-rust",
            title="Instance Roadmap",
            description="d",
            track="backend",
            levels=["junior"],
            sections=[{"heading": "Ownership", "content": "Borrowing rules."}],
            raw_content="raw",
        )

        async def _instance_done(self, **kwargs):
            yield {"type": "done", "topic": detail}
            yield {"type": "section", "index": 99, "total_sections": 1}

        original = generator_module.CustomTopicGenerator.generate_topic_stream
        generator_module.CustomTopicGenerator.generate_topic_stream = _instance_done
        try:
            events = self._stream_events(
                "/api/topics/custom/stream", {"topic": "Rust ownership"}
            )
        finally:
            generator_module.CustomTopicGenerator.generate_topic_stream = original
        self.assertEqual(events[-1]["topic"]["title"], "Instance Roadmap")
        upserts = [kwargs for name, kwargs in self.store.calls if name == "upsert_custom_topic"]
        self.assertEqual(upserts[-1]["topic_id"], "custom-rust")

    def test_a_too_short_custom_topic_is_rejected_by_the_schema(self):
        res = self.client.post("/api/topics/custom", json={"topic": "r"})
        self.assertEqual(res.status_code, 422)

    def test_an_out_of_range_custom_target_is_rejected_by_the_schema(self):
        res = self.client.post("/api/topics/custom", json={"topic": "Rust", "target_sections": 5})
        self.assertEqual(res.status_code, 422)


class _StubParser:
    def __init__(self, topic):
        self._topic = topic

    def get_topic(self, topic_id):
        return self._topic if topic_id == self._topic.id else None

    def list_topics(self, track=None, level=None, q=None):
        return [self._topic]


def _seed_dynamic_curriculum(store, topic_id: str, language: str) -> None:
    store.upsert_dynamic_topic_curriculum(
        user_id="alice",
        topic_id=topic_id,
        preferred_language=language,
        title="Cached Roadmap",
        description="Cached description.",
        track="system_design",
        levels=["mid", "senior"],
        sections=[{"heading": "Two Sum", "content": "Hash map walkthrough."}],
        raw_content="raw",
        target_sections=50,
        source="llm",
    )


def _patch_problem_solving_stream(case, events, impl=None):
    """Replace ``generate_topic_stream`` with a scripted async generator."""
    from app.services.problem_solving_generator import ProblemSolvingGenerator

    if impl is None:

        async def _scripted(self, **kwargs):
            for event in events:
                yield event

        impl = _scripted
    case._prev_stream = getattr(ProblemSolvingGenerator, "generate_topic_stream", None)
    ProblemSolvingGenerator.generate_topic_stream = impl

    def _restore():
        if case._prev_stream is not None:
            ProblemSolvingGenerator.generate_topic_stream = case._prev_stream

    case.addCleanup(_restore)