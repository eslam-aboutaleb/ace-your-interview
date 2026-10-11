import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import placement as placement_router
from app.services.doc_parser import DocParser
from app.services.learning_planner import LearningPlannerStore
from app.services.learning_store import LearningStore
from app.services.placement_service import (
    LEVELS,
    CatalogQuestionSource,
    PlacementService,
    PlacementUnavailableError,
    derive_placement,
    grade_answer,
    public_question,
    walk_is_complete,
)


class _FakeSource:
    """Deterministic question bank used to drive the walk in tests."""

    source_name = "fake"

    def __init__(self, per_level: dict[str, list[dict[str, Any]]]):
        self._per_level = per_level

    def questions(
        self,
        *,
        track: str,
        level: str,
        limit: int,
        exclude: set[str] | None = None,
        user_identity: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        excluded = set(exclude or ())
        return [
            question
            for question in self._per_level.get(level, [])
            if question["question_id"] not in excluded
        ][: max(1, int(limit))]


def _fake_question(level: str, index: int) -> dict[str, Any]:
    return {
        "question_id": f"fake-{level}-{index}",
        "question": f"{level} question {index}",
        "options": ["alpha", "beta", "gamma", "delta"],
        "correct_answer": "B",
        "explanation": "beta is correct",
        "topic_id": f"01-topic-{index}",
        "topic_title": f"Topic {index}",
        "concept": f"Concept {index}",
        "level": level,
        "source": "fake",
    }


def _fake_source() -> _FakeSource:
    return _FakeSource(
        {level: [_fake_question(level, i) for i in range(4)] for level in LEVELS}
    )


def _current_question(state: dict[str, Any]) -> dict[str, Any]:
    """Full question (answer key included) for the next step of a session."""
    level = state["level"]
    asked = set(state["asked"])
    for question in state["bank"].get(level, []):
        if question["question_id"] not in asked:
            return question
    raise AssertionError(f"no un-asked question at level {level}")


class _WalkDriver:
    """Runs a placement session with a scripted outcome per step."""

    def __init__(self, service: PlacementService, *, user_id: str, track: str = "backend"):
        self.service = service
        self.user_id = user_id
        self.track = track

    async def run(self, outcomes: list[bool]) -> dict[str, Any]:
        started = await self.service.start_session(
            user_id=self.user_id, track=self.track
        )
        session_id = started["session_id"]
        result = {
            "session_id": session_id,
            "steps": 0,
            "level_path": [],
            "final_level": started["level"],
            "complete": False,
        }
        for outcome in outcomes:
            state = self.service.get_session(
                user_id=self.user_id, session_id=session_id
            )
            if state["status"] == "complete":
                break
            question = _current_question(state)
            answer = await self.service.answer(
                user_id=self.user_id,
                session_id=session_id,
                question_id=question["question_id"],
                selected_answer=(
                    question["correct_answer"] if outcome else self._wrong_label(question)
                ),
            )
            result["steps"] = answer["step_index"]
            result["level_path"].append(question["level"])
            result["final_level"] = answer["level"]
            result["complete"] = answer["complete"]
        return result

    @staticmethod
    def _wrong_label(question: dict[str, Any]) -> str:
        correct = question["correct_answer"]
        for label in ("A", "B", "C", "D"):
            if label != correct and label < chr(ord("A") + len(question["options"])):
                return label
        raise AssertionError("no wrong label available")


class _FakeGamification:
    def __init__(self):
        self.goals: list[int] = []

    async def set_daily_goal(self, *, user_id: str, daily_goal: int) -> dict[str, Any]:
        self.goals.append(daily_goal)
        return {"daily_goal": daily_goal}


def _build_service(planner, *, sources=None, learning_store=None, gamification=None):
    parser = DocParser(curriculum_path=str(get_settings().curriculum_path))
    return PlacementService(
        planner=planner,
        parser=parser,
        sources=sources if sources is not None else [_fake_source()],
        learning_store=learning_store,
        gamification_service=gamification,
    )


class PlacementAlgorithmTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "learning.db")
        self.planner = LearningPlannerStore(self.db_path)

    async def _service(self, **kwargs) -> PlacementService:
        return _build_service(self.planner, **kwargs)

    def _parser_for_tests(self) -> DocParser:
        return DocParser(curriculum_path=str(get_settings().curriculum_path))

    async def test_correct_answer_steps_harder_and_wrong_steps_easier(self):
        service = await self._service()
        started = await service.start_session(
            user_id="u1", track="backend"
        )
        self.assertEqual(started["level"], "mid")
        state = service.get_session(user_id="u1", session_id=started["session_id"])
        mid_question = _current_question(state)

        harder = await service.answer(
            user_id="u1",
            session_id=started["session_id"],
            question_id=mid_question["question_id"],
            selected_answer=mid_question["correct_answer"],
        )
        self.assertTrue(harder["is_correct"])
        self.assertEqual(harder["level"], "senior")

        state = service.get_session(user_id="u1", session_id=started["session_id"])
        senior_question = _current_question(state)
        easier = await service.answer(
            user_id="u1",
            session_id=started["session_id"],
            question_id=senior_question["question_id"],
            selected_answer=self._wrong_label(senior_question),
        )
        self.assertFalse(easier["is_correct"])
        self.assertEqual(easier["level"], "mid")

        state = service.get_session(user_id="u1", session_id=started["session_id"])
        mid_question = _current_question(state)
        easier_again = await service.answer(
            user_id="u1",
            session_id=started["session_id"],
            question_id=mid_question["question_id"],
            selected_answer=self._wrong_label(mid_question),
        )
        self.assertFalse(easier_again["is_correct"])
        self.assertEqual(easier_again["level"], "junior")

    @staticmethod
    def _wrong_label(question: dict[str, Any]) -> str:
        for label in ("A", "B", "C", "D"):
            if label != question["correct_answer"]:
                return label
        raise AssertionError("no wrong label")

    async def test_walk_converges_after_direction_reversal(self):
        service = await self._service()
        result = await _WalkDriver(service, user_id="u2").run(
            [True, False, True, False, True, False, True, False, True]
        )
        self.assertEqual(result["steps"], 8)
        self.assertTrue(result["complete"])

    async def test_all_correct_walk_stalls_at_the_top_level(self):
        service = await self._service()
        result = await _WalkDriver(service, user_id="u3").run(
            [True, True, True, True, True]
        )
        self.assertEqual(result["steps"], 3)
        self.assertTrue(result["complete"])

    async def test_all_wrong_walk_stalls_at_the_bottom_level(self):
        service = await self._service()
        result = await _WalkDriver(service, user_id="u4").run(
            [False, False, False, False]
        )
        self.assertEqual(result["steps"], 3)
        self.assertTrue(result["complete"])

    def test_walk_completes_at_the_step_cap(self):
        answers = [
            {
                "level": LEVELS[index % len(LEVELS)],
                "is_correct": index % 2 == 0,
                "topic_id": "01-x",
                "topic_title": "X",
                "concept": "C",
            }
            for index in range(12)
        ]
        self.assertTrue(walk_is_complete(answers))

    def test_grading_is_deterministic_and_label_based(self):
        question = {
            "correct_answer": "C",
            "options": ["one", "two", "three", "four"],
        }
        for _ in range(3):
            self.assertTrue(grade_answer(question, "C"))
            self.assertFalse(grade_answer(question, "A"))
        self.assertTrue(grade_answer(question, "2"))
        self.assertTrue(grade_answer(question, "three"))
        self.assertFalse(grade_answer(question, "four"))
        self.assertFalse(grade_answer(question, ""))
        self.assertFalse(grade_answer({"correct_answer": "B", "options": []}, "B"))

    def test_public_question_never_leaks_the_answer_key(self):
        question = _fake_question("mid", 1)
        public = public_question(question)
        self.assertNotIn("correct_answer", public)
        self.assertNotIn("explanation", public)
        self.assertEqual(public["question_id"], question["question_id"])
        self.assertEqual(public["options"], question["options"])

    def test_derive_placement_maps_the_walked_path_to_levels(self):
        answers = [
            {"level": "mid", "is_correct": True, "concept": "a", "topic_id": "01-a"},
            {"level": "senior", "is_correct": True, "concept": "b", "topic_id": "01-b"},
            {"level": "senior", "is_correct": True, "concept": "c", "topic_id": "01-c"},
        ]
        placement = derive_placement(answers, track="backend")
        self.assertEqual(placement["derived_level"], "senior")
        self.assertEqual(placement["per_track_levels"], {"backend": "senior"})
        self.assertEqual(placement["steps_taken"], 3)

    def test_derive_placement_drops_a_level_that_was_never_right(self):
        answers = [
            {"level": "mid", "is_correct": False, "concept": "a", "topic_id": "01-a"},
            {"level": "junior", "is_correct": True, "concept": "b", "topic_id": "01-b"},
            {"level": "mid", "is_correct": False, "concept": "c", "topic_id": "01-c"},
            {"level": "junior", "is_correct": True, "concept": "d", "topic_id": "01-d"},
        ]
        placement = derive_placement(answers, track="frontend")
        self.assertEqual(placement["derived_level"], "junior")
        self.assertEqual(placement["per_track_levels"], {"frontend": "junior"})

    def test_derive_placement_ignores_a_single_lucky_guess(self):
        answers = [
            {"level": "mid", "is_correct": True, "concept": "a", "topic_id": "01-a"},
            {"level": "mid", "is_correct": True, "concept": "b", "topic_id": "01-b"},
            {"level": "senior", "is_correct": True, "concept": "c", "topic_id": "01-c"},
        ]
        placement = derive_placement(answers, track="backend")
        self.assertEqual(placement["derived_level"], "mid")
        self.assertEqual(placement["level_stats"]["senior"], {"correct": 1, "answered": 1})

    def test_derive_placement_reports_weak_objectives_at_or_above_the_level(self):
        answers = [
            {"level": "senior", "is_correct": False, "concept": "sharding", "topic_id": "11-x", "topic_title": "Scaling"},
            {"level": "junior", "is_correct": False, "concept": "basics", "topic_id": "01-y", "topic_title": "Fundamentals"},
            {"level": "mid", "is_correct": True, "concept": "queues", "topic_id": "01-z", "topic_title": "Fundamentals"},
            {"level": "mid", "is_correct": True, "concept": "http", "topic_id": "01-w", "topic_title": "Fundamentals"},
        ]
        placement = derive_placement(answers, track="backend")
        self.assertEqual(placement["derived_level"], "mid")
        self.assertEqual(
            [item["objective"] for item in placement["weak_objectives"]], ["sharding"]
        )

    def test_derive_placement_without_answers_falls_back_to_junior(self):
        placement = derive_placement([], track="ai_stack")
        self.assertEqual(placement["derived_level"], "junior")
        self.assertEqual(placement["per_track_levels"], {"ai_stack": "junior"})

    async def test_profile_seeding_writes_the_derived_levels(self):
        gamification = _FakeGamification()
        service = await self._service(gamification=gamification)
        driver = _WalkDriver(service, user_id="seeded")
        result = await driver.run([True, True, True])
        self.assertTrue(result["complete"])

        summary = await service.finish(
            user_id="seeded", session_id=result["session_id"]
        )
        self.assertEqual(summary["derived_level"], "senior")
        self.assertEqual(summary["per_track_levels"], {"backend": "senior"})
        profile = summary["profile"]
        self.assertEqual(profile["primary_track"], "backend")
        self.assertEqual(profile["current_level"], "senior")
        self.assertEqual(profile["target_level"], "senior")
        self.assertEqual(profile["confidence_by_track"]["backend"], 4)

        stored = self.planner.get_profile(user_id="seeded")
        self.assertEqual(stored["current_level"], "senior")
        self.assertEqual(stored["primary_track"], "backend")
        self.assertEqual(stored["confidence_by_track"]["backend"], 4)
        self.assertEqual(gamification.goals, [3])

    async def test_finish_seeds_target_role_and_daily_goal_budget(self):
        service = await self._service(gamification=_FakeGamification())
        started = await service.start_session(
            user_id="role-user",
            track="frontend",
            target_role="Senior Frontend Engineer",
            daily_goal=10,
        )
        session_id = started["session_id"]
        # mid right, senior wrong, mid right → the walk sits on mid.
        for outcome in (True, False, True):
            state = service.get_session(user_id="role-user", session_id=session_id)
            question = _current_question(state)
            await service.answer(
                user_id="role-user",
                session_id=session_id,
                question_id=question["question_id"],
                selected_answer=(
                    question["correct_answer"]
                    if outcome
                    else _first_wrong_label(question["correct_answer"], 4)
                ),
            )
        summary = await service.finish(user_id="role-user", session_id=session_id)
        self.assertEqual(summary["profile"]["target_role"], "Senior Frontend Engineer")
        self.assertEqual(summary["profile"]["preferred_session_minutes"], 60)
        self.assertEqual(summary["profile"]["weekly_minutes"], 750)
        self.assertEqual(summary["profile"]["current_level"], "mid")
        self.assertEqual(summary["profile"]["primary_track"], "frontend")

    async def test_answer_rejects_a_replayed_question(self):
        service = await self._service()
        started = await service.start_session(user_id="replay", track="backend")
        session_id = started["session_id"]
        state = service.get_session(user_id="replay", session_id=session_id)
        question = _current_question(state)
        await service.answer(
            user_id="replay",
            session_id=session_id,
            question_id=question["question_id"],
            selected_answer=question["correct_answer"],
        )
        with self.assertRaises(ValueError):
            await service.answer(
                user_id="replay",
                session_id=session_id,
                question_id=question["question_id"],
                selected_answer=question["correct_answer"],
            )

    async def test_finish_is_idempotent_for_a_completed_session(self):
        service = await self._service()
        result = await _WalkDriver(service, user_id="idem").run(
            [True, False, True, False, True, False, True, False]
        )
        first = await service.finish(user_id="idem", session_id=result["session_id"])
        second = await service.finish(user_id="idem", session_id=result["session_id"])
        self.assertEqual(first["derived_level"], second["derived_level"])
        self.assertEqual(first["per_track_levels"], second["per_track_levels"])
        self.assertEqual(
            service.get_completion(user_id="idem")["derived_level"],
            first["derived_level"],
        )
        self.assertTrue(service.get_completion(user_id="idem")["completed_at"])

    async def test_finish_without_answers_is_rejected(self):
        service = await self._service()
        started = await service.start_session(user_id="empty", track="backend")
        with self.assertRaises(ValueError):
            await service.finish(
                user_id="empty", session_id=started["session_id"]
            )

    async def test_a_dead_question_source_falls_back_to_the_catalog(self):
        service = _build_service(
            self.planner,
            sources=[_FailingSource(), CatalogQuestionSource(self._parser_for_tests())],
        )
        started = await service.start_session(user_id="fallback", track="backend")
        question = started["question"]
        assert question is not None
        self.assertEqual(question["source"], "catalog")

    async def test_a_service_without_any_source_raises_unavailable(self):
        service = _build_service(self.planner, sources=[_FailingSource()])
        with self.assertRaises(PlacementUnavailableError):
            await service.start_session(user_id="nosource", track="backend")


class CatalogQuestionSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = DocParser(curriculum_path=str(get_settings().curriculum_path))
        self.source = CatalogQuestionSource(self.parser)

    def test_catalog_questions_are_four_option_mcqs_with_a_valid_key(self):
        seen: set[str] = set()
        for track in ("backend", "frontend", "system_design", "ai_stack"):
            for level in LEVELS:
                questions = self.source.questions(
                    track=track, level=level, limit=3, exclude=set()
                )
                self.assertTrue(questions, f"{track}/{level} produced no questions")
                for question in questions:
                    self.assertNotIn(question["question_id"], seen)
                    seen.add(question["question_id"])
                    options = question["options"]
                    self.assertEqual(len(options), 4)
                    self.assertEqual(len(set(options)), 4)
                    self.assertIn(question["correct_answer"], ("A", "B", "C", "D"))
                    self.assertTrue(grade_answer(question, question["correct_answer"]))
                    self.assertTrue(
                        all(
                            grade_answer(question, label) is False
                            for label in ("A", "B", "C", "D")
                            if label != question["correct_answer"]
                        )
                    )
                    self.assertEqual(question["level"], level)
                    self.assertEqual(question["source"], "catalog")

    def test_catalog_questions_are_deterministic_and_excludable(self):
        first = self.source.questions(
            track="backend", level="mid", limit=4, exclude=set()
        )
        again = self.source.questions(
            track="backend", level="mid", limit=4, exclude=set()
        )
        self.assertEqual(
            [question["question_id"] for question in first],
            [question["question_id"] for question in again],
        )
        excluded = self.source.questions(
            track="backend",
            level="mid",
            limit=4,
            exclude={question["question_id"] for question in first},
        )
        self.assertTrue(excluded)
        self.assertNotIn(
            {question["question_id"] for question in first}
            & {question["question_id"] for question in excluded},
            set(),
        )

    def test_senior_questions_probe_the_topic_sections(self):
        questions = self.source.questions(
            track="backend", level="senior", limit=20, exclude=set()
        )
        self.assertTrue(questions)
        for question in questions:
            self.assertTrue(question["question"].startswith('In "'))
            self.assertIn("which section covers", question["question"])

    def test_catalog_skips_scaffolding_sections(self):
        questions = self.source.questions(
            track="system_design", level="mid", limit=60, exclude=set()
        )
        concepts = {question["concept"].lower() for question in questions}
        for meta in (
            "common pitfalls",
            "debugging workflow",
            "design review questions",
            "failure modes",
            "implementation checklist",
            "operational signals",
            "practice exercise",
            "tradeoff analysis",
        ):
            self.assertNotIn(meta, concepts)

    def test_unknown_track_has_no_questions(self):
        # ``DocParser.list_topics`` ignores an unknown track filter, so the
        # source must filter on the summary's own track.
        self.assertEqual(
            self.source.questions(
                track="nonexistent", level="mid", limit=3, exclude=set()
            ),
            [],
        )


class PlacementRouterContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "learning.db")
        self.planner = LearningPlannerStore(self.db_path)
        self.learning_store = LearningStore(self.db_path)
        self.parser = DocParser(curriculum_path=str(get_settings().curriculum_path))
        placement_router.init(self.planner, self.parser)
        self.addCleanup(self._reset_router)

        self.app = FastAPI()
        self.app.include_router(placement_router.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "contract-user",
            "provider": "local",
        }
        self.client = TestClient(self.app)

    def _reset_router(self) -> None:
        placement_router._service = None

    def _start(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.client.post(
            "/api/learning/placement/start", json=payload or {"track": "backend"}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _answer_question(self, session_id: str, question: dict[str, Any], correct: bool) -> dict[str, Any]:
        state = self._service_state(session_id)
        banked = [
            item
            for items in state["bank"].values()
            for item in items
            if item["question_id"] == question["question_id"]
        ]
        self.assertTrue(banked, "question is not banked on the server")
        selected = banked[0]["correct_answer"] if correct else _first_wrong_label(
            banked[0]["correct_answer"], len(banked[0]["options"])
        )
        response = self.client.post(
            "/api/learning/placement/answer",
            json={
                "session_id": session_id,
                "question_id": question["question_id"],
                "selected_answer": selected,
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _service_state(self, session_id: str) -> dict[str, Any]:
        service = placement_router.get_service()
        state = service.get_session(user_id="contract-user", session_id=session_id)
        assert state is not None
        return state

    def test_start_answer_finish_flow_and_profile_seeding(self):
        started = self._start()
        session_id = started["session_id"]
        self.assertEqual(started["status"], "active")
        self.assertEqual(started["level"], "mid")
        self.assertEqual(started["max_steps"], 12)
        question = started["question"]
        self.assertIsNotNone(question)
        assert question is not None
        self.assertEqual(len(question["options"]), 4)
        self.assertNotIn("correct_answer", question)
        self.assertNotIn("explanation", question)

        outcomes = [True, False, True, False, True, False, True, False]
        answered: dict[str, Any] | None = None
        for index, outcome in enumerate(outcomes):
            state = self._service_state(session_id)
            level = state["level"]
            banked = [
                item
                for item in state["bank"].get(level, [])
                if item["question_id"] not in set(state["asked"])
            ]
            self.assertTrue(banked, f"no question available at {level}")
            answer = self._answer_question(session_id, banked[0], outcome)
            answered = answer
            if answer["complete"]:
                break
            next_question = answer["question"]
            self.assertIsNotNone(next_question)
            assert next_question is not None
            self.assertNotIn("correct_answer", next_question)
        assert answered is not None
        self.assertTrue(answered["complete"])
        self.assertGreaterEqual(answered["step_index"], 3)

        finish = self.client.post(
            "/api/learning/placement/finish", json={"session_id": session_id}
        )
        self.assertEqual(finish.status_code, 200)
        summary = finish.json()
        self.assertIn(summary["track"], ("backend", "frontend", "system_design", "ai_stack"))
        self.assertIn(summary["derived_level"], ("junior", "mid", "senior"))
        self.assertEqual(
            summary["per_track_levels"], {"backend": summary["derived_level"]}
        )
        self.assertGreaterEqual(summary["steps_taken"], 3)
        self.assertLessEqual(len(summary["level_path"]), 12)
        self.assertEqual(summary["profile"]["current_level"], summary["derived_level"])
        self.assertEqual(summary["profile"]["primary_track"], "backend")

        status = self.client.get("/api/learning/placement/status")
        self.assertEqual(status.status_code, 200)
        payload = status.json()
        self.assertTrue(payload["completed"])
        self.assertEqual(payload["track"], "backend")
        self.assertEqual(payload["derived_level"], summary["derived_level"])

        self.assertEqual(
            self.planner.get_profile(user_id="contract-user")["current_level"],
            summary["derived_level"],
        )

    def test_start_requires_auth(self):
        anonymous = FastAPI()
        anonymous.include_router(placement_router.router)
        client = TestClient(anonymous)
        for method, path, payload in (
            ("post", "/api/learning/placement/start", {"track": "backend"}),
            (
                "post",
                "/api/learning/placement/answer",
                {"session_id": "x" * 12, "question_id": "q", "selected_answer": "A"},
            ),
            ("post", "/api/learning/placement/finish", {"session_id": "x" * 12}),
            ("get", "/api/learning/placement/status", None),
        ):
            response = (
                getattr(client, method)(path, json=payload)
                if method == "post"
                else getattr(client, method)(path)
            )
            self.assertEqual(response.status_code, 401, path)

    def test_placement_endpoints_respect_the_adaptive_learning_gate(self):
        settings = get_settings()
        previous = settings.enable_adaptive_learning
        settings.enable_adaptive_learning = False
        try:
            response = self.client.post(
                "/api/learning/placement/start", json={"track": "backend"}
            )
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["detail"], "Adaptive learning disabled")
        finally:
            settings.enable_adaptive_learning = previous

    def test_answer_rejects_unknown_sessions_and_questions(self):
        started = self._start()
        missing_session = self.client.post(
            "/api/learning/placement/answer",
            json={
                "session_id": "does-not-exist",
                "question_id": "nope",
                "selected_answer": "A",
            },
        )
        self.assertEqual(missing_session.status_code, 409)

        unknown_question = self.client.post(
            "/api/learning/placement/answer",
            json={
                "session_id": started["session_id"],
                "question_id": "not-in-session",
                "selected_answer": "A",
            },
        )
        self.assertEqual(unknown_question.status_code, 409)

    def test_answer_rejects_a_selection_that_is_not_an_option(self):
        started = self._start()
        question = started["question"]
        assert question is not None
        response = self.client.post(
            "/api/learning/placement/answer",
            json={
                "session_id": started["session_id"],
                "question_id": question["question_id"],
                "selected_answer": "Z",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("option", response.json()["detail"])

    def test_answer_accepts_the_option_text_as_a_selection(self):
        started = self._start()
        question = started["question"]
        assert question is not None
        response = self.client.post(
            "/api/learning/placement/answer",
            json={
                "session_id": started["session_id"],
                "question_id": question["question_id"],
                "selected_answer": question["options"][0],
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_finish_before_any_answer_is_rejected(self):
        started = self._start()
        response = self.client.post(
            "/api/learning/placement/finish", json={"session_id": started["session_id"]}
        )
        self.assertEqual(response.status_code, 409)


class StrictV2QuestionSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.parser = DocParser(curriculum_path=str(get_settings().curriculum_path))

    async def test_maps_strict_v2_questions_onto_placement_payloads(self):
        from app.services.placement_service import StrictV2QuestionSource

        generator = _FakeQuizGenerator(
            [
                SimpleNamespace(
                    question_id="q-1",
                    question="  Which HTTP method is idempotent?  ",
                    choices=[
                        SimpleNamespace(label="A", text="GET"),
                        SimpleNamespace(label="B", text="POST"),
                        SimpleNamespace(label="C", text="PATCH"),
                        SimpleNamespace(label="D", text="TRACE"),
                    ],
                    correct_answer="a",
                    explanation="GET is safe and idempotent.",
                    topic_id="01-backend-fundamentals-and-http",
                    learning_objective="HTTP semantics",
                )
            ]
        )
        source = StrictV2QuestionSource(generator, self.parser)
        questions = await source.questions(
            track="backend", level="mid", limit=3, exclude=set()
        )
        self.assertEqual(len(questions), 1)
        question = questions[0]
        self.assertEqual(question["question"], "Which HTTP method is idempotent?")
        self.assertEqual(question["options"], ["GET", "POST", "PATCH", "TRACE"])
        self.assertEqual(question["correct_answer"], "A")
        self.assertEqual(question["source"], "quiz_v2")
        self.assertEqual(question["topic_title"], "Backend Fundamentals and HTTP")
        self.assertTrue(grade_answer(question, "A"))
        self.assertFalse(grade_answer(question, "B"))
        self.assertEqual(generator.calls[0]["level"], "mid")
        self.assertEqual(generator.calls[0]["question_types"], ["mcq"])

    async def test_questions_without_a_resolvable_answer_are_dropped(self):
        from app.services.placement_service import StrictV2QuestionSource

        generator = _FakeQuizGenerator(
            [
                SimpleNamespace(
                    question_id="q-ambiguous",
                    question="Something?",
                    choices=[
                        SimpleNamespace(label="A", text="one"),
                        SimpleNamespace(label="B", text="two"),
                    ],
                    correct_answer="ZZZ",
                    explanation="",
                    topic_id="01-backend-fundamentals-and-http",
                    learning_objective="",
                )
            ]
        )
        source = StrictV2QuestionSource(generator, self.parser)
        self.assertEqual(
            await source.questions(
                track="backend", level="mid", limit=3, exclude=set()
            ),
            [],
        )


class _FakeQuizGenerator:
    def __init__(self, response, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_quiz_v2(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(questions=self._response)


class _FailingSource:
    """A source that always raises, standing in for a dead LLM."""

    source_name = "failing"

    async def questions(self, **_kwargs):
        raise RuntimeError("llm unavailable")


def _first_wrong_label(correct: str, option_count: int) -> str:
    for label in ("A", "B", "C", "D"):
        if label != correct and ord(label) - ord("A") < option_count:
            return label
    raise AssertionError("no wrong label available")


if __name__ == "__main__":
    unittest.main()
