"""The LLM-visible learner-state block and its prompt seams."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from app.routers import chat
from app.schemas.models import ChatFollowUpRequest
from app.services import learner_memory
from app.services import question_generator as qg
from app.services.interview_generator import InterviewGenerator
from app.services.learner_memory import build_learner_state_block

BACKEND = Path(__file__).resolve().parents[1]
EVALS = BACKEND / "evals"

LEARNER_STATE_LABEL = "Learner state (server-side, trusted context)"

_TOPIC = {
    "id": "01-caching",
    "title": "Caching Fundamentals",
    "content": "Cache invalidation is the hard part of caching.",
    "response_detail": "very_detailed",
    "preferred_language": "",
    "requires_programming": False,
}

_WEAK_AREAS = [
    {
        "topic_id": "14-consistency",
        "attempts": 12,
        "correct_attempts": 5,
        "accuracy": 0.42,
        "mastery_score": 0.31,
        "due_count": 3,
    },
    {
        "topic_id": "01-caching",
        "attempts": 20,
        "correct_attempts": 14,
        "accuracy": 0.70,
        "mastery_score": 0.55,
        "due_count": 1,
    },
    {
        "topic_id": "00-intro",
        "attempts": 0,
        "correct_attempts": 0,
        "accuracy": 0.0,
        "mastery_score": 0.0,
        "due_count": 0,
    },
]

_OBJECTIVES = [
    {
        "topic_id": "14-consistency",
        "learning_objective": "Explain cache invalidation on writes",
        "attempts": 11,
        "correct_attempts": 4,
        "accuracy": 0.36,
        "uncertainty": 0.21,
        "priority": 0.79,
    },
    {
        "topic_id": "14-consistency",
        "learning_objective": "Compare read-through and cache-aside",
        "attempts": 8,
        "correct_attempts": 6,
        "accuracy": 0.75,
        "uncertainty": 0.12,
        "priority": 0.31,
    },
]

_ERROR_PATTERNS = [
    {
        "topic_id": "14-consistency",
        "objective": "explain cache invalidation",
        "error_type": "misconception",
        "explanation": "Confuses write-through with write-back invalidation.",
        "evidence": "the client keeps a copy",
        "count": 4,
        "first_seen": "2026-01-01T00:00:00Z",
        "last_seen": "2026-01-05T00:00:00Z",
    },
    {
        "topic_id": "01-caching",
        "objective": "explain cache invalidation",
        "error_type": "misconception",
        "explanation": "Still confuses write-through with write-back invalidation.",
        "evidence": "still keeping a copy",
        "count": 2,
        "first_seen": "2026-01-02T00:00:00Z",
        "last_seen": "2026-01-06T00:00:00Z",
    },
    {
        "topic_id": "14-consistency",
        "objective": "compare caches",
        "error_type": "application_error",
        "explanation": "Applies TTL semantics where explicit invalidation is required.",
        "evidence": "the data never expires",
        "count": 2,
        "first_seen": "2026-01-03T00:00:00Z",
        "last_seen": "2026-01-07T00:00:00Z",
    },
]

_INTERVIEW_POINTS = [
    {
        "session_id": "session-1",
        "completed_at": "2026-01-01T00:00:00Z",
        "overall_score": 52.0,
        "rubric_averages": {
            "technical_accuracy": 2.4,
            "reasoning_depth": 1.9,
            "communication_clarity": 3.1,
            "completeness": 2.2,
            "confidence_signal": 3.4,
            "overall": 52.0,
        },
        "track": "backend",
        "level": "mid",
        "interview_type": "mixed",
        "readiness_label": "developing",
    },
    {
        "session_id": "session-2",
        "completed_at": "2026-01-08T00:00:00Z",
        "overall_score": 61.0,
        "rubric_averages": {
            "technical_accuracy": 3.0,
            "reasoning_depth": 2.1,
            "communication_clarity": 3.3,
            "completeness": 2.6,
            "confidence_signal": 3.6,
            "overall": 61.0,
        },
        "track": "backend",
        "level": "mid",
        "interview_type": "mixed",
        "readiness_label": "developing",
    },
]

_COMPETENCIES = [
    {
        "competency_id": "track:backend",
        "competency_type": "track",
        "label": "Backend",
        "track": "backend",
        "score": 41.5,
        "confidence_gap": 3.0,
        "uncertainty": 0.1,
        "evidence_count": 32,
        "priority": 90.0,
        "source": "profile_plus_learning",
        "metadata": {},
        "updated_at": "2026-01-08T00:00:00Z",
    },
]


class FakeLearningStore:
    """Duck-typed stand-in for LearningStore's three learner-state accessors."""

    def __init__(
        self,
        *,
        weak_areas: list[dict] | None = None,
        objectives: list[dict] | None = None,
        error_patterns: list[dict] | None = None,
        raises: set[str] | None = None,
    ):
        self.weak_areas = _WEAK_AREAS if weak_areas is None else weak_areas
        self.objectives = _OBJECTIVES if objectives is None else objectives
        self.error_patterns = _ERROR_PATTERNS if error_patterns is None else error_patterns
        self.raises = raises or set()
        self.calls: list[tuple[str, dict]] = []

    def get_weak_areas(self, *, user_id: str, limit: int = 10) -> dict:
        self.calls.append(("get_weak_areas", {"user_id": user_id, "limit": limit}))
        if "get_weak_areas" in self.raises:
            raise RuntimeError("weak areas unavailable")
        return {"weak_areas": list(self.weak_areas)}

    def select_next_objectives(
        self, *, user_id: str, topic_ids: list[str], limit: int = 5
    ) -> list[dict]:
        self.calls.append(
            (
                "select_next_objectives",
                {"user_id": user_id, "topic_ids": list(topic_ids), "limit": limit},
            )
        )
        if "select_next_objectives" in self.raises:
            raise RuntimeError("objectives unavailable")
        wanted = {str(tid) for tid in topic_ids}
        return [row for row in self.objectives if str(row["topic_id"]) in wanted][:limit]

    def list_error_patterns(
        self, *, user_id: str, topic_id: str | None = None, limit: int = 20, min_count: int = 1
    ) -> list[dict]:
        self.calls.append(
            ("list_error_patterns", {"user_id": user_id, "topic_id": topic_id, "limit": limit})
        )
        if "list_error_patterns" in self.raises:
            raise RuntimeError("error patterns unavailable")
        rows = list(self.error_patterns)
        if topic_id:
            rows = [row for row in rows if str(row.get("topic_id") or "") == topic_id]
        rows.sort(key=lambda row: -int(row.get("count") or 0))
        return rows[:limit]


class FakeLearningStoreWithoutErrorPatterns(FakeLearningStore):
    """A store that predates error-pattern memory: the accessor is absent."""

    list_error_patterns = None


class FakeInterviewStore:
    """Duck-typed stand-in for InterviewStore.get_trends."""

    def __init__(self, *, points: list[dict] | None = None, raises: bool = False):
        self.points = _INTERVIEW_POINTS if points is None else points
        self.raises = raises
        self.calls: list[dict] = []

    def get_trends(self, *, user_id: str, limit: int = 5) -> dict:
        self.calls.append({"user_id": user_id, "limit": limit})
        if self.raises:
            raise RuntimeError("interview history unavailable")
        return {"points": list(self.points), "summary": {"session_count": len(self.points)}}


class FakePlanner:
    """Duck-typed stand-in for LearningPlannerStore.list_competencies."""

    def __init__(self, *, competencies: list[dict] | None = None):
        self.competencies = _COMPETENCIES if competencies is None else competencies

    def list_competencies(self, *, user_id: str, limit: int = 50) -> list[dict]:
        return list(self.competencies)


class RecordingLLM:
    """Captures the prompt the generator finally sends."""

    def __init__(self, analysis: str):
        self.analysis = analysis
        self.last_prompt = ""

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.last_prompt = prompt
        return {
            "success": True,
            "analysis": self.analysis,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


def _learner_state_fixture(section_title: str, first_line: str) -> str:
    return (
        f'<untrusted_input label="{LEARNER_STATE_LABEL}">\n'
        f"{section_title}:\n{first_line}\n</untrusted_input>"
    )


def _fenced_body(block: str) -> str:
    """The learner content inside the single fenced block."""
    start = block.index(">") + 1
    end = block.rindex("</untrusted_input>")
    return block[start:end].strip()


class LearnerStateBlockTests(unittest.TestCase):
    def setUp(self):
        self.learning_store = FakeLearningStore()
        learner_memory.init(
            learning_store=self.learning_store,
            learning_planner=FakePlanner(),
            interview_store=FakeInterviewStore(),
        )
        self.addCleanup(learner_memory.init)

    # ── gathering ─────────────────────────────────────────────────

    def test_empty_user_id_has_no_block(self):
        self.assertEqual(build_learner_state_block(""), "")
        self.assertEqual(build_learner_state_block("   "), "")
        self.assertEqual(build_learner_state_block(None), "")

    def test_no_data_has_no_block(self):
        learner_memory.init(
            learning_store=FakeLearningStore(weak_areas=[], objectives=[], error_patterns=[]),
            learning_planner=FakePlanner(competencies=[]),
            interview_store=FakeInterviewStore(points=[]),
        )
        self.assertEqual(build_learner_state_block("alice"), "")

    def test_no_stores_installed_has_no_block(self):
        learner_memory.init()
        self.assertEqual(build_learner_state_block("alice"), "")

    def test_non_positive_budget_has_no_block(self):
        self.assertEqual(build_learner_state_block("alice", max_chars=0), "")
        self.assertEqual(build_learner_state_block("alice", max_chars=-5), "")

    def test_block_sections_appear_in_priority_order(self):
        block = build_learner_state_block("alice")
        self.assertTrue(
            block.startswith(f'<untrusted_input label="{LEARNER_STATE_LABEL}">'),
            block,
        )
        self.assertTrue(block.endswith("</untrusted_input>"), block)

        weak_at = block.index("Weak areas and mastery")
        objectives_at = block.index("Current learning objectives")
        errors_at = block.index("Recurring error patterns")
        interview_at = block.index("Interview weak dimensions")
        self.assertLess(weak_at, objectives_at)
        self.assertLess(objectives_at, errors_at)
        self.assertLess(errors_at, interview_at)

    def test_block_carries_weak_areas_objectives_error_patterns_and_interview(self):
        block = build_learner_state_block("alice", track="backend")
        self.assertIn(
            "14-consistency: mastery 0.31, accuracy 42% over 12 attempt(s), 3 due",
            block,
        )
        self.assertIn("01-caching: mastery 0.55, accuracy 70% over 20 attempt(s), 1 due", block)
        self.assertIn("backend track mastery: 42/100", block)
        self.assertIn(
            "Explain cache invalidation on writes (accuracy 36% over 11 attempt(s))",
            block,
        )
        self.assertIn("misconception (seen 4x): Confuses write-through", block)
        self.assertIn("application_error (seen 2x): Applies TTL semantics", block)
        self.assertIn("reasoning depth: 2.0/5", block)
        # A topic with no reviews yet is not a weak area.
        self.assertNotIn("00-intro", block)

    def test_topic_scope_filters_weak_areas_and_error_patterns(self):
        block = build_learner_state_block("alice", topic_id="14-consistency")
        self.assertIn("14-consistency", block)
        self.assertNotIn("01-caching: mastery", block)

        weak_call = self.learning_store.calls[0]
        self.assertEqual(weak_call[0], "get_weak_areas")
        self.assertEqual(weak_call[1]["user_id"], "alice")
        error_calls = [
            call for call in self.learning_store.calls
            if call[0] == "list_error_patterns"
        ]
        self.assertTrue(error_calls)
        self.assertEqual(error_calls[0][1]["topic_id"], "14-consistency")

    def test_objectives_scope_to_the_named_topic(self):
        build_learner_state_block("alice", topic_id="14-consistency")
        call = next(
            call for call in self.learning_store.calls
            if call[0] == "select_next_objectives"
        )
        self.assertEqual(call[1]["topic_ids"], ["14-consistency"])

    def test_objectives_scope_to_the_weak_topics_when_no_topic_is_named(self):
        build_learner_state_block("alice")
        call = next(
            call for call in self.learning_store.calls
            if call[0] == "select_next_objectives"
        )
        self.assertEqual(call[1]["topic_ids"], ["14-consistency", "01-caching"])

    def test_interview_dimensions_are_the_weakest_three(self):
        block = build_learner_state_block("alice", track="backend")
        self.assertIn("Interview weak dimensions", block)
        self.assertIn("reasoning depth: 2.0/5", block)
        self.assertIn("completeness: 2.4/5", block)
        self.assertIn("technical accuracy: 2.7/5", block)
        # Stronger dimensions stay out of the weakest-three list.
        self.assertNotIn("communication clarity", block)
        self.assertNotIn("confidence signal", block)

    def test_interview_dimensions_are_track_scoped(self):
        other_track = [dict(point, track="frontend") for point in _INTERVIEW_POINTS]
        learner_memory.init(
            learning_store=self.learning_store,
            interview_store=FakeInterviewStore(points=other_track),
        )
        self.assertNotIn(
            "Interview weak dimensions",
            build_learner_state_block("alice", track="backend"),
        )

    def test_error_patterns_are_deduped_by_type(self):
        block = build_learner_state_block("alice")
        misconceptions = [
            line for line in block.splitlines() if "misconception" in line
        ]
        self.assertEqual(len(misconceptions), 1)
        self.assertIn("seen 4x", misconceptions[0])

    def test_block_content_stays_within_the_budget(self):
        budget = 260
        block = build_learner_state_block("alice", max_chars=budget)
        self.assertLessEqual(len(_fenced_body(block)), budget)
        # The walk stops at the first section that does not fit, so the
        # earlier, higher-priority section survives and later ones do not.
        self.assertIn("Weak areas and mastery", block)
        self.assertNotIn("Interview weak dimensions", block)

    def test_a_later_smaller_section_still_fits_a_tight_budget(self):
        learner_memory.init(
            learning_store=FakeLearningStore(weak_areas=[], objectives=[]),
            learning_planner=FakePlanner(competencies=[]),
            interview_store=FakeInterviewStore(),
        )
        block = build_learner_state_block("alice", max_chars=1200)
        self.assertIn("Recurring error patterns", block)
        self.assertIn("Interview weak dimensions", block)
        self.assertNotIn("Weak areas and mastery", block)

    # ── failure tolerance ─────────────────────────────────────────

    def test_total_store_failure_yields_no_block_and_never_raises(self):
        learner_memory.init(
            learning_store=FakeLearningStore(
                raises={"get_weak_areas", "select_next_objectives", "list_error_patterns"}
            ),
            learning_planner=FakePlanner(),
            interview_store=FakeInterviewStore(raises=True),
        )
        self.assertEqual(build_learner_state_block("alice"), "")

    def test_partial_failure_degrades_to_the_surviving_sections(self):
        learner_memory.init(
            learning_store=FakeLearningStore(
                raises={"get_weak_areas", "select_next_objectives"}
            ),
            learning_planner=FakePlanner(),
            interview_store=FakeInterviewStore(),
        )
        block = build_learner_state_block("alice")
        self.assertNotIn("Weak areas and mastery", block)
        self.assertNotIn("Current learning objectives", block)
        self.assertIn("Recurring error patterns", block)
        self.assertIn("Interview weak dimensions", block)

    def test_missing_accessors_are_skipped_not_invented(self):
        learner_memory.init(
            learning_store=FakeLearningStoreWithoutErrorPatterns(),
            interview_store=FakeInterviewStore(points=[]),
        )
        block = build_learner_state_block("alice")
        self.assertIn("Weak areas and mastery", block)
        self.assertNotIn("Recurring error patterns", block)

    def test_absent_planner_skips_the_track_mastery_line(self):
        learner_memory.init(
            learning_store=self.learning_store,
            learning_planner=None,
            interview_store=FakeInterviewStore(),
        )
        block = build_learner_state_block("alice", track="backend")
        self.assertIn("Weak areas and mastery", block)
        self.assertNotIn("track mastery", block)


class QuestionPromptSeamTests(unittest.TestCase):
    """The no-state prompt path must stay byte-identical to the locked digests."""

    def test_question_prompt_without_learner_state_is_unchanged(self):
        baseline = qg._build_prompt(
            topic_id=_TOPIC["id"],
            topic_title=_TOPIC["title"],
            doc_content=_TOPIC["content"],
            count=1,
        )
        explicit_empty = qg._build_prompt(
            topic_id=_TOPIC["id"],
            topic_title=_TOPIC["title"],
            doc_content=_TOPIC["content"],
            count=1,
            learner_state="",
        )
        self.assertEqual(baseline, explicit_empty)
        self.assertNotIn(LEARNER_STATE_LABEL, baseline)

    def test_quiz_prompt_without_learner_state_is_unchanged(self):
        baseline = qg._build_quiz_prompt([_TOPIC], count=1)
        explicit_empty = qg._build_quiz_prompt([_TOPIC], count=1, learner_state="")
        self.assertEqual(baseline, explicit_empty)
        self.assertNotIn(LEARNER_STATE_LABEL, baseline)

    def test_question_prompt_includes_the_learner_state_block(self):
        learner_state = _learner_state_fixture(
            "Weak areas and mastery (weakest first; target these)",
            "- 14-consistency: mastery 0.31, accuracy 42% over 12 attempt(s), 3 due",
        )
        prompt = qg._build_prompt(
            topic_id=_TOPIC["id"],
            topic_title=_TOPIC["title"],
            doc_content=_TOPIC["content"],
            count=1,
            learner_state=learner_state,
        )
        self.assertIn(learner_state, prompt)

    def test_quiz_prompt_includes_the_learner_state_block(self):
        learner_state = _learner_state_fixture(
            "Weak areas and mastery (weakest first; target these)",
            "- 14-consistency: mastery 0.31, accuracy 42% over 12 attempt(s), 3 due",
        )
        prompt = qg._build_quiz_prompt([_TOPIC], count=1, learner_state=learner_state)
        self.assertIn(learner_state, prompt)

    def test_quiz_entry_point_threads_learner_state_into_the_prompt(self):
        llm = RecordingLLM(json.dumps(_VALID_QUIZ_ITEMS))
        generator = qg.QuestionGenerator(llm)
        learner_state = _learner_state_fixture(
            "Weak areas and mastery (weakest first; target these)",
            "- 14-consistency: mastery 0.31, accuracy 42% over 12 attempt(s), 3 due",
        )
        asyncio.run(
            generator.generate_quiz_v2(
                topics_content=[_TOPIC],
                count=1,
                llm_config=None,
                learner_state=learner_state,
            )
        )
        self.assertIn(learner_state, llm.last_prompt)

    def test_golden_prompt_digests_still_lock_the_no_state_path(self):
        """The offline digests the eval harness enforces, asserted per flow.

        ``rebuild_prompt`` never passes ``learner_state``, so a drift here
        means a no-state prompt path changed.
        """
        from scripts import eval_agent_quality as harness

        cases = {
            case.case_id: case
            for case in harness.load_cases(EVALS)
            if case.flow in {"questions", "quiz", "interview_question"}
        }
        for case_id in (
            "questions.conceptual.prior_progress.mid.count2",
            "quiz.multi_topic.mid.count4",
            "interview_question.coding.mid.middle",
        ):
            with self.subTest(case=case_id):
                case = cases[case_id]
                self.assertTrue(case.prompt, case_id)
                self.assertTrue(case.digest_expected, case_id)
                self.assertEqual(case.digest_expected, case.prompt_digest, case_id)
                self.assertNotIn(LEARNER_STATE_LABEL, case.prompt, case_id)


_VALID_QUIZ_ITEMS = [
    {
        "question": "Which HTTP mechanism lets a client validate a cached response cheaply?",
        "type": "mcq",
        "choices": [
            {"label": "A", "text": "ETag"},
            {"label": "B", "text": "Age"},
            {"label": "C", "text": "Vary"},
            {"label": "D", "text": "Expires"},
        ],
        "correct_answer": "A",
        "explanation": "The ETag header carries the cached response validator.",
        "difficulty": "medium",
        "topic_id": "01-caching",
        "source_quote": "Cache invalidation is the hard part of caching.",
        "reasoning_summary": "Identify the response validator header.",
        "target_level": "mid",
        "learning_objective": "Explain which header validates a cached response.",
        "expected_answer": "ETag",
        "distractor_rationales": [
            {"label": "B", "rationale": "Age reports freshness lifetime, not a validator."},
            {"label": "C", "rationale": "Vary selects cache variants, not a validator."},
            {"label": "D", "rationale": "Expires gives an absolute expiry, not a validator."},
        ],
    }
]


class InterviewPromptSeamTests(unittest.TestCase):
    def _session(self) -> dict:
        return {
            "session_id": "session-1",
            "user_id": "alice",
            "track": "backend",
            "level": "mid",
            "interview_type": "mixed",
            "turn_count": 5,
            "turns_completed": 0,
            "status": "active",
            "target_role": "Backend Engineer",
            "interviewer_style": "neutral",
            "feedback_mode": "concise",
            "job_description_text": "",
            "resume_summary_text": "",
            "focus_areas": [],
            "asked_questions": [],
            "current_question": "",
            "memory_summary": "",
        }

    def _question_payload(self) -> str:
        return json.dumps(
            {
                "question": "How do you invalidate a cache entry when one write touches it?",
                "competency_focus": "consistency",
                "expected_signals": ["explicit invalidation", "stampede control"],
            }
        )

    def test_question_prompt_without_learner_state_is_unchanged(self):
        baseline = InterviewGenerator._question_prompt(self._session(), [])
        explicit_empty = InterviewGenerator._question_prompt(
            self._session(), [], learner_state=""
        )
        self.assertEqual(baseline, explicit_empty)
        self.assertNotIn(LEARNER_STATE_LABEL, baseline)

    def test_question_prompt_includes_the_learner_state_block(self):
        learner_state = _learner_state_fixture(
            "Interview weak dimensions (recent average, weakest first)",
            "- reasoning depth: 2.1/5",
        )
        prompt = InterviewGenerator._question_prompt(
            self._session(), [], learner_state=learner_state
        )
        self.assertIn(learner_state, prompt)

    def test_generate_question_builds_the_block_from_user_identity(self):
        llm = RecordingLLM(self._question_payload())
        generator = InterviewGenerator(llm, None)
        learner_memory.init(
            learning_store=FakeLearningStore(),
            learning_planner=FakePlanner(),
            interview_store=FakeInterviewStore(),
        )
        self.addCleanup(learner_memory.init)
        asyncio.run(
            generator.generate_question(
                session=self._session(),
                turns=[],
                llm_config=None,
                user_identity={"user": "alice"},
            )
        )
        self.assertIn(LEARNER_STATE_LABEL, llm.last_prompt)
        self.assertIn("Interview weak dimensions", llm.last_prompt)

    def test_generate_question_without_user_context_leaves_the_prompt_unchanged(self):
        llm = RecordingLLM(self._question_payload())
        generator = InterviewGenerator(llm, None)
        learner_memory.init(
            learning_store=FakeLearningStore(),
            learning_planner=FakePlanner(),
            interview_store=FakeInterviewStore(),
        )
        self.addCleanup(learner_memory.init)
        asyncio.run(
            generator.generate_question(
                session=self._session(),
                turns=[],
                llm_config=None,
                user_identity=None,
            )
        )
        self.assertNotIn(LEARNER_STATE_LABEL, llm.last_prompt)


class ChatPromptSeamTests(unittest.TestCase):
    def _payload(self, *, use_memory: bool = True) -> ChatFollowUpRequest:
        return ChatFollowUpRequest(
            word="idempotency",
            context_question="What is idempotency?",
            context_answer="Idempotency means repeated requests have the same effect.",
            topic_id="01-caching",
            topic_title="Caching Fundamentals",
            topic_track="backend",
            user_message="How do I keep a cache consistent?",
            history=[],
            use_memory=use_memory,
        )

    def test_follow_up_prompt_includes_learner_state_when_memory_is_enabled(self):
        prev_llm = chat._llm_client
        prev_store = chat._learning_store
        fake = RecordingLLM("- A short answer.")
        try:
            learner_memory.init(
                learning_store=FakeLearningStore(),
                learning_planner=FakePlanner(),
                interview_store=FakeInterviewStore(),
            )
            chat.init(fake)
            asyncio.run(
                chat.chat_follow_up(
                    self._payload(), user={"user": "alice", "provider": "google"}
                )
            )
            self.assertIn(LEARNER_STATE_LABEL, fake.last_prompt)
            self.assertIn("Weak areas and mastery", fake.last_prompt)
        finally:
            chat._llm_client = prev_llm
            chat._learning_store = prev_store
            learner_memory.init()

    def test_follow_up_prompt_omits_learner_state_when_memory_is_disabled(self):
        prev_llm = chat._llm_client
        prev_store = chat._learning_store
        fake = RecordingLLM("- A short answer.")
        try:
            learner_memory.init(
                learning_store=FakeLearningStore(),
                learning_planner=FakePlanner(),
                interview_store=FakeInterviewStore(),
            )
            chat.init(fake)
            asyncio.run(
                chat.chat_follow_up(
                    self._payload(use_memory=False),
                    user={"user": "alice", "provider": "google"},
                )
            )
            self.assertNotIn(LEARNER_STATE_LABEL, fake.last_prompt)
        finally:
            chat._llm_client = prev_llm
            chat._learning_store = prev_store
            learner_memory.init()

    def test_follow_up_prompt_without_learner_data_is_unchanged(self):
        prev_llm = chat._llm_client
        prev_store = chat._learning_store
        fake = RecordingLLM("- A short answer.")
        try:
            learner_memory.init()
            chat.init(fake)
            asyncio.run(
                chat.chat_follow_up(
                    self._payload(), user={"user": "alice", "provider": "google"}
                )
            )
            self.assertNotIn(LEARNER_STATE_LABEL, fake.last_prompt)
        finally:
            chat._llm_client = prev_llm
            chat._learning_store = prev_store
            learner_memory.init()


if __name__ == "__main__":
    unittest.main()
