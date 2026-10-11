"""Unit tests for the timed exam store and service."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from app.services.card_store import CardStore
from app.services.exam_service import (
    ExamError,
    ExamNotFoundError,
    ExamService,
    ExamStateError,
    PASS_THRESHOLD,
)
from app.services.exam_store import ExamStore
from app.services.learning_store import LearningStore


class ExamStoreTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.store = ExamStore(self.db_path)

    def test_create_and_get_exam(self):
        exam = self.store.create_exam(
            user_id="u1",
            title="Net exam",
            questions=[
                {
                    "question_ref": "q1",
                    "type": "mcq",
                    "question_text": "What is HTTP?",
                    "topic_id": "t1",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "B",
                    "expected_answer": "B",
                    "source": "deck",
                }
            ],
            duration_minutes=30,
        )
        self.assertTrue(exam["exam_id"])
        self.assertEqual(exam["status"], "created")
        self.assertEqual(exam["question_count"], 1)
        self.assertEqual(exam["duration_minutes"], 30)
        self.assertEqual(len(exam["questions"]), 1)

        fetched = self.store.get_exam(
            user_id="u1", exam_id=exam["exam_id"]
        )
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Net exam")

        # Other users cannot see it.
        self.assertIsNone(
            self.store.get_exam(
                user_id="u2", exam_id=exam["exam_id"]
            )
        )

    def test_start_exam_stamps_started_at(self):
        exam = self.store.create_exam(
            user_id="u1", title="T", questions=[], duration_minutes=10
        )
        started = self.store.start_exam(
            user_id="u1",
            exam_id=exam["exam_id"],
            started_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(started["status"], "in_progress")
        self.assertEqual(started["started_at"], "2026-01-01T00:00:00+00:00")

        # Starting twice is a conflict.
        again = self.store.start_exam(
            user_id="u1",
            exam_id=exam["exam_id"],
            started_at="2026-01-01T00:05:00+00:00",
        )
        self.assertIsNone(again)

    def test_complete_exam_records_score(self):
        exam = self.store.create_exam(
            user_id="u1", title="T", questions=[], duration_minutes=10
        )
        self.store.start_exam(
            user_id="u1",
            exam_id=exam["exam_id"],
            started_at="2026-01-01T00:00:00+00:00",
        )
        completed = self.store.complete_exam(
            user_id="u1",
            exam_id=exam["exam_id"],
            completed_at="2026-01-01T00:05:00+00:00",
            score=0.8,
            passed=True,
            time_exceeded=False,
        )
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["score"], 0.8)
        self.assertTrue(completed["passed"])
        self.assertFalse(completed["time_exceeded"])

    def test_record_answer_and_get_answers(self):
        exam = self.store.create_exam(
            user_id="u1", title="T", questions=[], duration_minutes=10
        )
        self.store.record_answer(
            user_id="u1",
            exam_id=exam["exam_id"],
            question_ref="q1",
            answer_text="B",
            is_correct=True,
            score=1.0,
            graded_by="auto",
            rubric=None,
            feedback="Correct.",
            created_at="2026-01-01T00:00:00+00:00",
        )
        answers = self.store.get_answers(
            user_id="u1", exam_id=exam["exam_id"]
        )
        self.assertEqual(len(answers), 1)
        self.assertEqual(answers[0]["question_ref"], "q1")
        self.assertEqual(answers[0]["answer_text"], "B")
        self.assertTrue(answers[0]["is_correct"])

    def test_list_exams(self):
        self.store.create_exam(
            user_id="u1", title="A", questions=[], duration_minutes=10
        )
        self.store.create_exam(
            user_id="u1", title="B", questions=[], duration_minutes=20
        )
        self.store.create_exam(
            user_id="u2", title="Other", questions=[], duration_minutes=10
        )
        exams = self.store.list_exams(user_id="u1", limit=50)
        self.assertEqual(len(exams), 2)
        self.assertEqual(
            [e["title"] for e in exams], ["B", "A"]
        )

    def test_completed_exam_days(self):
        first = self.store.create_exam(
            user_id="u1", title="A", questions=[], duration_minutes=10
        )
        self.store.create_exam(
            user_id="u1", title="B", questions=[], duration_minutes=10
        )
        self.store.start_exam(
            user_id="u1",
            exam_id=first["exam_id"],
            started_at="2026-01-01T00:00:00+00:00",
        )
        self.store.complete_exam(
            user_id="u1",
            exam_id=first["exam_id"],
            completed_at="2026-01-01T00:05:00+00:00",
            score=1.0,
            passed=True,
            time_exceeded=False,
        )
        days = self.store.completed_exam_days(user_id="u1")
        self.assertEqual(len(days), 1)
        self.assertIn("2026-01-01", days[0]["completed_at"])


class ExamServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        self.learning = LearningStore(self.db_path)
        self.cards = CardStore(self.db_path)
        self.exam_store = ExamStore(self.db_path)
        self.service = ExamService(
            store=self.exam_store,
            learning_store=self.learning,
            card_store=self.cards,
        )

    def _make_deck_with_cards(self, user_id="u1", count=5):
        deck = self.cards.create_deck(
            user_id=user_id, name="Deck"
        )
        for i in range(count):
            self.cards.create_card(
                user_id=user_id,
                deck_id=deck["deck_id"],
                front=f"Question {i}?",
                back=f"Answer {i}.",
            )
        return deck

    async def test_create_exam_from_deck(self):
        deck = self._make_deck_with_cards()
        exam = await self.service.create_exam(
            user_id="u1",
            title="Deck exam",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=3,
            duration_minutes=15,
            question_types=["mcq"],
        )
        self.assertEqual(exam["status"], "created")
        self.assertEqual(len(exam["questions"]), 3)
        for question in exam["questions"]:
            self.assertEqual(question["type"], "mcq")
            self.assertEqual(len(question["options"]), 4)
            self.assertIn(
                question["correct_answer"], list("ABCD")
            )

    async def test_create_exam_written_when_no_distractors(self):
        deck = self.cards.create_deck(user_id="u1", name="Solo")
        self.cards.create_card(
            user_id="u1",
            deck_id=deck["deck_id"],
            front="Only card?",
            back="Only answer.",
        )
        exam = await self.service.create_exam(
            user_id="u1",
            title="Written exam",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=1,
            duration_minutes=10,
            question_types=["written"],
        )
        self.assertEqual(len(exam["questions"]), 1)
        self.assertEqual(exam["questions"][0]["type"], "written")
        self.assertEqual(exam["questions"][0]["options"], [])

    async def test_create_exam_no_questions_raises(self):
        with self.assertRaises(ExamError):
            await self.service.create_exam(
                user_id="u1",
                title="Empty",
                topic_ids=[],
                deck_ids=[],
                weak_areas=False,
                question_count=5,
                duration_minutes=10,
                question_types=["mcq"],
            )

    async def test_question_count_capped_at_50(self):
        deck = self._make_deck_with_cards(count=60)
        exam = await self.service.create_exam(
            user_id="u1",
            title="Big",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=100,
            duration_minutes=240,
            question_types=["mcq"],
        )
        self.assertLessEqual(len(exam["questions"]), 50)

    async def test_lifecycle_state_errors(self):
        deck = self._make_deck_with_cards()
        exam = await self.service.create_exam(
            user_id="u1",
            title="T",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=1,
            duration_minutes=10,
            question_types=["mcq"],
        )
        exam_id = exam["exam_id"]

        # Submit before start.
        with self.assertRaises(ExamStateError):
            await self.service.submit_exam(
                user_id="u1", exam_id=exam_id, answers=[]
            )
        # Remark before completion.
        with self.assertRaises(ExamStateError):
            await self.service.remark_exam(user_id="u1", exam_id=exam_id)

        await self.service.start_exam(user_id="u1", exam_id=exam_id)
        # Start twice.
        with self.assertRaises(ExamStateError):
            await self.service.start_exam(user_id="u1", exam_id=exam_id)

    async def test_submit_grades_mcq_and_passes(self):
        deck = self._make_deck_with_cards(count=4)
        exam = await self.service.create_exam(
            user_id="u1",
            title="T",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=4,
            duration_minutes=10,
            question_types=["mcq"],
        )
        exam_id = exam["exam_id"]
        await self.service.start_exam(user_id="u1", exam_id=exam_id)

        answers = [
            {
                "question_ref": q["question_ref"],
                "answer_text": q["correct_answer"],
                "time_ms": 1000,
            }
            for q in exam["questions"]
        ]
        result = await self.service.submit_exam(
            user_id="u1", exam_id=exam_id, answers=answers
        )
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["passed"])
        self.assertFalse(result["time_exceeded"])
        self.assertEqual(len(result["per_question"]), 4)
        for item in result["per_question"]:
            self.assertTrue(item["is_correct"])
            self.assertEqual(item["graded_by"], "auto")

        # Exam is now completed; resubmitting is a conflict.
        with self.assertRaises(ExamStateError):
            await self.service.submit_exam(
                user_id="u1", exam_id=exam_id, answers=[]
            )

    async def test_submit_late_is_flagged_but_graded(self):
        deck = self._make_deck_with_cards(count=4)
        exam = await self.service.create_exam(
            user_id="u1",
            title="T",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=4,
            duration_minutes=1,
            question_types=["mcq"],
        )
        exam_id = exam["exam_id"]
        # Start far in the past so the submit is late.
        self.exam_store.start_exam(
            user_id="u1",
            exam_id=exam_id,
            started_at="2020-01-01T00:00:00+00:00",
        )
        answers = [
            {
                "question_ref": q["question_ref"],
                "answer_text": q["correct_answer"],
            }
            for q in exam["questions"]
        ]
        result = await self.service.submit_exam(
            user_id="u1", exam_id=exam_id, answers=answers
        )
        self.assertTrue(result["time_exceeded"])
        self.assertEqual(result["score"], 1.0)

    async def test_mcq_accepts_index_answer(self):
        deck = self._make_deck_with_cards(count=4)
        exam = await self.service.create_exam(
            user_id="u1",
            title="T",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=4,
            duration_minutes=10,
            question_types=["mcq"],
        )
        exam_id = exam["exam_id"]
        await self.service.start_exam(user_id="u1", exam_id=exam_id)
        answers = []
        for q in exam["questions"]:
            letter = q["correct_answer"]
            index = ord(letter) - ord("A")
            answers.append(
                {"question_ref": q["question_ref"], "answer_text": str(index)}
            )
        result = await self.service.submit_exam(
            user_id="u1", exam_id=exam_id, answers=answers
        )
        self.assertEqual(result["score"], 1.0)

    async def test_written_grading_fallback_and_remark(self):
        deck = self.cards.create_deck(user_id="u1", name="Solo")
        self.cards.create_card(
            user_id="u1",
            deck_id=deck["deck_id"],
            front="What is HTTP?",
            back="HyperText Transfer Protocol for web requests.",
        )
        exam = await self.service.create_exam(
            user_id="u1",
            title="Written",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=1,
            duration_minutes=10,
            question_types=["written"],
        )
        exam_id = exam["exam_id"]
        await self.service.start_exam(user_id="u1", exam_id=exam_id)
        result = await self.service.submit_exam(
            user_id="u1",
            exam_id=exam_id,
            answers=[
                {
                    "question_ref": exam["questions"][0]["question_ref"],
                    "answer_text": (
                        "HyperText Transfer Protocol for web requests."
                    ),
                }
            ],
        )
        self.assertEqual(result["per_question"][0]["graded_by"], "auto")
        self.assertGreater(result["score"], 0.0)

        # Remark re-grades deterministically: same score.
        remark = await self.service.remark_exam(user_id="u1", exam_id=exam_id)
        self.assertEqual(remark["new_score"], result["score"])
        self.assertEqual(remark["diff"], 0.0)
        self.assertEqual(remark["changed"], [])

    async def test_get_exam_includes_results(self):
        deck = self._make_deck_with_cards(count=4)
        exam = await self.service.create_exam(
            user_id="u1",
            title="T",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=4,
            duration_minutes=10,
            question_types=["mcq"],
        )
        exam_id = exam["exam_id"]
        await self.service.start_exam(user_id="u1", exam_id=exam_id)
        answers = [
            {
                "question_ref": q["question_ref"],
                "answer_text": q["correct_answer"],
            }
            for q in exam["questions"]
        ]
        await self.service.submit_exam(
            user_id="u1", exam_id=exam_id, answers=answers
        )
        detail = await self.service.get_exam(user_id="u1", exam_id=exam_id)
        self.assertEqual(detail["status"], "completed")
        self.assertEqual(len(detail["results"]), 4)
        self.assertEqual(detail["score"], 1.0)

    async def test_get_exam_not_found(self):
        with self.assertRaises(ExamNotFoundError):
            await self.service.get_exam(user_id="u1", exam_id="missing")

    async def test_list_exams(self):
        deck = self._make_deck_with_cards()
        await self.service.create_exam(
            user_id="u1",
            title="A",
            topic_ids=[],
            deck_ids=[deck["deck_id"]],
            weak_areas=False,
            question_count=1,
            duration_minutes=10,
            question_types=["mcq"],
        )
        exams = await self.service.list_exams(user_id="u1")
        self.assertEqual(len(exams), 1)
        self.assertEqual(exams[0]["title"], "A")

    async def test_generated_topic_questions_carry_the_learner_state(self):
        """Exams must not be learner-blind.

        ``_generate_topic_questions`` used to call ``generate_quiz_v2`` with no
        learner state, so an exam asked the same questions of a learner with
        history as it did of a brand-new one. The seam now mirrors the one
        ``questions.py`` uses.
        """
        captured: dict[str, str] = {}

        class _Response:
            questions: list = []

        class _Generator:
            async def generate_quiz_v2(self, **kwargs):
                captured.update(kwargs)
                return _Response()

        service = ExamService(
            store=self.exam_store,
            learning_store=self.learning,
            card_store=self.cards,
            question_generator=_Generator(),
        )
        service._doc_parser = self._StubParser()
        with patch(
            "app.services.exam_service.build_learner_state_block",
            return_value="learner_state\nWeak areas: 01-http",
        ) as build:
            questions = await service._generate_topic_questions(
                user_id="u1", topic_ids=["01-http", "02-rest"], count=3
            )
        self.assertEqual(questions, [])
        # The fence is gathered through the production seam, scoped to the
        # exam's topics, and handed to the question generator.
        build.assert_called_once()
        self.assertEqual(build.call_args.args[0], "u1")
        self.assertEqual(build.call_args.kwargs["topic_id"], "01-http")
        self.assertIsNone(build.call_args.kwargs["track"])
        self.assertEqual(
            captured["learner_state"], "learner_state\nWeak areas: 01-http"
        )

    class _StubParser:
        def get_topic(self, topic_id):
            return type("Topic", (), {"title": "HTTP"})()

        def get_topic_content(self, topic_id):
            return "HTTP content."

    def test_pass_threshold_constant(self):
        self.assertEqual(PASS_THRESHOLD, 0.7)


if __name__ == "__main__":
    unittest.main()
