"""Timed exam service: question selection, grading, remark re-grade.

- Question selection: weak areas (via
  :meth:`LearningStore.get_weak_areas`) plus due cards
  (via the FSRS store) plus explicit deck/topic sources.
- MCQ answers are auto-graded locally.
- Written answers are graded with an interview-style
  rubric through the LLM client (structured JSON, fixed
  seed) with a deterministic keyword-coverage fallback
  when the LLM is unavailable.
- The remark endpoint re-grades with an identical rubric
  prompt and a fixed per-exam seed, so the same exam +
  answers re-grade identically.

Server-side timestamps are authoritative: ``start`` stamps
``started_at`` and ``submit`` stamps ``completed_at``; a
late submit is flagged ``time_exceeded`` but still graded.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from typing import Any

from app.services.exam_store import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    ExamStore,
)
from app.services.gamification_service import GamificationService
from app.services.learner_memory import build_learner_state_block
from app.services.learning_store import LearningStore
from app.services.card_store import CardStore
from app.services.llm_client import parse_json_object

logger = logging.getLogger(__name__)

#: Hard ceiling on exam size (LLM grading cost).
MAX_QUESTIONS = 50
#: Score (0-1) required to pass an exam.
PASS_THRESHOLD = 0.7
#: Fixed seed prefix — identical for initial grading and remark.
GRADING_SEED = "exam-grading-v1"

MCQ_LABELS = "ABCDEFGHIJ"


class ExamError(Exception):
    """Domain error mapped to an HTTP error by the router."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ExamNotFoundError(ExamError):
    def __init__(self, exam_id: str):
        super().__init__(f"Exam not found: {exam_id}", status_code=404)


class ExamStateError(ExamError):
    def __init__(self, message: str):
        super().__init__(message, status_code=409)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def written_grading_prompt(
    *,
    question_text: str,
    expected_answer: str,
    user_answer: str,
    seed: str,
) -> str:
    """Build the interview-style written-answer grading prompt.

    The seed is embedded so the initial grade and the remark
    re-grade send a byte-identical prompt.
    """
    return (
        f"Grading seed: {seed}\n\n"
        f"Question: {question_text}\n"
        f"Expected answer: {expected_answer}\n"
        f"Candidate answer: {user_answer}\n\n"
        "Score the candidate's answer against the expected answer "
        "using this rubric. Score each dimension as an integer 0-5:\n"
        '- "technical_accuracy": factual correctness vs the expected answer\n'
        '- "reasoning_depth": depth and soundness of reasoning\n'
        '- "communication_clarity": clarity and structure\n'
        '- "completeness": coverage of the key points\n'
        'Also score "overall" as an integer 0-100 and write a one-sentence '
        '"feedback".\n'
        "Return ONLY a JSON object: "
        '{"technical_accuracy": 0-5, "reasoning_depth": 0-5, '
        '"communication_clarity": 0-5, "completeness": 0-5, '
        '"overall": 0-100, "feedback": "..."}'
    )


class ExamService:
    """Create, grade, and re-grade timed exams."""

    def __init__(
        self,
        store: ExamStore,
        learning_store: LearningStore,
        card_store: CardStore | None = None,
        doc_parser: Any | None = None,
        question_generator: Any | None = None,
        llm_client: Any | None = None,
        gamification_service: GamificationService | None = None,
    ):
        self._store = store
        self._learning_store = learning_store
        self._card_store = card_store
        self._doc_parser = doc_parser
        self._question_generator = question_generator
        self._llm_client = llm_client
        self._gamification_service = gamification_service

    # ── Exam creation ──────────────────────────
    async def create_exam(
        self,
        *,
        user_id: str,
        title: str,
        topic_ids: list[str],
        deck_ids: list[str],
        weak_areas: bool,
        question_count: int,
        duration_minutes: int,
        question_types: list[str],
    ) -> dict[str, Any]:
        """Select questions and persist a new exam session."""
        count = max(1, min(int(question_count), MAX_QUESTIONS))
        types = [t for t in question_types if t in ("mcq", "written")]
        if not types:
            types = ["mcq"]

        # Weak areas contribute their topic ids to the pool.
        resolved_topic_ids: list[str] = []
        if weak_areas:
            weak = await self._learning_store.run_async(
                self._learning_store.get_weak_areas,
                user_id=user_id,
                limit=10,
            )
            for area in weak.get("weak_areas", []):
                topic_id = str(area.get("topic_id") or "").strip()
                if topic_id and topic_id not in resolved_topic_ids:
                    resolved_topic_ids.append(topic_id)
        for topic_id in topic_ids:
            topic_id = str(topic_id).strip()
            if topic_id and topic_id not in resolved_topic_ids:
                resolved_topic_ids.append(topic_id)

        questions: list[dict[str, Any]] = []
        seen_refs: set[str] = set()

        def _add(question: dict[str, Any] | None) -> None:
            if question is None:
                return
            ref = str(question["question_ref"])
            if ref in seen_refs:
                return
            seen_refs.add(ref)
            questions.append(question)

        # Due cards: map each due FSRS card back to its deck card.
        due_cards = await self._learning_store.run_async(
            self._learning_store.get_due_cards,
            user_id=user_id,
            limit=MAX_QUESTIONS,
        )
        deck_card_cache: dict[str, list[dict[str, Any]]] = {}
        for card in due_cards:
            deck_card = await self._deck_card_for_fsrs_card(
                user_id=user_id, fsrs_card_id=str(card["card_id"])
            )
            if deck_card is None:
                # A question-source due card: its topic joins the
                # LLM generation pool below.
                topic_id = str(card.get("topic_id") or "").strip()
                if topic_id and topic_id not in resolved_topic_ids:
                    resolved_topic_ids.append(topic_id)
                continue
            deck_id = str(deck_card["deck_id"])
            if deck_id not in deck_card_cache and self._card_store is not None:
                deck_card_cache[deck_id] = await self._card_store.run_async(
                    self._card_store.list_cards,
                    user_id=user_id,
                    deck_id=deck_id,
                    limit=MAX_QUESTIONS,
                )
            _add(
                self._question_from_card(
                    deck_card,
                    question_types=types,
                    deck_cards=deck_card_cache.get(deck_id) or [],
                    source="due_card",
                )
            )

        # Explicit deck sources.
        for deck_id in deck_ids:
            deck_id = str(deck_id).strip()
            if not deck_id or self._card_store is None:
                continue
            deck_cards = await self._card_store.run_async(
                self._card_store.list_cards,
                user_id=user_id,
                deck_id=deck_id,
                limit=MAX_QUESTIONS,
            )
            for deck_card in deck_cards:
                _add(
                    self._question_from_card(
                        deck_card,
                        question_types=types,
                        deck_cards=deck_cards,
                        source="deck",
                    )
                )

        # Weak-area / topic questions via the quiz generator.
        remaining = count - len(questions)
        if resolved_topic_ids and remaining > 0 and self._question_generator is not None:
            for question in await self._generate_topic_questions(
                user_id=user_id,
                topic_ids=resolved_topic_ids,
                count=remaining,
            ):
                _add(question)

        questions = questions[:count]
        if not questions:
            raise ExamError(
                "No questions available for this exam. Add flashcards, "
                "review due cards, or request weak-area generation."
            )

        return await self._store.run_async(
            self._store.create_exam,
            user_id=user_id,
            title=title,
            questions=questions,
            duration_minutes=duration_minutes,
        )

    async def _deck_card_for_fsrs_card(
        self, *, user_id: str, fsrs_card_id: str
    ) -> dict[str, Any] | None:
        if self._card_store is None:
            return None
        return await self._card_store.run_async(
            self._card_store.find_card_by_fsrs_id,
            user_id=user_id,
            fsrs_card_id=fsrs_card_id,
        )

    async def _generate_topic_questions(
        self,
        *,
        user_id: str,
        topic_ids: list[str],
        count: int,
    ) -> list[dict[str, Any]]:
        """Generate MCQ questions for the given topics via the LLM."""
        if self._doc_parser is None:
            return []
        topics_content: list[dict[str, Any]] = []
        for topic_id in topic_ids:
            topic = self._doc_parser.get_topic(topic_id)
            if topic is None:
                continue
            topics_content.append(
                {
                    "id": topic_id,
                    "title": topic.title,
                    "content": self._doc_parser.get_topic_content(topic_id),
                }
            )
        if not topics_content:
            return []
        # Same learner-state seam questions.py uses: exams must see the
        # learner's weak areas, error patterns, and interview competency gaps
        # instead of generating questions from a blank slate.
        learner_state = build_learner_state_block(
            str(user_id or ""),
            topic_id=(topic_ids[0] if topic_ids else None),
            track=None,
        )
        try:
            response = await self._question_generator.generate_quiz_v2(
                topics_content=topics_content,
                count=max(1, min(int(count), MAX_QUESTIONS)),
                question_types=["mcq"],
                learner_state=learner_state,
            )
        except Exception:
            logger.warning(
                "exam question generation failed user=%s topics=%s",
                user_id,
                topic_ids,
                exc_info=True,
            )
            return []
        questions: list[dict[str, Any]] = []
        for index, item in enumerate(response.questions):
            options = [str(choice.text) for choice in item.choices]
            if not options:
                continue
            questions.append(
                {
                    "question_ref": f"gen:{item.topic_id}:{index}",
                    "type": "mcq",
                    "question_text": str(item.question),
                    "topic_id": str(item.topic_id or ""),
                    "options": options,
                    "correct_answer": str(item.correct_answer).strip().upper(),
                    "expected_answer": str(getattr(item, "expected_answer", "") or ""),
                    "source": "weak_area",
                }
            )
        return questions

    def _question_from_card(
        self,
        card: dict[str, Any],
        *,
        question_types: list[str],
        deck_cards: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any] | None:
        """Build an exam question from a flashcard.

        MCQ when the deck supplies at least three distractor
        backs; otherwise a written question (if requested).
        """
        front = str(card.get("front") or "").strip()
        back = str(card.get("back") or "").strip()
        if not front or not back:
            return None
        card_id = str(card.get("card_id") or "")
        topic_id = str(card.get("topic_id") or card.get("deck_id") or "")
        ref = f"card:{card_id}"

        if "mcq" in question_types:
            distractors = self._pick_distractors(
                card=card, deck_cards=deck_cards
            )
            if len(distractors) >= 3:
                options = [back, *distractors]
                rng = random.Random(f"mcq-options:{card_id}")
                rng.shuffle(options)
                correct_index = options.index(back)
                return {
                    "question_ref": ref,
                    "type": "mcq",
                    "question_text": front,
                    "topic_id": topic_id,
                    "options": options,
                    "correct_answer": MCQ_LABELS[correct_index],
                    "expected_answer": back,
                    "source": source,
                }
        if "written" in question_types:
            return {
                "question_ref": ref,
                "type": "written",
                "question_text": front,
                "topic_id": topic_id,
                "options": [],
                "correct_answer": None,
                "expected_answer": back,
                "source": source,
            }
        return None

    @staticmethod
    def _pick_distractors(
        *, card: dict[str, Any], deck_cards: list[dict[str, Any]]
    ) -> list[str]:
        """Deterministic distractor backs from the same deck."""
        card_id = str(card.get("card_id") or "")
        correct_back = str(card.get("back") or "").strip()
        candidates = sorted(
            {
                str(other.get("back") or "").strip()
                for other in deck_cards
                if str(other.get("card_id") or "") != card_id
            }
        )
        return [
            candidate
            for candidate in candidates
            if candidate and candidate != correct_back
        ][:3]

    # ── Lifecycle ──────────────────────────────
    async def start_exam(self, *, user_id: str, exam_id: str) -> dict[str, Any]:
        exam = await self._store.run_async(
            self._store.get_exam, user_id=user_id, exam_id=exam_id
        )
        if exam is None:
            raise ExamNotFoundError(exam_id)
        if exam["status"] != "created":
            raise ExamStateError(
                f"Exam is already {exam['status']}; only a created exam can start."
            )
        started = await self._store.run_async(
            self._store.start_exam,
            user_id=user_id,
            exam_id=exam_id,
            started_at=_utc_now(),
        )
        assert started is not None
        return started

    async def get_exam(self, *, user_id: str, exam_id: str) -> dict[str, Any]:
        exam = await self._store.run_async(
            self._store.get_exam, user_id=user_id, exam_id=exam_id
        )
        if exam is None:
            raise ExamNotFoundError(exam_id)
        answers = await self._store.run_async(
            self._store.get_answers, user_id=user_id, exam_id=exam_id
        )
        exam = dict(exam)
        exam["results"] = [
            {
                "question_ref": answer["question_ref"],
                "type": next(
                    (
                        str(q["type"])
                        for q in exam["questions"]
                        if str(q["question_ref"]) == answer["question_ref"]
                    ),
                    "",
                ),
                "is_correct": answer["is_correct"],
                "score": answer["score"],
                "graded_by": answer["graded_by"],
                "rubric": answer["rubric"],
                "feedback": answer["feedback"],
            }
            for answer in answers
        ]
        return exam

    async def list_exams(self, *, user_id: str) -> list[dict[str, Any]]:
        return await self._store.run_async(
            self._store.list_exams, user_id=user_id, limit=50
        )

    # ── Grading ────────────────────────────────
    @staticmethod
    def _normalise_mcq_answer(answer_text: str) -> str:
        """Accept a letter label ("A") or a 0-based index ("0")."""
        text = str(answer_text or "").strip().upper()
        if text.isdigit():
            index = int(text)
            if 0 <= index < len(MCQ_LABELS):
                return MCQ_LABELS[index]
        return text

    def _grade_mcq(
        self, question: dict[str, Any], answer_text: str
    ) -> dict[str, Any]:
        correct = str(question.get("correct_answer") or "").strip().upper()
        submitted = self._normalise_mcq_answer(answer_text)
        is_correct = bool(correct) and submitted == correct
        return {
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "graded_by": "auto",
            "rubric": None,
            "feedback": (
                "Correct."
                if is_correct
                else f"Incorrect. The correct answer is {correct}."
            ),
        }

    def _grading_seed(self, exam_id: str) -> str:
        """Fixed per-exam seed — identical for grade and remark."""
        return f"{GRADING_SEED}:{exam_id}"

    async def _grade_written(
        self,
        *,
        exam_id: str,
        question: dict[str, Any],
        answer_text: str,
        user_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        seed = self._grading_seed(exam_id)
        prompt = written_grading_prompt(
            question_text=str(question.get("question_text") or ""),
            expected_answer=str(question.get("expected_answer") or ""),
            user_answer=str(answer_text or ""),
            seed=seed,
        )
        rubric: dict[str, Any] | None = None
        graded_by = "auto"
        if self._llm_client is not None:
            try:
                result = await self._llm_client.completion(
                    prompt,
                    structured=True,
                    flow="interview_eval",
                    user_identity=user_identity,
                    system=(
                        "You are a rigorous exam grader. Score the "
                        "candidate's answer against the expected answer "
                        "using the rubric. Return only JSON."
                    ),
                )
                if result.get("success"):
                    payload = parse_json_object(result.get("analysis", ""))
                    if payload:
                        rubric = self._normalise_written_rubric(payload)
                        graded_by = "llm"
            except Exception:
                logger.warning(
                    "exam written-answer grading failed exam=%s",
                    exam_id,
                    exc_info=True,
                )
        if rubric is None:
            rubric = self._fallback_written_rubric(
                question=question, answer_text=answer_text
            )
        score = round(float(rubric.get("overall", 0.0)) / 100.0, 4)
        return {
            "is_correct": None,
            "score": score,
            "graded_by": graded_by,
            "rubric": rubric,
            "feedback": str(rubric.get("feedback", "")),
        }

    @staticmethod
    def _normalise_written_rubric(payload: dict[str, Any]) -> dict[str, Any]:
        dimensions = (
            "technical_accuracy",
            "reasoning_depth",
            "communication_clarity",
            "completeness",
        )
        rubric: dict[str, Any] = {}
        for dimension in dimensions:
            try:
                rubric[dimension] = max(
                    0, min(5, int(float(payload.get(dimension, 0))))
                )
            except (TypeError, ValueError):
                rubric[dimension] = 0
        try:
            overall = float(payload.get("overall", 0.0))
        except (TypeError, ValueError):
            overall = 0.0
        if "overall" not in payload:
            overall = (sum(rubric.values()) / len(dimensions)) * 20.0
        rubric["overall"] = round(max(0.0, min(100.0, overall)), 2)
        rubric["feedback"] = str(payload.get("feedback", ""))[:500]
        return rubric

    @staticmethod
    def _fallback_written_rubric(
        *, question: dict[str, Any], answer_text: str
    ) -> dict[str, Any]:
        """Deterministic keyword-coverage grade when the LLM is unavailable."""
        expected = str(question.get("expected_answer") or "").lower()
        answer = str(answer_text or "").lower().strip()
        if not expected or not answer:
            coverage = 0.0
        else:
            keywords = [word for word in expected.split() if len(word) > 3]
            if not keywords:
                coverage = 1.0
            else:
                hits = sum(1 for word in keywords if word in answer)
                coverage = hits / len(keywords)
        dimension = round(coverage * 5)
        return {
            "technical_accuracy": dimension,
            "reasoning_depth": dimension,
            "communication_clarity": 3,
            "completeness": dimension,
            "overall": round(coverage * 100, 2),
            "feedback": "Graded by keyword coverage (LLM unavailable).",
        }

    # ── Submit ─────────────────────────────────
    async def submit_exam(
        self,
        *,
        user_id: str,
        exam_id: str,
        answers: list[dict[str, Any]],
        user_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        exam = await self._store.run_async(
            self._store.get_exam, user_id=user_id, exam_id=exam_id
        )
        if exam is None:
            raise ExamNotFoundError(exam_id)
        if exam["status"] != STATUS_IN_PROGRESS:
            raise ExamStateError(
                f"Exam is {exam['status']}; only an in-progress exam can be submitted."
            )

        now = _utc_now()
        started_at = _from_iso(str(exam["started_at"]))
        elapsed_minutes = (now - started_at).total_seconds() / 60.0
        # Server timestamps are authoritative: a late submit is
        # flagged but still graded.
        time_exceeded = elapsed_minutes > float(exam["duration_minutes"])

        questions = exam["questions"]
        answer_map = {
            str(answer.get("question_ref")): str(answer.get("answer_text") or "")
            for answer in answers
        }

        results: list[dict[str, Any]] = []
        for question in questions:
            answer_text = answer_map.get(str(question["question_ref"]), "")
            if question.get("type") == "mcq":
                result = self._grade_mcq(question, answer_text)
            else:
                result = await self._grade_written(
                    exam_id=exam_id,
                    question=question,
                    answer_text=answer_text,
                    user_identity=user_identity,
                )
            results.append(result)
            await self._store.run_async(
                self._store.record_answer,
                user_id=user_id,
                exam_id=exam_id,
                question_ref=str(question["question_ref"]),
                answer_text=answer_text,
                is_correct=result["is_correct"],
                score=result["score"],
                graded_by=result["graded_by"],
                rubric=result["rubric"],
                feedback=result["feedback"],
                created_at=now,
            )

        score = (
            round(sum(r["score"] for r in results) / len(results), 4)
            if results
            else 0.0
        )
        passed = score >= PASS_THRESHOLD
        completed = await self._store.run_async(
            self._store.complete_exam,
            user_id=user_id,
            exam_id=exam_id,
            completed_at=now,
            score=score,
            passed=passed,
            time_exceeded=time_exceeded,
        )
        assert completed is not None

        if self._gamification_service is not None:
            await self._gamification_service.record_exam_completion_async(
                user_id=user_id,
                exam_id=exam_id,
                passed=passed,
            )

        return {
            "exam_id": exam_id,
            "score": score,
            "passed": passed,
            "time_exceeded": time_exceeded,
            "per_question": [
                {
                    "question_ref": str(question["question_ref"]),
                    "type": str(question.get("type") or ""),
                    "is_correct": result["is_correct"],
                    "score": result["score"],
                    "graded_by": result["graded_by"],
                    "rubric": result["rubric"],
                    "feedback": result["feedback"],
                }
                for question, result in zip(questions, results)
            ],
            "completed_at": _to_iso(now),
        }

    # ── Remark ─────────────────────────────────
    async def remark_exam(
        self,
        *,
        user_id: str,
        exam_id: str,
        user_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Re-grade with an identical rubric and a fixed seed.

        MCQ grading is deterministic by construction; written
        answers are re-graded through the same prompt (fixed
        per-exam seed), so the same exam + answers re-grade
        identically.
        """
        exam = await self._store.run_async(
            self._store.get_exam, user_id=user_id, exam_id=exam_id
        )
        if exam is None:
            raise ExamNotFoundError(exam_id)
        if exam["status"] != STATUS_COMPLETED:
            raise ExamStateError(
                "Only a completed exam can be re-graded (remark)."
            )
        original_score = float(exam["score"] or 0.0)

        stored_answers = await self._store.run_async(
            self._store.get_answers, user_id=user_id, exam_id=exam_id
        )
        stored_by_ref = {
            str(answer["question_ref"]): answer for answer in stored_answers
        }

        questions = exam["questions"]
        new_results: list[dict[str, Any]] = []
        changed: list[dict[str, Any]] = []
        for question in questions:
            ref = str(question["question_ref"])
            stored = stored_by_ref.get(ref)
            answer_text = str(stored["answer_text"]) if stored else ""
            if question.get("type") == "mcq":
                result = self._grade_mcq(question, answer_text)
            else:
                result = await self._grade_written(
                    exam_id=exam_id,
                    question=question,
                    answer_text=answer_text,
                    user_identity=user_identity,
                )
            new_results.append(result)
            score_changed = stored is None or abs(
                float(stored["score"]) - result["score"]
            ) > 1e-9
            correctness_changed = stored is None or (
                stored["is_correct"] != result["is_correct"]
            )
            if score_changed or correctness_changed:
                changed.append(
                    {
                        "question_ref": ref,
                        "type": str(question.get("type") or ""),
                        "is_correct": result["is_correct"],
                        "score": result["score"],
                        "graded_by": result["graded_by"],
                        "rubric": result["rubric"],
                        "feedback": result["feedback"],
                    }
                )
            await self._store.run_async(
                self._store.record_answer,
                user_id=user_id,
                exam_id=exam_id,
                question_ref=ref,
                answer_text=answer_text,
                is_correct=result["is_correct"],
                score=result["score"],
                graded_by=result["graded_by"],
                rubric=result["rubric"],
                feedback=result["feedback"],
            )

        new_score = (
            round(sum(r["score"] for r in new_results) / len(new_results), 4)
            if new_results
            else 0.0
        )
        new_passed = new_score >= PASS_THRESHOLD
        await self._store.run_async(
            self._store.update_score,
            user_id=user_id,
            exam_id=exam_id,
            score=new_score,
            passed=new_passed,
        )

        return {
            "exam_id": exam_id,
            "original_score": original_score,
            "new_score": new_score,
            "diff": round(new_score - original_score, 4),
            "changed": changed,
        }
