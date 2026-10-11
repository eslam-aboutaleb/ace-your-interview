"""Adaptive placement onboarding — seed learner profiles with real evidence.

The planner used to cold-start every new learner on a fixed guess
(``mid``/``senior``, confidence 3/5 on all tracks), so the study plan,
recommendations, and readiness score were weakly grounded. This module
replaces that guess with a short adaptive placement quiz:

* :class:`CatalogQuestionSource` builds deterministic multiple-choice
  questions from the static topic catalog
  (:meth:`DocParser.list_topics`) so the flow always works offline.
* :class:`StrictV2QuestionSource` reuses the existing strict-v2 quiz
  generation (:meth:`QuestionGenerator.generate_quiz_v2`) when an LLM is
  wired, and is the preferred source in production.
* :class:`PlacementService` walks the level ladder
  (``junior -> mid -> senior``), grades every answer deterministically
  (never with an LLM), and seeds the learner profile through the
  planner's ``upsert_profile`` when the walk converges.

Difficulty ladder of the deterministic catalog source: a junior
question is answered from the broad topic map (cross-track
distractors), a mid question needs the right topic among same-track
peers, and a senior question needs the right section inside one topic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import secrets
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar

from app.config import get_settings
from app.services.doc_parser import DocParser

T = TypeVar("T")

logger = logging.getLogger(__name__)

LEVELS: tuple[str, ...] = ("junior", "mid", "senior")
LEVEL_INDEX: dict[str, int] = {level: index for index, level in enumerate(LEVELS)}
MCQ_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

#: Every walk starts in the middle so a single answer moves the learner
#: in either direction.
START_LEVEL = "mid"
#: Answered steps required before a direction reversal ends the walk.
MIN_STEPS = 8
#: Hard cap on answered steps in one placement session.
MAX_STEPS = 12
#: Options per catalog question (matches the onboarding UI layout).
OPTION_COUNT = 4
#: Questions requested per level so the walk can stay on a level.
BANK_FETCH_SIZE = 3
#: Weak objectives surfaced in the placement summary.
WEAK_OBJECTIVE_LIMIT = 5
PLACEMENT_MODE = "quiz"
STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"

#: Catalog sub-section headings that are scaffolding rather than a
#: teachable concept (also seen with a " 02"/" 03" ordinal suffix).
_META_SUFFIXES = (
    "common pitfalls",
    "debugging workflow",
    "design review questions",
    "failure modes",
    "implementation checklist",
    "operational signals",
    "practice exercise",
    "tradeoff analysis",
)


class PlacementStateError(ValueError):
    """The placement session is missing, finished, or out of sync."""


class PlacementAnswerError(PlacementStateError):
    """The submitted answer does not resolve to any question option."""


class PlacementUnavailableError(RuntimeError):
    """No placement question could be produced for the requested track."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return text[:80] or "item"


def _normalise_answer_label(answer: str, option_count: int) -> str:
    """Map a submitted answer onto a choice label.

    Mirrors the deterministic MCQ grading used by the exam service: a
    letter label ("B"), a 0-based index ("1"), or the option text are
    all accepted. Anything else is returned upper-cased so it simply
    never matches a label.
    """
    text = str(answer or "").strip()
    if not text:
        return ""
    upper = text.upper()
    if upper in MCQ_LABELS[: max(option_count, 1)]:
        return upper
    if upper.isdigit():
        index = int(upper)
        if 0 <= index < option_count:
            return MCQ_LABELS[index]
    return ""


def _resolve_option_label(answer: str, options: list[str]) -> str:
    """Resolve a correct answer that is a label, an index, or the text."""
    text = str(answer or "").strip()
    if not text:
        return ""
    label = _normalise_answer_label(text, len(options))
    if label:
        return label
    upper = text.upper()
    for index, option in enumerate(options):
        if str(option).strip().upper() == upper:
            return MCQ_LABELS[index]
    return ""


def grade_answer(question: dict[str, Any], selected_answer: str) -> bool:
    """Deterministic MCQ grading: selected label == correct label.

    No model is ever asked to grade a placement answer, so the same
    submission always produces the same result.
    """
    options = [str(option) for option in (question.get("options") or [])]
    correct = _resolve_option_label(
        str(question.get("correct_answer") or ""), options
    )
    submitted = _resolve_option_label(str(selected_answer or ""), options)
    return bool(correct) and submitted == correct


def adapt_level(level: str, is_correct: bool) -> str:
    """Step the level ladder: correct -> harder, wrong -> easier."""
    index = LEVEL_INDEX.get(str(level), LEVEL_INDEX[START_LEVEL])
    if is_correct:
        index = min(len(LEVELS) - 1, index + 1)
    else:
        index = max(0, index - 1)
    return LEVELS[index]


def _step_direction(answer: dict[str, Any]) -> int:
    return 1 if answer.get("is_correct") else -1


def _boundary_stall(answers: list[dict[str, Any]]) -> bool:
    """True when the walk is stuck on a boundary level.

    ``senior`` has nothing harder and ``junior`` nothing easier, so two
    consecutive answers at the same boundary level with the same
    outcome end the walk: the learner is either above the ladder or
    below it and more questions cannot sharpen the placement.
    """
    if len(answers) < 2:
        return False
    last = answers[-1]
    previous = answers[-2]
    if str(last.get("level") or "") != str(previous.get("level") or ""):
        return False
    if last.get("level") == LEVELS[-1]:
        return bool(last.get("is_correct")) and bool(previous.get("is_correct"))
    if last.get("level") == LEVELS[0]:
        return (not last.get("is_correct")) and (not previous.get("is_correct"))
    return False


def walk_is_complete(answers: list[dict[str, Any]]) -> bool:
    """Decide whether the adaptive walk has converged.

    Three stopping rules, all deterministic:

    1. **Boundary stall** — two consecutive answers at ``senior`` that
       are both correct, or at ``junior`` that are both wrong (see
       :func:`_boundary_stall`).
    2. **Direction reversal** — the last two steps moved the level in
       opposite directions (up then down, or down then up). The walk has
       bracketed the learner's boundary level, so more questions cannot
       sharpen the placement. Only honoured once ``MIN_STEPS`` answers
       exist, otherwise a single oscillation ends the walk too early.
    3. **Hard cap** — ``MAX_STEPS`` answered questions.
    """
    steps = len(answers)
    if steps >= MAX_STEPS:
        return True
    if steps < 2:
        return False
    if _boundary_stall(answers):
        return True
    if steps < MIN_STEPS:
        return False
    return _step_direction(answers[-1]) != _step_direction(answers[-2])


def derive_placement(
    answers: list[dict[str, Any]], *, track: str
) -> dict[str, Any]:
    """Derive starting levels, confidence, and weak objectives.

    The starting level is the highest level the learner answered at
    least twice with a Laplace-smoothed accuracy of 0.5 or better
    (``(correct + 1) / (answered + 2)``), so a single lucky guess cannot
    promote anyone and a level the learner never got right is never
    selected. Levels with no answers are not considered.
    """
    stats: dict[str, list[int]] = {level: [0, 0] for level in LEVELS}
    for answer in answers:
        level = str(answer.get("level") or "")
        if level not in stats:
            continue
        stats[level][1] += 1
        if answer.get("is_correct"):
            stats[level][0] += 1

    derived = LEVELS[0]
    for level in reversed(LEVELS):
        correct, asked = stats[level]
        if asked < 2:
            continue
        if (correct + 1) / (asked + 2) >= 0.5:
            derived = level
            break

    correct, asked = stats[derived]
    strength = round((correct + 1) / (asked + 2), 4)
    confidence = LEVEL_INDEX[derived] + 1
    if asked and strength >= 0.75:
        confidence += 1
    elif asked and strength <= 0.34:
        confidence -= 1
    confidence = max(1, min(5, confidence))

    derived_index = LEVEL_INDEX[derived]
    weak = [
        {
            "topic_id": str(answer.get("topic_id") or ""),
            "topic_title": str(answer.get("topic_title") or ""),
            "objective": str(answer.get("concept") or ""),
            "level": str(answer.get("level") or derived),
        }
        for answer in answers
        if not answer.get("is_correct")
        and LEVEL_INDEX.get(str(answer.get("level") or ""), 0) >= derived_index
    ]
    weak.sort(
        key=lambda item: (
            -LEVEL_INDEX.get(item["level"], 0),
            item["topic_id"],
            item["objective"],
        )
    )

    return {
        "track": track,
        "derived_level": derived,
        "confidence": confidence,
        "per_track_levels": {track: derived},
        "per_track_confidence": {track: confidence},
        "weak_objectives": weak[:WEAK_OBJECTIVE_LIMIT],
        "level_path": [str(answer.get("level") or "") for answer in answers],
        "steps_taken": len(answers),
        "level_stats": {
            level: {"correct": stats[level][0], "answered": stats[level][1]}
            for level in LEVELS
        },
    }


def _topic_structure(topic: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """Return ``(section groups, [(group, concept), ...])`` for a topic."""
    groups: list[str] = []
    concepts: list[tuple[str, str]] = []
    for section in getattr(topic, "sections", None) or []:
        heading = " ".join(str(section.get("heading") or "").split())
        if not heading:
            continue
        group, separator, concept = heading.partition(":")
        group = group.strip()
        concept = concept.strip()
        if not group:
            continue
        if not separator:
            if group not in groups:
                groups.append(group)
            continue
        if not concept or _is_meta_concept(concept):
            continue
        if group not in groups:
            groups.append(group)
        concepts.append((group, concept))
    return groups, concepts


def _is_meta_concept(concept: str) -> bool:
    normalised = re.sub(r"\s+\d+$", "", str(concept or "").strip()).strip().lower()
    return normalised in _META_SUFFIXES


class CatalogQuestionSource:
    """Deterministic placement MCQs derived from the topic catalog."""

    source_name = "catalog"

    def __init__(self, parser: DocParser | None = None):
        self._parser = parser
        self._cache: dict[str, dict[str, Any]] = {}

    # ── Catalog index ────────────────────────────────────────
    def _index(self, track: str) -> dict[str, Any]:
        cached = self._cache.get(track)
        if cached is not None:
            return cached
        parser = self._parser
        if parser is None:
            index: dict[str, Any] = {"topics": [], "peer_titles": [], "other_titles": []}
            self._cache[track] = index
            return index

        topics: list[dict[str, Any]] = []
        for summary in parser.list_topics(track=track):
            # ``list_topics`` ignores an unknown track filter, so match
            # the summary's own track to keep the bank track-pure.
            if summary.track != track:
                continue
            detail = parser.get_topic(summary.id)
            groups, concepts = _topic_structure(detail) if detail else ([], [])
            topics.append(
                {
                    "id": summary.id,
                    "title": summary.title,
                    "groups": groups,
                    "concepts": concepts,
                }
            )

        same_track = parser.list_topics(track=track)
        peer_titles = [
            {"id": summary.id, "title": summary.title}
            for summary in same_track
            if summary.track == track
        ]
        other_titles = [
            {"id": summary.id, "title": summary.title}
            for summary in parser.list_topics()
            if summary.id not in {entry["id"] for entry in topics}
        ]
        index = {
            "topics": topics,
            "peer_titles": peer_titles,
            "other_titles": other_titles,
        }
        self._cache[track] = index
        return index

    # ── Question building ────────────────────────────────────
    def questions(
        self,
        *,
        track: str,
        level: str,
        limit: int,
        exclude: set[str] | None = None,
        user_identity: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build up to ``limit`` catalog questions at ``level``."""
        excluded = set(exclude or ())
        index = self._index(track)
        built: list[dict[str, Any]] = []
        for topic in index["topics"]:
            for group, concept in topic["concepts"]:
                question = self._build(
                    topic=topic, group=group, concept=concept, level=level, index=index
                )
                if question is None:
                    continue
                if question["question_id"] in excluded:
                    continue
                built.append(question)
                if len(built) >= max(1, int(limit)):
                    return built
        return built

    def _build(
        self,
        *,
        topic: dict[str, Any],
        group: str,
        concept: str,
        level: str,
        index: dict[str, Any],
    ) -> dict[str, Any] | None:
        question_id = f"placement:{topic['id']}:{_slug(concept)}:{level}"
        if level == LEVELS[-1]:
            correct = group
            pool = [
                other
                for entry in index["topics"]
                for other in entry["groups"]
                if other.lower() != group.lower()
            ]
            question_text = (
                f'In "{topic["title"]}", which section covers "{concept}"?'
            )
            explanation = (
                f'"{concept}" is a subsection of "{group}" '
                f"in {topic['title']}."
            )
        else:
            correct = topic["title"]
            pool_source = (
                index["peer_titles"] if level == "mid" else index["other_titles"]
            )
            pool = [
                entry["title"]
                for entry in pool_source
                if entry["id"] != topic["id"]
                and entry["title"].lower() != correct.lower()
            ]
            question_text = f'Which curriculum topic covers "{concept}"?'
            explanation = (
                f'"{concept}" belongs to {topic["title"]} '
                f'(section "{group}").'
            )

        options = self._options(correct=correct, pool=pool, question_id=question_id)
        if options is None:
            return None
        return {
            "question_id": question_id,
            "question": question_text,
            "options": options,
            "correct_answer": MCQ_LABELS[options.index(correct)],
            "explanation": explanation,
            "topic_id": topic["id"],
            "topic_title": topic["title"],
            "concept": concept[:120],
            "level": level,
            "source": self.source_name,
        }

    def _options(
        self, *, correct: str, pool: list[str], question_id: str
    ) -> list[str] | None:
        """Deterministic option list with the correct answer placed by seed."""
        unique_pool = list(
            dict.fromkeys(
                text for text in pool if str(text).strip().lower() != correct.lower()
            )
        )
        rng = random.Random(f"placement-options:{question_id}")
        rng.shuffle(unique_pool)
        distractors = unique_pool[: OPTION_COUNT - 1]
        if len(distractors) < OPTION_COUNT - 1:
            return None
        options = [correct, *distractors]
        rng.shuffle(options)
        return options


class StrictV2QuestionSource:
    """Placement MCQs from the existing strict-v2 quiz generator.

    Grading stays deterministic: only the *generation* of the question
    and its choices is delegated to the model that already backs
    ``/api/questions/quiz/generate-v2``.
    """

    source_name = "quiz_v2"

    def __init__(self, generator: Any, parser: DocParser | None = None):
        self._generator = generator
        self._parser = parser

    async def questions(
        self,
        *,
        track: str,
        level: str,
        limit: int,
        exclude: set[str] | None = None,
        user_identity: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        count = max(2, int(limit))
        topics_content, titles = self._topics_content(track)
        if not topics_content:
            return []
        response = await self._generator.generate_quiz_v2(
            topics_content=topics_content,
            count=count,
            question_types=["mcq"],
            level=level,
            user_identity=user_identity,
        )
        out: list[dict[str, Any]] = []
        for item in getattr(response, "questions", None) or []:
            options = [
                str(choice.text).strip() for choice in (getattr(item, "choices", None) or [])
            ]
            options = [option for option in options if option]
            if len(options) < 2:
                continue
            correct = _resolve_option_label(str(item.correct_answer or ""), options)
            if not correct:
                continue
            topic_id = str(getattr(item, "topic_id", "") or "")
            question_text = " ".join(str(getattr(item, "question", "") or "").split())
            if not question_text:
                continue
            objective = str(getattr(item, "learning_objective", "") or "").strip()
            out.append(
                {
                    "question_id": (
                        f"placement:v2:{topic_id or 'topic'}:{_slug(question_text)}"
                    ),
                    "question": question_text,
                    "options": options,
                    "correct_answer": correct,
                    "explanation": str(getattr(item, "explanation", "") or "").strip()[:800],
                    "topic_id": topic_id,
                    "topic_title": titles.get(topic_id, ""),
                    "concept": objective[:120] or question_text[:120],
                    "level": level,
                    "source": self.source_name,
                }
            )
        return out

    def _topics_content(self, track: str) -> tuple[list[dict[str, str]], dict[str, str]]:
        if self._parser is None:
            return [], {}
        titles: dict[str, str] = {}
        topics_content: list[dict[str, str]] = []
        for summary in self._parser.list_topics(track=track)[:2]:
            titles[summary.id] = summary.title
            topics_content.append(
                {
                    "id": summary.id,
                    "title": summary.title,
                    "content": self._parser.get_topic_content(summary.id),
                }
            )
        return topics_content, titles


def public_question(question: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip the answer key before a question leaves the server."""
    if not question:
        return None
    return {
        "question_id": str(question.get("question_id") or ""),
        "question": str(question.get("question") or ""),
        "options": [str(option) for option in (question.get("options") or [])],
        "level": str(question.get("level") or ""),
        "topic_id": str(question.get("topic_id") or ""),
        "topic_title": str(question.get("topic_title") or ""),
        "concept": str(question.get("concept") or ""),
        "source": str(question.get("source") or ""),
    }


class PlacementService:
    """Server-side placement sessions and profile seeding."""

    def __init__(
        self,
        *,
        planner: Any,
        parser: DocParser | None = None,
        question_generator: Any = None,
        gamification_service: Any = None,
        learning_store: Any = None,
        sources: list[Any] | None = None,
    ):
        self._planner = planner
        self._parser = parser
        self._gamification = gamification_service
        self._learning_store = learning_store
        if sources is not None:
            self._sources = list(sources)
        else:
            self._sources = []
            if question_generator is not None:
                self._sources.append(
                    StrictV2QuestionSource(question_generator, parser)
                )
            self._sources.append(CatalogQuestionSource(parser))

        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="placement-service"
        )
        self._conn = sqlite3.connect(
            getattr(planner, "db_path", ":memory:"), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
        """Run synchronous SQLite work on a dedicated worker thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor,
                lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    def _init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS placement_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    track TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_placement_sessions_user
                ON placement_sessions(user_id, updated_at DESC)
                """
            )

    # ── Session persistence ──────────────────────────────────
    def _new_state(
        self,
        *,
        user_id: str,
        track: str,
        target_role: str,
        daily_goal: int,
    ) -> dict[str, Any]:
        now_iso = _iso(_utc_now())
        return {
            "session_id": secrets.token_urlsafe(18),
            "user_id": user_id,
            "track": track,
            "target_role": target_role,
            "daily_goal": daily_goal,
            "level": START_LEVEL,
            "steps": 0,
            "asked": [],
            "answers": [],
            "bank": {},
            "status": STATUS_ACTIVE,
            "completed_at": "",
            "created_at": now_iso,
            "updated_at": now_iso,
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _iso(_utc_now())
        payload = json.dumps(state, ensure_ascii=True)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO placement_sessions(
                    session_id, user_id, track, status, state_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (
                    state["session_id"],
                    state["user_id"],
                    state["track"],
                    state["status"],
                    payload,
                    state["created_at"],
                    state["updated_at"],
                ),
            )

    def _load_state(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT state_json FROM placement_sessions
            WHERE session_id = ? AND user_id = ?
            """,
            (session_id, user_id),
        ).fetchone()
        if row is None:
            return None
        try:
            state = json.loads(str(row["state_json"]))
        except json.JSONDecodeError:
            logger.warning("placement session %s has corrupt state", session_id)
            return None
        return state if isinstance(state, dict) else None

    def get_session(self, *, user_id: str, session_id: str) -> dict[str, Any] | None:
        return self._load_state(user_id=user_id, session_id=session_id)

    def get_completion(self, *, user_id: str) -> dict[str, Any]:
        """Latest completed placement, used to hide the onboarding CTA."""
        row = self._conn.execute(
            """
            SELECT state_json FROM placement_sessions
            WHERE user_id = ? AND status = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (user_id, STATUS_COMPLETE),
        ).fetchone()
        if row is None:
            return {
                "completed": False,
                "completed_at": "",
                "track": "",
                "derived_level": "",
                "steps_taken": 0,
            }
        try:
            state = json.loads(str(row["state_json"]))
        except json.JSONDecodeError:
            state = {}
        answers = state.get("answers") or []
        placement = derive_placement(answers, track=str(state.get("track") or ""))
        return {
            "completed": True,
            "completed_at": str(state.get("completed_at") or ""),
            "track": str(state.get("track") or ""),
            "derived_level": placement["derived_level"],
            "steps_taken": placement["steps_taken"],
        }

    # ── Question resolution ──────────────────────────────────
    async def _fetch_questions(
        self,
        *,
        track: str,
        level: str,
        limit: int,
        exclude: set[str],
        user_identity: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen = set(exclude)
        for source in self._sources:
            if len(collected) >= limit:
                break
            try:
                if asyncio.iscoroutinefunction(getattr(source, "questions", None)):
                    batch = await source.questions(
                        track=track,
                        level=level,
                        limit=limit,
                        exclude=set(seen),
                        user_identity=user_identity,
                    )
                else:
                    batch = source.questions(
                        track=track,
                        level=level,
                        limit=limit,
                        exclude=set(seen),
                    )
            except Exception:
                logger.warning(
                    "placement question source %s failed track=%s level=%s",
                    getattr(source, "source_name", source),
                    track,
                    level,
                    exc_info=True,
                )
                continue
            for question in batch or []:
                question_id = str(question.get("question_id") or "")
                if not question_id or question_id in seen:
                    continue
                seen.add(question_id)
                collected.append(question)
                if len(collected) >= limit:
                    break
        return collected

    async def _question_for_level(
        self,
        state: dict[str, Any],
        level: str,
        *,
        user_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        asked = set(state.get("asked") or [])
        for question in state.get("bank", {}).get(level, []):
            if question.get("question_id") not in asked:
                return question
        fetched = await self._fetch_questions(
            track=str(state["track"]),
            level=level,
            limit=BANK_FETCH_SIZE,
            exclude=asked | {str(q.get("question_id")) for q in state.get("bank", {}).get(level, [])},
            user_identity=user_identity,
        )
        if not fetched:
            return None
        state.setdefault("bank", {}).setdefault(level, []).extend(fetched)
        for question in fetched:
            if question.get("question_id") not in asked:
                return question
        return None

    async def _next_question(
        self, state: dict[str, Any], *, user_identity: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        level = str(state.get("level") or START_LEVEL)
        question = await self._question_for_level(
            state, level, user_identity=user_identity
        )
        if question is not None:
            return question
        # Bank exhaustion at the current level: borrow from an adjacent
        # level without moving the walk, so onboarding never dead-ends.
        index = LEVEL_INDEX.get(level, LEVEL_INDEX[START_LEVEL])
        for offset in (-1, 1):
            neighbour = index + offset
            if neighbour < 0 or neighbour >= len(LEVELS):
                continue
            question = await self._question_for_level(
                state, LEVELS[neighbour], user_identity=user_identity
            )
            if question is not None:
                logger.info(
                    "placement borrowed a %s question for session %s (level=%s)",
                    LEVELS[neighbour],
                    state.get("session_id"),
                    level,
                )
                return question
        return None

    # ── Public flow ──────────────────────────────────────────
    async def start_session(
        self,
        *,
        user_id: str,
        track: str,
        target_role: str = "",
        daily_goal: int = 3,
        user_identity: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._new_state(
            user_id=user_id,
            track=track,
            target_role=str(target_role or "").strip()[:200],
            daily_goal=max(1, min(50, int(daily_goal or 3))),
        )
        question = await self._next_question(state, user_identity=user_identity)
        if question is None:
            raise PlacementUnavailableError(
                f"No placement questions are available for track '{track}'."
            )
        self._save_state(state)
        return {
            "session_id": state["session_id"],
            "track": state["track"],
            "status": state["status"],
            "level": state["level"],
            "step_index": state["steps"],
            "max_steps": MAX_STEPS,
            "question": public_question(question),
        }

    async def answer(
        self,
        *,
        user_id: str,
        session_id: str,
        question_id: str,
        selected_answer: str,
        user_identity: dict[str, Any] | None = None,
        response_time_ms: int = 0,
    ) -> dict[str, Any]:
        state = self.get_session(user_id=user_id, session_id=session_id)
        if state is None:
            raise PlacementStateError("Placement session not found")
        if state.get("status") == STATUS_COMPLETE:
            raise PlacementStateError("Placement session already complete")

        question = self._find_question(state, question_id)
        if question is None:
            raise PlacementStateError(
                "Question is not part of this placement session"
            )

        options = [str(option) for option in (question.get("options") or [])]
        if not _resolve_option_label(str(selected_answer or ""), options):
            raise PlacementAnswerError(
                "selected_answer must be an option label (A-D), its index, or its text"
            )

        is_correct = grade_answer(question, selected_answer)
        state.setdefault("asked", []).append(question_id)
        state.setdefault("answers", []).append(
            {
                "question_id": question_id,
                "topic_id": str(question.get("topic_id") or ""),
                "topic_title": str(question.get("topic_title") or ""),
                "concept": str(question.get("concept") or ""),
                "level": str(question.get("level") or state["level"]),
                "selected_answer": str(selected_answer or ""),
                "correct_answer": str(question.get("correct_answer") or ""),
                "is_correct": is_correct,
            }
        )
        state["level"] = adapt_level(state["level"], is_correct)
        state["steps"] = len(state["answers"])

        await self._record_attempt(
            state=state,
            question=question,
            is_correct=is_correct,
            selected_answer=selected_answer,
            response_time_ms=response_time_ms,
        )

        complete = walk_is_complete(state["answers"])
        next_question: dict[str, Any] | None = None
        if complete:
            state["status"] = STATUS_COMPLETE
        else:
            next_question = await self._next_question(
                state, user_identity=user_identity
            )
            if next_question is None:
                state["status"] = STATUS_COMPLETE
                complete = True
        if state["status"] == STATUS_COMPLETE and not state.get("completed_at"):
            state["completed_at"] = _iso(_utc_now())
        self._save_state(state)

        return {
            "session_id": state["session_id"],
            "status": state["status"],
            "level": state["level"],
            "step_index": state["steps"],
            "max_steps": MAX_STEPS,
            "is_correct": is_correct,
            "correct_answer": str(question.get("correct_answer") or ""),
            "explanation": str(question.get("explanation") or ""),
            "complete": complete,
            "question": public_question(next_question),
        }

    async def finish(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        """Seed the learner profile from the walked path (idempotent)."""
        state = self.get_session(user_id=user_id, session_id=session_id)
        if state is None:
            raise PlacementStateError("Placement session not found")
        answers = list(state.get("answers") or [])
        if not answers:
            raise PlacementStateError(
                "Placement session has no answers yet; finish is not possible"
            )

        placement = derive_placement(answers, track=str(state["track"]))
        profile = await self._seed_profile(state, placement)
        if state.get("status") != STATUS_COMPLETE:
            state["status"] = STATUS_COMPLETE
            state["level"] = placement["derived_level"]
            if not state.get("completed_at"):
                state["completed_at"] = _iso(_utc_now())
            self._save_state(state)

        summary = {
            "session_id": state["session_id"],
            "track": state["track"],
            "derived_level": placement["derived_level"],
            "per_track_levels": placement["per_track_levels"],
            "steps_taken": placement["steps_taken"],
            "weak_objectives": placement["weak_objectives"],
            "level_path": placement["level_path"],
            "level_stats": placement["level_stats"],
            "profile": profile,
        }
        return summary

    # ── Internals ────────────────────────────────────────────
    @staticmethod
    def _find_question(
        state: dict[str, Any], question_id: str
    ) -> dict[str, Any] | None:
        wanted = str(question_id or "")
        for questions in (state.get("bank") or {}).values():
            for question in questions or []:
                if str(question.get("question_id") or "") == wanted:
                    if wanted in (state.get("asked") or []):
                        return None
                    return question
        return None

    async def _record_attempt(
        self,
        *,
        state: dict[str, Any],
        question: dict[str, Any],
        is_correct: bool,
        selected_answer: str,
        response_time_ms: int,
    ) -> None:
        """Mirror placement answers into the learning evidence store.

        The profile seed is the contract; these attempts additionally
        give the planner/diagnostic real per-topic evidence. A failure
        here must never break onboarding.
        """
        if self._learning_store is None:
            return
        topic_id = str(question.get("topic_id") or "")
        if not topic_id:
            return
        confidence = LEVEL_INDEX.get(str(question.get("level") or ""), 1) + 1
        try:
            await self._learning_store.run_async(
                self._learning_store.record_attempt,
                user_id=state["user_id"],
                question_id=str(question.get("question_id") or ""),
                topic_id=topic_id,
                user_answer=str(selected_answer or ""),
                is_correct=bool(is_correct),
                confidence=max(1, min(5, confidence)),
                response_time_ms=max(0, int(response_time_ms or 0)),
                mode=PLACEMENT_MODE,
                learning_objective=str(question.get("concept") or "")[:240],
            )
        except Exception:
            logger.warning(
                "placement attempt recording failed user=%s question=%s",
                state.get("user_id"),
                question.get("question_id"),
                exc_info=True,
            )

    async def _seed_profile(
        self, state: dict[str, Any], placement: dict[str, Any]
    ) -> dict[str, Any]:
        """Write the derived levels into the planner's learner profile."""
        track = str(state["track"])
        derived = placement["derived_level"]
        confidence = int(placement["confidence"])
        daily_goal = int(state.get("daily_goal") or 3)

        payload: dict[str, Any] = {
            "primary_track": track,
            "current_level": derived,
            "target_level": LEVELS[
                min(len(LEVELS) - 1, LEVEL_INDEX[derived] + 1)
            ],
            "confidence_by_track": {track: confidence},
            # One daily goal item is budgeted at ~15 minutes, five days
            # a week; a session never exceeds an hour.
            "weekly_minutes": max(30, min(2400, daily_goal * 15 * 5)),
            "preferred_session_minutes": max(10, min(60, daily_goal * 15)),
        }
        target_role = str(state.get("target_role") or "").strip()
        if target_role:
            payload["target_role"] = target_role

        profile = await self._planner.run_async(
            self._planner.upsert_profile,
            user_id=state["user_id"],
            payload=payload,
        )

        await self._seed_daily_goal(user_id=state["user_id"], daily_goal=daily_goal)
        return profile

    async def _seed_daily_goal(self, *, user_id: str, daily_goal: int) -> None:
        if self._gamification is None:
            return
        try:
            if not get_settings().enable_gamification_v1:
                return
            await self._gamification.set_daily_goal(
                user_id=user_id, daily_goal=daily_goal
            )
        except Exception:
            logger.warning(
                "placement daily-goal seeding failed user=%s", user_id, exc_info=True
            )
