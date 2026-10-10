"""Gamification service: XP awarding, streaks, daily goals, badges.

XP sources (plan: ``1790987112911-analytics-gamification-exam-mode.md``):

- ``learning_attempts`` — quiz attempts and interview turns, hooked
  into :meth:`LearningStore.record_attempt` via the attempt listener.
- Card reviews — hooked into :meth:`LearningStore.record_review`
  (deck-card reviews delegate to it, so one hook covers both).
- Exam completion — invoked directly by the exam service.

Streaks are computed in the user's timezone (``tz`` from
``LearnerProfile``); ``last_activity_date`` is stored as a
user-tz date string and the live streak is recomputed on read
so a lapsed streak reads as zero without mutating history.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services.gamification_store import (
    EVENT_ATTEMPT,
    EVENT_EXAM,
    EVENT_REVIEW,
    GamificationStore,
    add_days,
    today_in_timezone,
)
from app.services.learning_planner import LearningPlannerStore

logger = logging.getLogger(__name__)

#: XP awarded per learning attempt.
XP_ATTEMPT = 10
#: Bonus XP when the attempt was correct.
XP_ATTEMPT_CORRECT_BONUS = 5
#: XP awarded per card review.
XP_REVIEW = 5
#: XP awarded for completing an exam.
XP_EXAM_COMPLETE = 25
#: Bonus XP when the exam was passed.
XP_EXAM_PASS_BONUS = 50
#: XP required per level (level = total_xp // LEVEL_XP + 1).
LEVEL_XP = 500

#: Badge rule keys (plan: first review, 7-day streak, 100 cards,
#: first exam, exam passed).
BADGE_FIRST_REVIEW = "first_review"
BADGE_STREAK_7 = "streak_7"
BADGE_CARDS_100 = "cards_100"
BADGE_FIRST_EXAM = "first_exam"
BADGE_EXAM_PASSED = "exam_passed"

#: Reviews (attempts + card reviews) needed for the 100-cards badge.
CARDS_100_THRESHOLD = 100
#: Streak length needed for the 7-day streak badge.
STREAK_7_THRESHOLD = 7


class GamificationService:
    """Award XP, maintain streaks, and evaluate badge rules."""

    def __init__(
        self,
        store: GamificationStore,
        planner: LearningPlannerStore | None = None,
    ):
        self._store = store
        self._planner = planner

    # ── Timezone resolution ──────────────────────
    def user_timezone(self, *, user_id: str) -> str:
        """The user's tz-database name from ``LearnerProfile``."""
        if self._planner is None:
            return "UTC"
        try:
            profile = self._planner.get_profile(user_id=user_id)
            return str(profile.get("timezone") or "UTC")
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "gamification timezone lookup failed user=%s", user_id
            )
            return "UTC"

    # ── Synchronous core (invoked from store hooks) ──
    def record_learning_attempt(
        self,
        *,
        user_id: str,
        attempt_id: int,
        is_correct: bool,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Award XP for a ``learning_attempts`` row.

        Called from the learning-store attempt listener, so it
        runs synchronously on the learning store's worker
        thread. Idempotent via the ``event_key`` UNIQUE
        constraint.
        """
        event_key = f"attempt:{user_id}:{attempt_id}"
        xp = XP_ATTEMPT + (XP_ATTEMPT_CORRECT_BONUS if is_correct else 0)
        state = self._store.award_xp(
            user_id=user_id,
            event_key=event_key,
            event_type=EVENT_ATTEMPT,
            xp=xp,
            timezone=timezone,
        )
        self._evaluate_badges(user_id=user_id, timezone=timezone)
        return state

    def record_card_review(
        self,
        *,
        user_id: str,
        card_id: str,
        review_log_id: int,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Award XP for an FSRS card review.

        ``review_log_id`` (the ``fsrs_review_log`` row id) makes
        the event key unique per review of the same card.
        """
        event_key = f"review:{user_id}:{card_id}:{review_log_id}"
        state = self._store.award_xp(
            user_id=user_id,
            event_key=event_key,
            event_type=EVENT_REVIEW,
            xp=XP_REVIEW,
            timezone=timezone,
        )
        self._evaluate_badges(user_id=user_id, timezone=timezone)
        return state

    def record_exam_completion(
        self,
        *,
        user_id: str,
        exam_id: str,
        passed: bool,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Award XP for completing (and possibly passing) an exam."""
        event_key = f"exam:{user_id}:{exam_id}"
        xp = XP_EXAM_COMPLETE + (XP_EXAM_PASS_BONUS if passed else 0)
        state = self._store.award_xp(
            user_id=user_id,
            event_key=event_key,
            event_type=EVENT_EXAM,
            xp=xp,
            timezone=timezone,
        )
        self._evaluate_badges(
            user_id=user_id, timezone=timezone, exam_passed=passed
        )
        return state

    async def record_exam_completion_async(
        self,
        *,
        user_id: str,
        exam_id: str,
        passed: bool,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Async wrapper for :meth:`record_exam_completion`."""
        effective_tz = timezone or self.user_timezone(user_id=user_id)
        return await self._store.run_async(
            self.record_exam_completion,
            user_id=user_id,
            exam_id=exam_id,
            passed=passed,
            timezone=effective_tz,
        )

    # ── Badges ───────────────────────────────────
    def _evaluate_badges(
        self,
        *,
        user_id: str,
        timezone: str | None = None,
        exam_passed: bool | None = None,
    ) -> None:
        state = self._store.get_state(user_id=user_id)
        badges = set(state["badges"])
        learning_events = self._store.count_events(
            user_id=user_id, event_types=(EVENT_ATTEMPT, EVENT_REVIEW)
        )
        exam_events = self._store.count_events(
            user_id=user_id, event_types=(EVENT_EXAM,)
        )

        if BADGE_FIRST_REVIEW not in badges and learning_events >= 1:
            badges.add(BADGE_FIRST_REVIEW)
        if (
            BADGE_STREAK_7 not in badges
            and state["current_streak"] >= STREAK_7_THRESHOLD
        ):
            badges.add(BADGE_STREAK_7)
        if (
            BADGE_CARDS_100 not in badges
            and learning_events >= CARDS_100_THRESHOLD
        ):
            badges.add(BADGE_CARDS_100)
        if BADGE_FIRST_EXAM not in badges and exam_events >= 1:
            badges.add(BADGE_FIRST_EXAM)
        if (
            BADGE_EXAM_PASSED not in badges
            and (exam_passed or self._store.has_passed_exam(user_id=user_id))
        ):
            badges.add(BADGE_EXAM_PASSED)

        if badges != set(state["badges"]):
            self._store.set_badges(user_id=user_id, badges=sorted(badges))

    # ── Async surface (routers) ──────────────────
    async def get_summary(
        self, *, user_id: str, timezone: str | None = None
    ) -> dict[str, Any]:
        """Gamification summary with the streak recomputed on read."""
        effective_tz = timezone or self.user_timezone(user_id=user_id)
        state = await self._store.run_async(
            self._store.get_state, user_id=user_id
        )
        today = today_in_timezone(effective_tz)
        last_activity = state["last_activity_date"]
        if not last_activity:
            current_streak = 0
        elif last_activity == today or add_days(last_activity, 1) == today:
            # Active today, or yesterday and the day is not over yet.
            current_streak = state["current_streak"]
        else:
            # More than a day of silence: the streak has lapsed.
            current_streak = 0
        today_progress = await self._store.run_async(
            self._store.count_events_on_date,
            user_id=user_id,
            date_str=today,
            timezone=effective_tz,
        )
        total_xp = state["total_xp"]
        return {
            "total_xp": total_xp,
            "level": total_xp // LEVEL_XP + 1,
            "xp_progress": round(
                (total_xp % LEVEL_XP) / LEVEL_XP, 4
            ),
            "current_streak": current_streak,
            "longest_streak": state["longest_streak"],
            "daily_goal": state["daily_goal"],
            "today_progress": today_progress,
            "badges": list(state["badges"]),
        }

    async def set_daily_goal(
        self, *, user_id: str, daily_goal: int
    ) -> dict[str, Any]:
        state = await self._store.run_async(
            self._store.set_daily_goal, user_id=user_id, daily_goal=daily_goal
        )
        return {"daily_goal": state["daily_goal"]}


def register_gamification_hooks(service: GamificationService) -> None:
    """Wire XP awarding into the learning store's event hooks.

    The learning store exposes module-level listener registries;
    registering here keeps the store decoupled from the
    gamification module (no import cycle) while XP awarding
    rides along with every recorded attempt and card review.
    """
    from app.services import learning_store as learning_store_module

    def _on_attempt(**payload: Any) -> None:
        user_id = str(payload["user_id"])
        service.record_learning_attempt(
            user_id=user_id,
            attempt_id=int(payload["attempt_id"]),
            is_correct=bool(payload["is_correct"]),
            timezone=service.user_timezone(user_id=user_id),
        )

    def _on_review(**payload: Any) -> None:
        user_id = str(payload["user_id"])
        service.record_card_review(
            user_id=user_id,
            card_id=str(payload["card_id"]),
            review_log_id=int(payload["review_log_id"]),
            timezone=service.user_timezone(user_id=user_id),
        )

    learning_store_module.register_attempt_listener(_on_attempt)
    learning_store_module.register_review_listener(_on_review)
