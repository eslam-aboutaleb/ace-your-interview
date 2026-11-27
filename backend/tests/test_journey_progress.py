"""Store tests for per-question journey progress (issue #5).

Covers the progress semantics the reader depends on: explicit
complete/uncomplete, the viewed/studied/practiced/mastered ladder, mastery
requiring two correct attempts with the latest one correct, distinct-slot
percentages that retries cannot inflate, mutation_id replay idempotency,
and retired objectives leaving the summary denominator without losing
their recorded states.
"""

import os
import tempfile
import unittest

from app.services.journey_store import (
    JourneyNotFoundError,
    JourneyStore,
    JourneyValidationError,
)


def _plan(sections) -> dict:
    return {
        "phases": [
            {
                "title": "Phase 1",
                "description": "",
                "modules": [
                    {
                        "title": heading,
                        "description": "",
                        "objectives": [
                            {"title": title, "kind": "concept", "source_section": heading}
                            for title in bullets
                        ],
                    }
                    for heading, bullets in sections
                ],
            }
        ]
    }


DEFAULT_PLAN = _plan(
    [
        ("HTTP Semantics", ["GET method", "POST method"]),
        ("Caching", ["ETag", "Cache-Control"]),
    ]
)


class _Store:
    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self.store = JourneyStore(os.path.join(self._td.name, "learning.db"))
        return self.store

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _create_with_slots(store, *, user_id="alice", plan=None):
    journey = store.create_journey(
        user_id=user_id,
        title="Backend journey",
        goal="ace the interview",
        source_type="static_topic",
        source_topic_id="01-backend-fundamentals-and-http",
        track="backend",
        level="junior",
        plan=plan if plan is not None else DEFAULT_PLAN,
        provenance={"source": "static_curriculum"},
    )
    batch, created = store.create_batch(
        user_id=user_id, journey_id=journey["id"], idempotency_key="batch-1", slot_count=10
    )
    assert created
    slots = store.get_batch_slots(user_id=user_id, journey_id=journey["id"], batch_id=batch["id"])
    return journey, slots


class ProgressSummaryTests(unittest.TestCase):
    def test_fresh_journey_has_zero_progress_over_all_slots(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            self.assertEqual(len(slots), 4)
            progress = store.get_journey_progress(user_id="alice", journey_id=journey["id"])
            self.assertEqual(progress["total_questions"], 4)
            self.assertEqual(progress["completed"], 0)
            self.assertEqual(progress["percent_complete"], 0.0)
            self.assertEqual(len(progress["states"]), 4)
            for state in progress["states"]:
                self.assertFalse(state["completed"])
                self.assertFalse(state["viewed"])

    def test_completion_counts_distinct_slots_not_events(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            # a second, different mutation completing the SAME slot must not
            # inflate the summary
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m2",
            )
            progress = store.get_journey_progress(user_id="alice", journey_id=journey["id"])
            self.assertEqual(progress["completed"], 1)
            self.assertEqual(progress["percent_complete"], 25.0)

    def test_completion_sets_viewed_and_studied_ladder(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            result = store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slots[0]["id"],
                completed=True, mutation_id="m1",
            )
            state = result["state"]
            self.assertTrue(state["completed"])
            self.assertTrue(state["studied"])
            self.assertTrue(state["viewed"])
            self.assertTrue(state["completed_at"])
            self.assertEqual(state["revision"], 1)

    def test_uncomplete_clears_completed_and_mastered_but_keeps_history(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            result = store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=False, mutation_id="m2",
            )
            state = result["state"]
            self.assertFalse(state["completed"])
            self.assertFalse(state["mastered"])
            self.assertEqual(state["completed_at"], "")
            self.assertTrue(state["viewed"])  # interaction history survives
            progress = store.get_journey_progress(user_id="alice", journey_id=journey["id"])
            self.assertEqual(progress["completed"], 0)

    def test_completion_replay_is_idempotent(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            first = store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            replay = store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            self.assertEqual(replay["state"]["revision"], first["state"]["revision"])
            self.assertEqual(replay["state"]["completed_at"], first["state"]["completed_at"])
            self.assertEqual(replay["summary"]["completed"], 1)

    def test_completion_rejects_foreign_slot_and_foreign_user(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            other = _create_with_slots(store, user_id="alice", plan=_plan([("X", ["Y"])]))[0]
            with self.assertRaises(JourneyValidationError):
                store.set_question_completion(
                    user_id="alice", journey_id=journey["id"], slot_id="slot_nope",
                    completed=True, mutation_id="m1",
                )
            with self.assertRaises(JourneyNotFoundError):
                store.set_question_completion(
                    user_id="bob", journey_id=journey["id"], slot_id=slots[0]["id"],
                    completed=True, mutation_id="m1",
                )
            self.assertNotEqual(other["id"], journey["id"])


class AttemptAndMasteryTests(unittest.TestCase):
    def test_attempt_marks_practiced_and_viewed(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            result = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slots[0]["id"],
                outcome="correct", confidence=4, response_time_ms=900, mutation_id="a1",
            )
            state = result["state"]
            self.assertTrue(state["practiced"])
            self.assertTrue(state["viewed"])
            self.assertFalse(state["mastered"])  # needs completion + 2 correct
            self.assertEqual(result["summary"]["practiced"], 1)

    def test_mastery_ladder_requires_completion_two_correct_and_latest_correct(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="correct", mutation_id="a1",
            )
            store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="correct", mutation_id="a2",
            )
            # two correct but not completed -> not mastered
            state = store.get_journey_progress(user_id="alice", journey_id=journey["id"])["states"][0]
            self.assertFalse(state["mastered"])
            # completing now promotes to mastered
            result = store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            self.assertTrue(result["state"]["mastered"])
            # a fresh incorrect attempt demotes again
            result = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="incorrect", mutation_id="a3",
            )
            self.assertFalse(result["state"]["mastered"])
            # back-to-back correct restores it (latest correct, >= 2 correct total)
            result = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="correct", mutation_id="a4",
            )
            self.assertTrue(result["state"]["mastered"])

    def test_attempt_replay_does_not_double_count(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            first = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="correct", mutation_id="a1",
            )
            replay = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="correct", mutation_id="a1",
            )
            self.assertEqual(replay["state"]["revision"], first["state"]["revision"])
            # only one recorded attempt -> mastery must not trigger from the replay
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            state = store.get_journey_progress(user_id="alice", journey_id=journey["id"])["states"][0]
            self.assertFalse(state["mastered"])

    def test_invalid_outcome_rejected(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            with self.assertRaises(JourneyValidationError):
                store.record_question_attempt(
                    user_id="alice", journey_id=journey["id"], slot_id=slots[0]["id"],
                    outcome="maybe", mutation_id="a1",
                )

    def test_skipped_attempt_does_not_clear_mastery(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            slot = slots[0]["id"]
            for mid in ("a1", "a2"):
                store.record_question_attempt(
                    user_id="alice", journey_id=journey["id"], slot_id=slot,
                    outcome="correct", mutation_id=mid,
                )
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            result = store.record_question_attempt(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                outcome="skipped", mutation_id="a3",
            )
            self.assertTrue(result["state"]["mastered"])


class RetirementInteractionTests(unittest.TestCase):
    def test_retired_objectives_leave_denominator_but_keep_states(self):
        with _Store() as store:
            journey, slots = _create_with_slots(store)
            # slots follow phase/module/objective order; the last one is the
            # "Cache-Control" objective in the "Caching" module
            slot = slots[-1]["id"]
            store.set_question_completion(
                user_id="alice", journey_id=journey["id"], slot_id=slot,
                completed=True, mutation_id="m1",
            )
            # replan drops the trailing objective -> it retires without
            # shifting the positional IDs of the survivors
            store.refresh_plan(
                user_id="alice", journey_id=journey["id"],
                plan=_plan([
                    ("HTTP Semantics", ["GET method", "POST method"]),
                    ("Caching", ["ETag"]),
                ]),
            )
            progress = store.get_journey_progress(user_id="alice", journey_id=journey["id"])
            self.assertEqual(progress["total_questions"], 3)
            self.assertEqual(progress["completed"], 0)
            self.assertEqual(progress["percent_complete"], 0.0)
            # the retired slot's recorded state is still queryable per-slot
            row = store._get_state_row_locked(user_id="alice", journey_id=journey["id"], slot_id=slot)
            self.assertIsNotNone(row)
            self.assertEqual(row["completed"], 1)


if __name__ == "__main__":
    unittest.main()
