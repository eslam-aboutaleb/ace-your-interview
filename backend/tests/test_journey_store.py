"""Tests for app/services/journey_store.py (issue #4).

Covers the durable-journey invariants the reader and generation stages rely
on: stable plan IDs across replans, immutable numbered batches, idempotent
reservations, optimistic checkpoint revisions, mutation replay, retirement
instead of deletion, and per-user isolation.
"""

import os
import tempfile
import unittest

from app.services.journey_store import (
    JourneyNotFoundError,
    JourneyStore,
    JourneyValidationError,
    NoPendingObjectivesError,
    RevisionConflictError,
)


def _plan(sections) -> dict:
    """sections: list of (heading, [objective titles])."""
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
                            {
                                "title": title,
                                "kind": "concept",
                                "source_section": heading,
                            }
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
        ("HTTP Semantics", ["GET method", "POST method", "Idempotency"]),
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


def _create(store, *, user_id="alice", plan=None, title="Backend journey"):
    return store.create_journey(
        user_id=user_id,
        title=title,
        goal="ace the interview",
        source_type="static_topic",
        source_topic_id="01-backend-fundamentals-and-http",
        track="backend",
        level="junior",
        plan=plan if plan is not None else DEFAULT_PLAN,
        provenance={"source": "static_curriculum"},
    )


def _objective_titles(detail):
    return [
        obj["title"]
        for phase in detail["phases"]
        for module in phase["modules"]
        for obj in module["objectives"]
    ]


def _objectives(detail):
    return [
        obj
        for phase in detail["phases"]
        for module in phase["modules"]
        for obj in module["objectives"]
    ]


class CreateAndGetTests(unittest.TestCase):
    def test_create_roundtrip_with_hierarchy(self):
        with _Store() as store:
            detail = _create(store)
            self.assertTrue(detail["id"].startswith("jrn_"))
            self.assertEqual(detail["status"], "active")
            self.assertEqual(detail["plan_version"], 1)
            self.assertEqual(detail["objective_count"], 5)
            self.assertEqual(detail["batch_count"], 0)
            self.assertEqual(
                _objective_titles(detail),
                ["GET method", "POST method", "Idempotency", "ETag", "Cache-Control"],
            )
            for phase in detail["phases"]:
                self.assertTrue(phase["id"].startswith("ph_"))
                for module in phase["modules"]:
                    self.assertTrue(module["id"].startswith("mod_"))
                    for obj in module["objectives"]:
                        self.assertTrue(obj["id"].startswith("obj_"))
                        self.assertEqual(obj["status"], "pending")

    def test_list_isolates_users(self):
        with _Store() as store:
            _create(store, user_id="alice")
            _create(store, user_id="bob", title="Bob journey")
            alice = store.list_journeys(user_id="alice")
            bob = store.list_journeys(user_id="bob")
            self.assertEqual([j["title"] for j in alice], ["Backend journey"])
            self.assertEqual([j["title"] for j in bob], ["Bob journey"])

    def test_get_other_users_journey_raises_not_found(self):
        with _Store() as store:
            detail = _create(store, user_id="alice")
            with self.assertRaises(JourneyNotFoundError):
                store.get_journey(user_id="bob", journey_id=detail["id"])
            with self.assertRaises(JourneyNotFoundError):
                store.get_journey(user_id="alice", journey_id="jrn_missing")


class BatchTests(unittest.TestCase):
    def test_batches_are_numbered_immutably_and_slots_follow_plan_order(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            first, created = store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k1", slot_count=2
            )
            self.assertTrue(created)
            self.assertEqual(first["batch_number"], 1)
            self.assertEqual(first["status"], "planned")
            slots = store.get_batch_slots(
                user_id="alice", journey_id=jid, batch_id=first["id"]
            )
            self.assertEqual(
                [s["objective_title"] for s in slots], ["GET method", "POST method"]
            )
            self.assertEqual([s["position"] for s in slots], [0, 1])
            self.assertTrue(all(s["status"] == "reserved" for s in slots))

            second, created = store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k2", slot_count=2
            )
            self.assertTrue(created)
            self.assertEqual(second["batch_number"], 2)
            slots2 = store.get_batch_slots(
                user_id="alice", journey_id=jid, batch_id=second["id"]
            )
            # Reservation continues with the next unreserved objectives.
            self.assertEqual(
                [s["objective_title"] for s in slots2], ["Idempotency", "ETag"]
            )

    def test_idempotency_key_replay_returns_original_batch(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            first, created = store.create_batch(
                user_id="alice",
                journey_id=jid,
                idempotency_key="same-key",
                slot_count=3,
            )
            self.assertTrue(created)
            replay, created = store.create_batch(
                user_id="alice",
                journey_id=jid,
                idempotency_key="same-key",
                slot_count=3,
            )
            self.assertFalse(created)
            self.assertEqual(replay["id"], first["id"])
            self.assertEqual(replay["batch_number"], first["batch_number"])
            # No extra slots were reserved by the replay.
            self.assertEqual(
                len(store.list_batches(user_id="alice", journey_id=jid)), 1
            )

    def test_no_pending_objectives_raises(self):
        with _Store() as store:
            detail = _create(store, plan=_plan([("Only", ["One objective"])]))
            jid = detail["id"]
            store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k1", slot_count=10
            )
            with self.assertRaises(NoPendingObjectivesError):
                store.create_batch(
                    user_id="alice", journey_id=jid, idempotency_key="k2"
                )

    def test_slot_count_bounds(self):
        with _Store() as store:
            detail = _create(store)
            for bad in (0, 51):
                with self.subTest(bad=bad), self.assertRaises(JourneyValidationError):
                    store.create_batch(
                        user_id="alice",
                        journey_id=detail["id"],
                        idempotency_key=f"bad-{bad}",
                        slot_count=bad,
                    )

    def test_archived_journey_rejects_new_batches(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            store._conn.execute(
                "UPDATE journeys SET status = 'archived' WHERE id = ?", (jid,)
            )
            store._conn.commit()
            with self.assertRaises(JourneyValidationError):
                store.create_batch(
                    user_id="alice", journey_id=jid, idempotency_key="k1"
                )

    def test_batches_require_ownership(self):
        with _Store() as store:
            detail = _create(store, user_id="alice")
            with self.assertRaises(JourneyNotFoundError):
                store.create_batch(
                    user_id="bob", journey_id=detail["id"], idempotency_key="k1"
                )
            with self.assertRaises(JourneyNotFoundError):
                store.list_batches(user_id="bob", journey_id=detail["id"])


class RefreshPlanTests(unittest.TestCase):
    def test_unchanged_plan_keeps_version_and_ids(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            before_ids = [o["id"] for o in _objectives(detail)]
            result = store.refresh_plan(
                user_id="alice", journey_id=jid, plan=DEFAULT_PLAN
            )
            self.assertFalse(result["plan_changed"])
            self.assertEqual(result["added_objectives"], 0)
            self.assertEqual(result["retired_objectives"], 0)
            self.assertEqual(result["journey"]["plan_version"], 1)
            self.assertEqual(
                [o["id"] for o in _objectives(result["journey"])], before_ids
            )

    def test_added_content_bumps_version_and_preserves_existing_ids(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            before_ids = {o["id"] for o in _objectives(detail)}
            upgraded = _plan(
                [
                    ("HTTP Semantics", ["GET method", "POST method", "Idempotency"]),
                    ("Caching", ["ETag", "Cache-Control"]),
                    ("HTTP/2", ["Multiplexing"]),
                ]
            )
            result = store.refresh_plan(user_id="alice", journey_id=jid, plan=upgraded)
            self.assertTrue(result["plan_changed"])
            self.assertEqual(result["added_objectives"], 1)
            self.assertEqual(result["journey"]["plan_version"], 2)
            after = _objectives(result["journey"])
            after_ids = {o["id"] for o in after}
            self.assertTrue(before_ids.issubset(after_ids))
            self.assertIn("Multiplexing", [o["title"] for o in after])
            # New batch reservations only pick up unreserved objectives.
            store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k1", slot_count=5
            )
            batch, _ = store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k2", slot_count=5
            )
            self.assertEqual(batch["plan_version"], 2)

    def test_replan_can_add_a_whole_phase(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            two_phase = {
                "phases": [
                    DEFAULT_PLAN["phases"][0],
                    {
                        "title": "Phase 2",
                        "description": "",
                        "modules": [
                            {
                                "title": "Deep dive",
                                "description": "",
                                "objectives": [
                                    {"title": "TCP slow start", "kind": "concept", "source_section": "Deep dive"}
                                ],
                            }
                        ],
                    },
                ]
            }
            result = store.refresh_plan(user_id="alice", journey_id=jid, plan=two_phase)
            self.assertTrue(result["plan_changed"])
            self.assertEqual(result["added_objectives"], 1)
            titles = [ph["title"] for ph in result["journey"]["phases"]]
            self.assertEqual(titles, ["Phase 1", "Phase 2"])

    def test_removed_content_retires_objectives_without_deleting(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            shrunk = _plan(
                [("HTTP Semantics", ["GET method", "POST method", "Idempotency"])]
            )
            result = store.refresh_plan(user_id="alice", journey_id=jid, plan=shrunk)
            self.assertTrue(result["plan_changed"])
            self.assertEqual(result["retired_objectives"], 2)
            retired = [
                o for o in _objectives(result["journey"]) if o["status"] == "retired"
            ]
            self.assertEqual({o["title"] for o in retired}, {"ETag", "Cache-Control"})
            # Retired objectives no longer count toward the active plan size.
            self.assertEqual(result["journey"]["objective_count"], 3)

            # Content that comes back revives the same objective IDs.
            revived = store.refresh_plan(
                user_id="alice", journey_id=jid, plan=DEFAULT_PLAN
            )
            self.assertTrue(revived["plan_changed"])
            self.assertEqual(revived["retired_objectives"], 0)
            statuses = {
                o["title"]: o["status"] for o in _objectives(revived["journey"])
            }
            self.assertEqual(statuses["ETag"], "pending")

    def test_refresh_requires_ownership(self):
        with _Store() as store:
            detail = _create(store, user_id="alice")
            with self.assertRaises(JourneyNotFoundError):
                store.refresh_plan(
                    user_id="bob", journey_id=detail["id"], plan=DEFAULT_PLAN
                )


class CheckpointTests(unittest.TestCase):
    def _journey_with_batch(self, store):
        detail = _create(store)
        jid = detail["id"]
        batch, _ = store.create_batch(
            user_id="alice", journey_id=jid, idempotency_key="k1", slot_count=2
        )
        slots = store.get_batch_slots(
            user_id="alice", journey_id=jid, batch_id=batch["id"]
        )
        return jid, batch, slots

    def test_first_write_then_partial_merge(self):
        with _Store() as store:
            jid, batch, slots = self._journey_with_batch(store)
            state = store.update_checkpoint(
                user_id="alice",
                journey_id=jid,
                expected_revision=0,
                mutation_id="m1",
                last_batch_id=batch["id"],
                last_question_id=slots[0]["id"],
                scroll_offset=0.25,
            )
            self.assertEqual(state["revision"], 1)
            self.assertEqual(state["last_question_id"], slots[0]["id"])
            self.assertAlmostEqual(state["scroll_offset"], 0.25)

            # A partial update keeps the fields it does not mention.
            state = store.update_checkpoint(
                user_id="alice",
                journey_id=jid,
                expected_revision=1,
                mutation_id="m2",
                last_section_id="sec-9",
            )
            self.assertEqual(state["revision"], 2)
            self.assertEqual(state["last_question_id"], slots[0]["id"])
            self.assertEqual(state["last_section_id"], "sec-9")

    def test_stale_revision_conflicts_and_reports_current_state(self):
        with _Store() as store:
            jid, batch, slots = self._journey_with_batch(store)
            store.update_checkpoint(
                user_id="alice",
                journey_id=jid,
                expected_revision=0,
                mutation_id="m1",
                last_question_id=slots[0]["id"],
            )
            with self.assertRaises(RevisionConflictError) as ctx:
                store.update_checkpoint(
                    user_id="alice",
                    journey_id=jid,
                    expected_revision=0,
                    mutation_id="m2",
                    last_question_id=slots[1]["id"],
                )
            self.assertEqual(ctx.exception.current["revision"], 1)
            self.assertEqual(ctx.exception.current["last_question_id"], slots[0]["id"])
            # Server state was not overwritten by the losing write.
            state = store.get_reader_state(user_id="alice", journey_id=jid)
            self.assertEqual(state["checkpoint"]["last_question_id"], slots[0]["id"])

    def test_mutation_replay_is_idempotent(self):
        with _Store() as store:
            jid, batch, slots = self._journey_with_batch(store)
            first = store.update_checkpoint(
                user_id="alice",
                journey_id=jid,
                expected_revision=0,
                mutation_id="m1",
                last_question_id=slots[0]["id"],
            )
            replay = store.update_checkpoint(
                user_id="alice",
                journey_id=jid,
                expected_revision=0,
                mutation_id="m1",
                last_question_id=slots[1]["id"],
            )
            self.assertEqual(replay["revision"], first["revision"])
            self.assertEqual(replay["last_question_id"], slots[0]["id"])

    def test_references_must_belong_to_the_journey(self):
        with _Store() as store:
            jid, batch, slots = self._journey_with_batch(store)
            other = _create(store, title="Other")
            other_batch, _ = store.create_batch(
                user_id="alice",
                journey_id=other["id"],
                idempotency_key="k9",
                slot_count=1,
            )
            with self.assertRaises(JourneyValidationError):
                store.update_checkpoint(
                    user_id="alice",
                    journey_id=jid,
                    expected_revision=0,
                    mutation_id="m1",
                    last_batch_id=other_batch["id"],
                )
            with self.assertRaises(JourneyValidationError):
                store.update_checkpoint(
                    user_id="alice",
                    journey_id=jid,
                    expected_revision=0,
                    mutation_id="m1",
                    last_question_id="slot_does_not_exist",
                )

    def test_checkpoint_requires_ownership(self):
        with _Store() as store:
            detail = _create(store, user_id="alice")
            with self.assertRaises(JourneyNotFoundError):
                store.update_checkpoint(
                    user_id="bob",
                    journey_id=detail["id"],
                    expected_revision=0,
                    mutation_id="m1",
                )


class ReaderStateTests(unittest.TestCase):
    def test_reader_state_without_batches(self):
        with _Store() as store:
            detail = _create(store)
            state = store.get_reader_state(user_id="alice", journey_id=detail["id"])
            self.assertIsNone(state["latest_batch"])
            self.assertEqual(state["questions"], [])
            self.assertEqual(state["checkpoint"]["revision"], 0)

    def test_reader_state_returns_latest_batch_in_order(self):
        with _Store() as store:
            detail = _create(store)
            jid = detail["id"]
            store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k1", slot_count=2
            )
            second, _ = store.create_batch(
                user_id="alice", journey_id=jid, idempotency_key="k2", slot_count=2
            )
            state = store.get_reader_state(user_id="alice", journey_id=jid)
            self.assertEqual(state["latest_batch"]["id"], second["id"])
            self.assertEqual(
                [q["objective_title"] for q in state["questions"]],
                ["Idempotency", "ETag"],
            )
            self.assertEqual([q["position"] for q in state["questions"]], [0, 1])

    def test_reader_state_requires_ownership(self):
        with _Store() as store:
            detail = _create(store, user_id="alice")
            with self.assertRaises(JourneyNotFoundError):
                store.get_reader_state(user_id="bob", journey_id=detail["id"])


if __name__ == "__main__":
    unittest.main()
