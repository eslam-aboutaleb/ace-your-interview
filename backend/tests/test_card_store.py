"""Unit tests for the flashcard CardStore."""

from __future__ import annotations

import os
import tempfile
import unittest

from app.services.card_store import CardStore
from app.services.learning_store import LearningStore


class CardStoreTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.db_path = os.path.join(self._td.name, "learning.db")
        # LearningStore owns the fsrs_cards / fsrs_review_log
        # tables that CardStore links into.
        self.learning = LearningStore(self.db_path)
        self.store = CardStore(self.db_path)

    def test_create_deck_and_list(self):
        deck = self.store.create_deck(
            user_id="u1", name="Deck A", description="first"
        )
        self.assertTrue(deck["deck_id"])
        self.assertEqual(deck["name"], "Deck A")
        self.assertEqual(deck["description"], "first")
        self.assertEqual(deck["card_count"], 0)
        self.assertEqual(deck["due_count"], 0)

        decks = self.store.list_decks(user_id="u1")
        self.assertEqual(len(decks), 1)
        self.assertEqual(decks[0]["deck_id"], deck["deck_id"])

        # Other users are isolated.
        self.assertEqual(self.store.list_decks(user_id="u2"), [])

    def test_get_deck_missing_returns_none(self):
        self.assertIsNone(
            self.store.get_deck(user_id="u1", deck_id="nope")
        )

    def test_update_deck(self):
        deck = self.store.create_deck(user_id="u1", name="Old")
        updated = self.store.update_deck(
            user_id="u1", deck_id=deck["deck_id"],
            name="New", description="desc",
        )
        self.assertEqual(updated["name"], "New")
        self.assertEqual(updated["description"], "desc")
        self.assertIsNone(
            self.store.update_deck(
                user_id="u1", deck_id="missing",
                name="x", description="",
            )
        )

    def test_create_card_links_fsrs_row(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        card = self.store.create_card(
            user_id="u1",
            deck_id=deck["deck_id"],
            front="What is HTTP?",
            back="HyperText Transfer Protocol for web requests.",
            tags=["net", "web"],
        )
        self.assertTrue(card["card_id"])
        self.assertEqual(card["deck_id"], deck["deck_id"])
        self.assertEqual(card["front"], "What is HTTP?")
        self.assertEqual(card["tags"], ["net", "web"])
        self.assertTrue(card["fsrs_card_id"])
        self.assertEqual(card["suspended"], 0)
        # The linked FSRS row exists and is a fresh flashcard.
        fsrs = card["fsrs"]
        self.assertEqual(fsrs["state"], "new")
        self.assertEqual(fsrs["reps"], 0)
        self.assertEqual(fsrs["lapses"], 0)
        self.assertTrue(fsrs["due_at"])

    def test_list_cards_due_only(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        all_cards = self.store.list_cards(
            user_id="u1", deck_id=deck["deck_id"]
        )
        self.assertEqual(len(all_cards), 1)
        # A fresh card is due immediately (due_at is in the past).
        due = self.store.due_cards_for_deck(
            user_id="u1", deck_id=deck["deck_id"]
        )
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["front"], "What is HTTP?")

    def test_update_card_and_delete_card(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        card = self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        updated = self.store.update_card(
            user_id="u1", card_id=card["card_id"],
            front="What is HTTPS?", back="HTTP over TLS.",
            tags=["secure"],
        )
        self.assertEqual(updated["front"], "What is HTTPS?")
        self.assertEqual(updated["tags"], ["secure"])

        self.assertTrue(
            self.store.delete_card(user_id="u1", card_id=card["card_id"])
        )
        self.assertIsNone(
            self.store.get_card(user_id="u1", card_id=card["card_id"])
        )
        self.assertFalse(
            self.store.delete_card(user_id="u1", card_id=card["card_id"])
        )

    def test_delete_deck_cascades_cards_and_fsrs(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        card = self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        self.assertTrue(
            self.store.delete_deck(user_id="u1", deck_id=deck["deck_id"])
        )
        self.assertEqual(
            self.store.list_decks(user_id="u1"), []
        )
        self.assertIsNone(
            self.store.get_card(user_id="u1", card_id=card["card_id"])
        )
        # The linked FSRS row is gone too.
        rows = self.learning._conn.execute(
            "SELECT COUNT(*) FROM fsrs_cards WHERE user_id = ?",
            ("u1",),
        ).fetchone()
        self.assertEqual(rows[0], 0)

    def test_import_cards_dedupes_and_stores_logs(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        result = self.store.import_cards(
            user_id="u1",
            deck_id=deck["deck_id"],
            cards=[
                {"front": "What is HTTP?", "back": "HyperText Transfer Protocol.", "tags": ["net"]},
                {"front": "What is HTTP?", "back": "HyperText Transfer Protocol."},  # duplicate
                {"front": "", "back": "empty front is skipped"},  # invalid
                {"front": "What is DNS?", "back": "Domain Name System resolves hostnames."},
            ],
            review_logs={
                "0": [
                    {
                        "rating": "good",
                        "state": "learning",
                        "review_duration_ms": 1200,
                        "scheduled_days": 1,
                        "elapsed_days": 0,
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            },
        )
        self.assertEqual(len(result["imported"]), 2)
        self.assertEqual(result["skipped_duplicates"], 2)

        cards = self.store.list_cards(user_id="u1", deck_id=deck["deck_id"])
        self.assertEqual(len(cards), 2)
        fronts = {c["front"] for c in cards}
        self.assertEqual(
            fronts, {"What is HTTP?", "What is DNS?"}
        )

        # Review history was stored against the imported FSRS rows.
        fsrs_ids = [c["fsrs_card_id"] for c in cards]
        logs = self.store.review_logs_for_cards(
            user_id="u1", fsrs_card_ids=fsrs_ids
        )
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["rating"], "good")
        self.assertEqual(logs[0]["review_duration_ms"], 1200)
        self.assertEqual(logs[0]["scheduled_days"], 1)

    def test_find_card_by_text(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        found = self.store.find_card_by_text(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        self.assertIsNotNone(found)
        missing = self.store.find_card_by_text(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is DNS?", back="Domain Name System.",
        )
        self.assertIsNone(missing)

    def test_set_card_suspended(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        card = self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
        )
        suspended = self.store.set_card_suspended(
            user_id="u1", card_id=card["card_id"], suspended=True
        )
        self.assertEqual(suspended["suspended"], 1)
        self.assertEqual(suspended["fsrs"]["suspended"], 1)
        # Suspended cards are excluded from due queries.
        due = self.store.due_cards_for_deck(
            user_id="u1", deck_id=deck["deck_id"]
        )
        self.assertEqual(due, [])
        restored = self.store.set_card_suspended(
            user_id="u1", card_id=card["card_id"], suspended=False
        )
        self.assertEqual(restored["suspended"], 0)

    def test_list_card_texts(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        self.store.create_card(
            user_id="u1", deck_id=deck["deck_id"],
            front="What is HTTP?", back="HyperText Transfer Protocol.",
            tags=["net"],
        )
        texts = self.store.list_card_texts(
            user_id="u1", deck_id=deck["deck_id"]
        )
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0]["front"], "What is HTTP?")
        self.assertEqual(texts[0]["tags"], ["net"])
        self.assertEqual(
            self.store.list_card_texts(user_id="u2"), []
        )

    def test_review_logs_for_cards_batches(self):
        deck = self.store.create_deck(user_id="u1", name="D")
        fsrs_ids = []
        for i in range(5):
            card = self.store.create_card(
                user_id="u1", deck_id=deck["deck_id"],
                front=f"What is HTTP {i}?", back="HyperText Transfer Protocol.",
            )
            fsrs_ids.append(card["fsrs_card_id"])
        logs = self.store.review_logs_for_cards(
            user_id="u1", fsrs_card_ids=fsrs_ids
        )
        self.assertEqual(logs, [])
        self.assertEqual(
            self.store.review_logs_for_cards(user_id="u1", fsrs_card_ids=[]),
            [],
        )


if __name__ == "__main__":
    unittest.main()
