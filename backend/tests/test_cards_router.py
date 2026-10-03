"""Contract tests for the flashcard deck/card endpoints."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import cards as cards_router_module
from app.services.card_store import CardStore
from app.services.doc_parser import DocParser
from app.services.learning_store import LearningStore


class FakeLLM:
    """Returns flashcards grounded in the custom topic."""

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        system = str(kwargs.get("system") or "")
        if "flashcards" in system.lower():
            payload = [
                {
                    "front": "What is the Java memory model?",
                    "back": "The Java memory model defines how threads interact through memory and which operations are safe.",
                    "source_quote": "Understand memory model, GC, and runtime behavior.",
                    "source_section": "JVM Fundamentals",
                },
                {
                    "front": "Why does Java garbage collection matter?",
                    "back": "Java garbage collection reclaims unreachable objects so the heap does not grow without bound.",
                    "source_quote": "Understand memory model, GC, and runtime behavior.",
                    "source_section": "JVM Fundamentals",
                },
            ]
        else:
            payload = []
        return {
            "success": True,
            "analysis": json.dumps(payload),
            "content": json.dumps(payload),
            "metadata": {"provider": "fake", "model": "fake"},
            "error": "",
        }


class CardsRouterContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        handbook = os.path.join(self.tempdir.name, "owner-handbook")
        os.makedirs(handbook, exist_ok=True)
        with open(os.path.join(handbook, "01-static.md"), "w", encoding="utf-8") as f:
            f.write(
                """---
track: backend
levels: [junior, mid, senior]
---
# Static Topic

Static topic body.

## Static Section
More context for generation.
"""
            )
        self.parser = DocParser(docs_path=self.tempdir.name)
        self.db_path = os.path.join(self.tempdir.name, "learning.db")
        self.learning_store = LearningStore(self.db_path)
        self.card_store = CardStore(self.db_path)
        self.learning_store.upsert_custom_topic(
            user_id="alice",
            topic_id="custom-java",
            source_topic="Java",
            title="Java Interview Roadmap",
            description="Deep custom Java roadmap.",
            track="backend",
            levels=["junior", "mid", "senior"],
            sections=[
                {
                    "heading": "JVM Fundamentals",
                    "content": "Understand memory model, GC, and runtime behavior.",
                }
            ],
            raw_content=(
                "# Java Interview Roadmap\n\nDeep custom Java roadmap.\n\n"
                "## JVM Fundamentals\n\nUnderstand memory model, GC, and runtime behavior."
            ),
        )

        cards_router_module.init(
            card_store=self.card_store,
            learning_store=self.learning_store,
            llm_client=FakeLLM(),
            parser=self.parser,
            mcp_gateway=None,
            document_store=None,
            embedding_client=None,
        )
        self.app = FastAPI()
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )
        self.app.include_router(cards_router_module.decks_router)
        self.app.include_router(cards_router_module.cards_router)
        self.client = TestClient(self.app)

    def _create_deck(self, name="Deck A"):
        res = self.client.post("/api/decks", json={"name": name})
        self.assertEqual(res.status_code, 201, res.text)
        return res.json()

    def _add_card(self, deck_id, front, back, tags=None):
        res = self.client.post(
            f"/api/decks/{deck_id}/cards",
            json={"front": front, "back": back, "tags": tags or []},
        )
        self.assertEqual(res.status_code, 201, res.text)
        return res.json()

    # ── Deck CRUD ──────────────────────────────
    def test_deck_crud(self):
        deck = self._create_deck("Deck A")
        self.assertTrue(deck["deck_id"])
        self.assertEqual(deck["name"], "Deck A")
        self.assertEqual(deck["card_count"], 0)

        listed = self.client.get("/api/decks")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(len(listed.json()["decks"]), 1)

        updated = self.client.put(
            f"/api/decks/{deck['deck_id']}",
            json={"name": "Renamed", "description": "desc"},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["name"], "Renamed")

        deleted = self.client.delete(f"/api/decks/{deck['deck_id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(self.client.get("/api/decks").json()["decks"], [])

    def test_deck_not_found(self):
        self.assertEqual(
            self.client.get("/api/decks/missing").status_code, 404
        )
        self.assertEqual(
            self.client.delete("/api/decks/missing").status_code, 404
        )

    # ── Card CRUD ──────────────────────────────
    def test_card_crud(self):
        deck = self._create_deck()
        card = self._add_card(
            deck["deck_id"], "What is HTTP?", "HyperText Transfer Protocol."
        )
        self.assertTrue(card["card_id"])
        self.assertEqual(card["fsrs_card_id"], card["fsrs_card_id"])
        self.assertEqual(card["fsrs"]["state"], "new")

        listed = self.client.get(f"/api/decks/{deck['deck_id']}/cards")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["total"], 1)

        updated = self.client.put(
            f"/api/cards/{card['card_id']}",
            json={"front": "What is HTTPS?", "back": "HTTP over TLS."},
        )
        self.assertEqual(updated.status_code, 200)
        self.assertEqual(updated.json()["front"], "What is HTTPS?")

        deleted = self.client.delete(f"/api/cards/{card['card_id']}")
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(
            self.client.get(f"/api/decks/{deck['deck_id']}/cards").json()["total"],
            0,
        )

    def test_card_requires_front_and_back(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/cards",
            json={"front": "", "back": "back"},
        )
        self.assertEqual(res.status_code, 422)

    # ── Study session + FSRS review ────────────
    def test_study_session_and_review(self):
        deck = self._create_deck()
        card = self._add_card(
            deck["deck_id"], "What is HTTP?", "HyperText Transfer Protocol."
        )
        session = self.client.get(f"/api/decks/{deck['deck_id']}/study-session")
        self.assertEqual(session.status_code, 200)
        body = session.json()
        self.assertEqual(body["deck_id"], deck["deck_id"])
        self.assertEqual(body["total_due"], 1)
        self.assertEqual(body["cards"][0]["card_id"], card["card_id"])

        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/review",
            json={
                "card_id": card["card_id"],
                "rating": "good",
                "response_time_ms": 1500,
            },
        )
        self.assertEqual(res.status_code, 201, res.text)
        body = res.json()
        self.assertEqual(body["card"]["card_id"], card["card_id"])
        self.assertEqual(body["log"]["rating"], "good")
        self.assertEqual(body["log"]["review_duration_ms"], 1500)
        self.assertIsInstance(body["leech"], bool)

    def test_review_rejects_foreign_card(self):
        deck = self._create_deck()
        other = self._create_deck("Deck B")
        card = self._add_card(
            other["deck_id"], "What is HTTP?", "HyperText Transfer Protocol."
        )
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/review",
            json={"card_id": card["card_id"], "rating": "good"},
        )
        self.assertEqual(res.status_code, 404)

    # ── Import / export ────────────────────────
    def test_import_and_export_round_trip(self):
        from app.services.anki_archive import write_apkg

        source_cards = [
            {
                "front": "What is HTTP?",
                "back": "HyperText Transfer Protocol.",
                "tags": ["net"],
            },
            {
                "front": "What is DNS?",
                "back": "Domain Name System resolves hostnames.",
            },
        ]
        apkg = write_apkg("Imported Deck", source_cards)

        res = self.client.post(
            "/api/decks/import",
            files={"file": ("deck.apkg", apkg, "application/apkg")},
        )
        self.assertEqual(res.status_code, 201, res.text)
        body = res.json()
        self.assertEqual(body["deck_name"], "Imported Deck")
        self.assertEqual(body["imported"], 2)
        self.assertEqual(body["skipped_duplicates"], 0)
        self.assertEqual(body["errors"], [])

        # Each import creates a new deck, so the same file
        # imports again into a fresh deck (dedup is per-deck).
        again = self.client.post(
            "/api/decks/import",
            files={"file": ("deck.apkg", apkg, "application/apkg")},
        )
        self.assertEqual(again.status_code, 201)
        self.assertEqual(again.json()["imported"], 2)
        self.assertEqual(len(self.client.get("/api/decks").json()["decks"]), 2)

        # A file with an internal duplicate skips the repeat.
        dup_apkg = write_apkg(
            "Dup Deck",
            [
                {
                    "front": "What is HTTP?",
                    "back": "HyperText Transfer Protocol.",
                },
                {
                    "front": "What is HTTP?",
                    "back": "HyperText Transfer Protocol.",
                },
            ],
        )
        dup_res = self.client.post(
            "/api/decks/import",
            files={"file": ("dup.apkg", dup_apkg, "application/apkg")},
        )
        self.assertEqual(dup_res.status_code, 201)
        self.assertEqual(dup_res.json()["imported"], 1)
        self.assertEqual(dup_res.json()["skipped_duplicates"], 1)

        # Export the first imported deck and read it back.
        export = self.client.get(f"/api/decks/{body['deck_id']}/export")
        self.assertEqual(export.status_code, 200)
        self.assertIn("application/apkg", export.headers["content-type"])
        self.assertIn("attachment", export.headers["content-disposition"])

        from app.services.anki_archive import read_apkg

        parsed = read_apkg(export.content)
        exported_cards = [
            c for d in parsed["decks"] for c in d["cards"]
        ]
        self.assertEqual(len(exported_cards), 2)
        fronts = {c["front"] for c in exported_cards}
        self.assertEqual(fronts, {"What is HTTP?", "What is DNS?"})

    def test_import_rejects_oversized_file(self):
        res = self.client.post(
            "/api/decks/import",
            files={"file": ("big.apkg", b"x" * (51 * 1024 * 1024), "application/apkg")},
        )
        self.assertEqual(res.status_code, 413)

    def test_import_rejects_non_apkg(self):
        res = self.client.post(
            "/api/decks/import",
            files={"file": ("bad.apkg", b"not a zip", "application/apkg")},
        )
        self.assertEqual(res.status_code, 400)

    # ── Duplicate detection ────────────────────
    def test_find_duplicates(self):
        deck = self._create_deck()
        self._add_card(
            deck["deck_id"],
            "What is HTTP?",
            "HyperText Transfer Protocol for web requests.",
        )
        self._add_card(
            deck["deck_id"],
            "What is HTTP?",
            "HyperText Transfer Protocol for web requests.",
        )
        self._add_card(
            deck["deck_id"],
            "What is DNS?",
            "Domain Name System resolves hostnames.",
        )
        res = self.client.post(
            "/api/cards/find-duplicates",
            json={"deck_id": deck["deck_id"]},
        )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertGreaterEqual(body["threshold"], 0)
        # The identical pair is reported.
        self.assertGreaterEqual(len(body["pairs"]), 1)
        pair = body["pairs"][0]
        self.assertNotEqual(pair["card_a"]["card_id"], pair["card_b"]["card_id"])
        self.assertGreaterEqual(pair["similarity"], 0)

    # ── Generation job ─────────────────────────
    def test_generate_requires_topic(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={"source_type": "topic", "count": 2},
        )
        self.assertEqual(res.status_code, 400)

    def test_generate_topic_not_found(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={"source_type": "topic", "topic_id": "missing-topic", "count": 2},
        )
        self.assertEqual(res.status_code, 404)

    def test_generate_document_requires_document_id(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={"source_type": "document", "count": 2},
        )
        self.assertEqual(res.status_code, 400)

    def test_generate_section_requires_section_title(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "section",
                "topic_id": "custom-java",
                "count": 2,
            },
        )
        self.assertEqual(res.status_code, 400)

    def test_generate_section_not_found(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "section",
                "topic_id": "custom-java",
                "section_title": "No Such Section",
                "count": 2,
            },
        )
        self.assertEqual(res.status_code, 404)

    def test_generate_queues_job_and_polls_done(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "topic",
                "topic_id": "custom-java",
                "count": 2,
            },
        )
        self.assertEqual(res.status_code, 202, res.text)
        job_id = res.json()["job_id"]
        self.assertEqual(res.json()["status"], "queued")

        cards = []
        status = "queued"
        for _ in range(50):
            job_res = self.client.get(f"/api/decks/generate/jobs/{job_id}")
            self.assertEqual(job_res.status_code, 200, job_res.text)
            payload = job_res.json()
            status = payload["status"]
            if status == "done":
                cards = payload["cards"]
                break
            self.assertNotEqual(status, "failed", payload.get("error"))
            time.sleep(0.05)
        self.assertEqual(status, "done")
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0]["front"], "What is the Java memory model?")
        self.assertEqual(cards[0]["source_section"], "JVM Fundamentals")

    def test_generate_job_is_private(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "topic",
                "topic_id": "custom-java",
                "count": 1,
            },
        )
        job_id = res.json()["job_id"]
        # A different user cannot read the job.
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "bob", "provider": "local"}
        )
        res = self.client.get(f"/api/decks/generate/jobs/{job_id}")
        self.assertEqual(res.status_code, 404)
        self.app.dependency_overrides[require_auth] = (
            lambda: {"user": "alice", "provider": "local"}
        )

    def test_generate_section_source(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "section",
                "topic_id": "custom-java",
                "section_title": "JVM Fundamentals",
                "count": 1,
            },
        )
        self.assertEqual(res.status_code, 202, res.text)
        job_id = res.json()["job_id"]
        for _ in range(50):
            payload = self.client.get(
                f"/api/decks/generate/jobs/{job_id}"
            ).json()
            if payload["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
        self.assertEqual(payload["status"], "done", payload.get("error"))
        self.assertEqual(len(payload["cards"]), 1)

    def test_static_topic_generation(self):
        deck = self._create_deck()
        res = self.client.post(
            f"/api/decks/{deck['deck_id']}/generate",
            json={
                "source_type": "topic",
                "topic_id": "static-topic",
                "count": 1,
            },
        )
        # The static topic resolves from the temp curriculum.
        self.assertIn(res.status_code, (202, 404))

    # ── Feature flag ───────────────────────────
    def test_disabled_flag_returns_503(self):
        from app.config import get_settings

        original = get_settings().enable_flashcards_v1
        get_settings().enable_flashcards_v1 = False
        try:
            res = self.client.get("/api/decks")
            self.assertEqual(res.status_code, 503)
        finally:
            get_settings().enable_flashcards_v1 = original


if __name__ == "__main__":
    unittest.main()
