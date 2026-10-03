"""Unit tests for the minimal Anki .apkg reader/writer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile

from app.services.anki_archive import (
    MAX_IMPORT_BYTES,
    AnkiArchiveError,
    extract_media,
    read_apkg,
    write_apkg,
)


def _make_apkg_with_media(media_name: str = "image.png") -> bytes:
    """Build a hand-rolled .apkg with a media manifest."""
    buffer = __import__("io").BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("collection.anki2", b"not-a-real-db")
        archive.writestr(
            "media", json.dumps({"1": media_name})
        )
        archive.writestr(media_name, b"\x89PNG-fake-bytes")
    return buffer.getvalue()


class AnkiArchiveRoundTripTests(unittest.TestCase):
    def test_write_then_read_round_trip(self):
        cards = [
            {
                "card_id": "c1",
                "front": "What is HTTP?",
                "back": "HyperText Transfer Protocol.",
                "tags": ["net", "web"],
                "fsrs": {
                    "state": "learning",
                    "reps": 2,
                    "lapses": 0,
                    "scheduled_days": 3,
                },
            },
            {
                "card_id": "c2",
                "front": "What is DNS?",
                "back": "Domain Name System resolves hostnames.",
                "tags": [],
            },
        ]
        review_logs = [
            {
                "card_id": "c1",
                "rating": "good",
                "state": "learning",
                "review_duration_ms": 1500,
                "scheduled_days": 3,
                "elapsed_days": 0,
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        data = write_apkg("My Deck", cards, review_logs)
        self.assertTrue(data)

        parsed = read_apkg(data)
        self.assertEqual(parsed["errors"], [])
        decks = parsed["decks"]
        self.assertEqual(len(decks), 1)
        self.assertEqual(decks[0]["name"], "My Deck")
        parsed_cards = decks[0]["cards"]
        self.assertEqual(len(parsed_cards), 2)

        first = parsed_cards[0]
        self.assertEqual(first["front"], "What is HTTP?")
        self.assertEqual(first["back"], "HyperText Transfer Protocol.")
        self.assertEqual(first["tags"], ["net", "web"])
        # The revlog entry round-trips through the ease mapping.
        self.assertEqual(len(first["review_logs"]), 1)
        log = first["review_logs"][0]
        self.assertEqual(log["rating"], "good")
        self.assertEqual(log["state"], "learning")
        self.assertEqual(log["review_duration_ms"], 1500)
        self.assertEqual(log["scheduled_days"], 3)

        second = parsed_cards[1]
        self.assertEqual(second["front"], "What is DNS?")
        self.assertEqual(second["tags"], [])
        self.assertEqual(second["review_logs"], [])

    def test_write_apkg_caps_cards(self):
        cards = [
            {
                "front": f"What is card {i}?",
                "back": "A sufficiently long back answer text.",
            }
            for i in range(2500)
        ]
        data = write_apkg("Big", cards)
        parsed = read_apkg(data)
        total = sum(len(d["cards"]) for d in parsed["decks"])
        self.assertLessEqual(total, 2000)

    def test_read_rejects_non_zip(self):
        with self.assertRaises(AnkiArchiveError):
            read_apkg(b"definitely not a zip")

    def test_read_rejects_missing_collection(self):
        buffer = __import__("io").BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("media", "{}")
        with self.assertRaises(AnkiArchiveError):
            read_apkg(buffer.getvalue())

    def test_read_rejects_oversized_file(self):
        # A zip whose declared size exceeds the import cap.
        data = _make_apkg_with_media()
        with self.assertRaises(AnkiArchiveError):
            # Patch the cap check by feeding a huge prefix is not
            # possible for a real zip; instead assert the guard
            # constant is enforced by the reader contract.
            from app.services import anki_archive

            original = anki_archive.MAX_IMPORT_BYTES
            anki_archive.MAX_IMPORT_BYTES = 10
            try:
                anki_archive.read_apkg(data)
            finally:
                anki_archive.MAX_IMPORT_BYTES = original

    def test_read_rejects_zip_bomb_ratio(self):
        from app.services import anki_archive

        data = _make_apkg_with_media()
        original = anki_archive.MAX_COMPRESSION_RATIO
        anki_archive.MAX_COMPRESSION_RATIO = 0
        try:
            with self.assertRaises(AnkiArchiveError):
                anki_archive.read_apkg(data)
        finally:
            anki_archive.MAX_COMPRESSION_RATIO = original

    def test_read_handles_unreadable_collection(self):
        # A valid zip with a collection.anki2 that is not SQLite.
        # The reader degrades gracefully: per-table failures are
        # captured in errors instead of failing the whole archive.
        buffer = __import__("io").BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("collection.anki2", b"garbage-not-sqlite")
            archive.writestr("media", "{}")
        parsed = read_apkg(buffer.getvalue())
        self.assertEqual(parsed["decks"], [])
        self.assertTrue(parsed["errors"])


class ExtractMediaTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.media_dir = os.path.join(self._td.name, "media")

    def test_extract_media_writes_manifest_files(self):
        data = _make_apkg_with_media("image.png")
        written = extract_media(data, self.media_dir)
        self.assertEqual(written, ["image.png"])
        with open(
            os.path.join(self.media_dir, "image.png"), "rb"
        ) as f:
            self.assertEqual(f.read(), b"\x89PNG-fake-bytes")

    def test_extract_media_blocks_path_traversal(self):
        data = _make_apkg_with_media("../escape.png")
        written = extract_media(data, self.media_dir)
        self.assertEqual(written, [])
        self.assertFalse(
            os.path.exists(os.path.join(self._td.name, "escape.png"))
        )

    def test_extract_media_blocks_absolute_and_hidden(self):
        for bad in ("/etc/passwd", ".hidden", "sub/dir.png"):
            buffer = __import__("io").BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("collection.anki2", b"x")
                archive.writestr("media", json.dumps({"1": bad}))
                archive.writestr(bad, b"blob")
            self.assertEqual(extract_media(buffer.getvalue(), self.media_dir), [])

    def test_extract_media_ignores_unmanifested_entries(self):
        buffer = __import__("io").BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("collection.anki2", b"x")
            archive.writestr("media", json.dumps({}))
            archive.writestr("sneaky.png", b"blob")
        self.assertEqual(extract_media(buffer.getvalue(), self.media_dir), [])

    def test_extract_media_rejects_oversized(self):
        from app.services import anki_archive

        data = _make_apkg_with_media()
        original = anki_archive.MAX_IMPORT_BYTES
        anki_archive.MAX_IMPORT_BYTES = 10
        try:
            with self.assertRaises(AnkiArchiveError):
                extract_media(data, self.media_dir)
        finally:
            anki_archive.MAX_IMPORT_BYTES = original


class ImportCapTests(unittest.TestCase):
    def test_cap_constants(self):
        self.assertEqual(MAX_IMPORT_BYTES, 50 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
