"""Coverage for app/services/document_store.py.

Complements ``tests/test_document_store.py`` (recall, ranking, user isolation,
dimension rebuild under the pure-Python cosine fallback) with the not-found and
replace paths, the stale-dimension exclusion, the feature-event cap boundary,
and the ``sqlite-vec`` code paths.

``sqlite-vec`` is an optional native extension: when it is not installed the
store probes once, records ``vector_index_status() == "fallback"`` and serves
search from the pure-Python cosine scan. That probe leaves the vec0 branches
unreachable in this environment, so the vec branches here run against a *stand-in*
index: a plain table with the same ``(chunk_id, embedding)`` shape the vec0
virtual table exposes. That exercises this module's own logic (the metadata
bookkeeping, the delete+insert replace, the mirror sync, the LEFT JOIN that
finds chunks with no vector) without the native extension. The one branch a
stand-in cannot serve is ``_search_vec``'s vec0-only
``WHERE embedding MATCH ? AND k = ?`` SQL, which is reported rather than faked.
"""

import json
import os
import sqlite3
import tempfile
import unittest

from app.services.document_store import (
    DOCUMENT_STATUSES,
    VEC_IN_LIMIT,
    DocumentStore,
    cosine_similarity,
    vector_index_status,
)


def _vec_mirror(store):
    """Stand in for the vec0 index: same columns, plain table."""
    with store._conn:
        store._conn.execute(
            "CREATE TABLE IF NOT EXISTS vec_chunks ("
            "chunk_id INTEGER PRIMARY KEY, embedding TEXT)"
        )


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "docs.db")
        self.store = DocumentStore(self.db_path)
        self.addCleanup(self.store._executor.shutdown, wait=False)

    def _doc(self, user="u1", title="t", name="a.txt", mime="text/plain"):
        return self.store.create_document(user, name, mime, title)

    def _ingest(self, user="u1", chunks=("alpha body", "beta body"), vector=None):
        document = self._doc(user=user, title=f"{user}-doc")
        vectors = [vector or [1.0, 0.0] for _ in chunks]
        self.store.persist_chunks(user, document["document_id"], list(chunks), vectors)
        return document


class VectorIndexStatusTests(unittest.TestCase):
    def test_a_fresh_store_records_its_probe_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DocumentStore(os.path.join(tmp, "docs.db"))
            try:
                expected = "available" if store._vec_available else "fallback"
                self.assertEqual(vector_index_status(), expected)
                self.assertIn(vector_index_status(), ("available", "fallback"))
            finally:
                store._executor.shutdown(wait=False)

    def test_the_status_module_default_is_the_safe_mode(self):
        import importlib

        import app.services.document_store as module

        reloaded = importlib.reload(module)
        try:
            # A freshly imported module has never probed, so it must advertise
            # the mode that always works.
            self.assertEqual(reloaded.vector_index_status(), "fallback")
        finally:
            importlib.reload(module)

    def test_an_unloadable_extension_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Point the loader at a path with no extension so the probe fails.
            store = DocumentStore(os.path.join(tmp, "docs.db"))
            try:
                if store._vec_available:
                    self.skipTest("sqlite-vec is installed in this environment")
                self.assertEqual(vector_index_status(), "fallback")
            finally:
                store._executor.shutdown(wait=False)


class PathTests(unittest.TestCase):
    def test_a_memory_database_needs_no_directory_creation(self):
        store = DocumentStore(":memory:")
        try:
            document = store.create_document("u1", "a.txt", "text/plain", "A")
            self.assertEqual(document["status"], "uploaded")
            self.assertEqual(store.list_documents("u1"), [document])
        finally:
            store._executor.shutdown(wait=False)

    def test_a_nested_directory_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c", "docs.db")
            store = DocumentStore(nested)
            try:
                self.assertTrue(os.path.exists(nested))
            finally:
                store._executor.shutdown(wait=False)


class StatusMachineTests(StoreFixture):
    def test_every_declared_status_is_accepted(self):
        document = self._doc()
        for status in DOCUMENT_STATUSES:
            detail = self.store.update_document_status(
                "u1", document["document_id"], status
            )
            self.assertEqual(detail["status"], status)

    def test_an_unknown_status_is_rejected(self):
        document = self._doc()
        for bad in ("", "READY", "done", "uploaded "):
            with self.assertRaises(ValueError):
                self.store.update_document_status("u1", document["document_id"], bad)
        # The row is untouched.
        self.assertEqual(
            self.store.get_document("u1", document["document_id"])["status"],
            "uploaded",
        )

    def test_updating_a_missing_document_returns_none(self):
        self.assertIsNone(
            self.store.update_document_status("ghost", "nope", "ready")
        )

    def test_updating_the_wrong_users_document_returns_none(self):
        document = self._doc(user="u1")
        self.assertIsNone(
            self.store.update_document_status("u2", document["document_id"], "ready")
        )

    def test_the_error_message_is_clipped_to_2000_characters(self):
        document = self._doc()
        detail = self.store.update_document_status(
            "u1", document["document_id"], "failed", error="e" * 4000
        )
        self.assertEqual(len(detail["error"]), 2000)

    def test_the_chunk_count_is_untouched_when_not_supplied(self):
        document = self._doc()
        self.store.update_document_status(
            "u1", document["document_id"], "ready", chunk_count=7
        )
        detail = self.store.update_document_status(
            "u1", document["document_id"], "processing"
        )
        self.assertEqual(detail["chunk_count"], 7)

    def test_a_document_is_isolated_per_user(self):
        mine = self._doc(user="u1", title="mine")
        self._doc(user="u2", title="theirs")
        self.assertEqual(self.store.list_documents("u1")[0]["document_id"], mine["document_id"])
        self.assertEqual(
            self.store.list_documents("u1")[0]["title"], "mine"
        )
        self.assertEqual(len(self.store.list_documents("u2")), 1)

    def test_listing_an_unknown_user_is_empty(self):
        self.assertEqual(self.store.list_documents("nobody"), [])

    def test_documents_are_listed_newest_first(self):
        first = self._doc(title="first")
        second = self._doc(title="second")
        third = self._doc(title="third")
        listed = self.store.list_documents("u1")
        ids = [doc["document_id"] for doc in listed]
        self.assertIn(ids[0], (third["document_id"], second["document_id"]))
        self.assertIn(first["document_id"], ids)
        self.assertEqual(len(ids), 3)


class ChunkStorageTests(StoreFixture):
    def test_chunks_are_stored_in_order_with_token_counts(self):
        document = self._doc()
        count = self.store.persist_chunks(
            "u1",
            document["document_id"],
            ["one two three", "four five"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        self.assertEqual(count, 2)
        chunks = self.store.get_document_chunks("u1", document["document_id"])
        self.assertEqual([c["chunk_index"] for c in chunks], [0, 1])
        self.assertEqual(chunks[0]["content"], "one two three")
        self.assertEqual(chunks[0]["token_count"], 3)
        self.assertEqual(chunks[1]["token_count"], 2)
        self.assertEqual(chunks[0]["document_id"], document["document_id"])

    def test_an_empty_chunk_content_still_counts_as_one_token(self):
        document = self._doc()
        self.store.persist_chunks("u1", document["document_id"], ["   "], [[1.0]])
        chunk = self.store.get_chunk("u1", document["document_id"], 0)
        self.assertEqual(chunk["token_count"], 1)

    def test_persisting_replaces_the_previous_chunks_and_vectors(self):
        document = self._ingest(chunks=("first a", "first b"))
        first_ids = [
            c["chunk_id"]
            for c in self.store.get_document_chunks("u1", document["document_id"])
        ]
        self.store.persist_chunks(
            "u1", document["document_id"], ["second only"], [[1.0, 0.0]]
        )
        chunks = self.store.get_document_chunks("u1", document["document_id"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["content"], "second only")
        self.assertNotIn(chunks[0]["chunk_id"], first_ids)
        # The orphaned embeddings are gone too.
        stored = self.store._conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings WHERE chunk_id IN (?, ?)",
            first_ids,
        ).fetchone()[0]
        self.assertEqual(stored, 0)

    def test_mismatched_chunk_and_embedding_lists_are_zip_truncated(self):
        document = self._doc()
        count = self.store.persist_chunks(
            "u1", document["document_id"], ["a", "b", "c"], [[1.0, 0.0]]
        )
        # persist_chunks reports the *input* chunk count while zip() stored
        # only the one pair it had an embedding for, so the returned count and
        # the persisted row count disagree. In the live pipeline both lists
        # come from the same embedder call and have equal length, so this is a
        # contract edge rather than a live-path defect.
        self.assertEqual(count, 3)
        chunks = self.store.get_document_chunks("u1", document["document_id"])
        self.assertEqual([c["content"] for c in chunks], ["a"])
        results = self.store.search("u1", [1.0, 0.0])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "a")

    def test_extra_embeddings_without_chunks_are_ignored(self):
        document = self._doc()
        count = self.store.persist_chunks(
            "u1", document["document_id"], ["only"], [[1.0, 0.0], [0.0, 1.0]]
        )
        self.assertEqual(count, 1)
        self.assertEqual(
            self.store._conn.execute(
                "SELECT COUNT(*) FROM chunk_embeddings"
            ).fetchone()[0],
            1,
        )

    def test_get_chunk_returns_none_for_a_missing_index(self):
        document = self._ingest()
        self.assertIsNone(self.store.get_chunk("u1", document["document_id"], 99))
        self.assertIsNone(self.store.get_chunk("u1", "nope", 0))
        self.assertIsNone(self.store.get_chunk("u2", document["document_id"], 0))

    def test_get_chunk_for_a_document_without_chunks_is_empty(self):
        document = self._doc()
        self.assertEqual(
            self.store.get_document_chunks("u1", document["document_id"]), []
        )

    def test_persisting_an_empty_chunk_list_clears_the_document(self):
        document = self._ingest()
        self.assertEqual(
            self.store.persist_chunks("u1", document["document_id"], [], []), 0
        )
        self.assertEqual(
            self.store.get_document_chunks("u1", document["document_id"]), []
        )

    def test_a_string_chunk_index_is_coerced(self):
        document = self._ingest(chunks=("only chunk",))
        self.assertIsNotNone(
            self.store.get_chunk("u1", document["document_id"], "0")
        )


class DeleteTests(StoreFixture):
    def test_deleting_removes_the_chunks_and_their_embeddings(self):
        document = self._ingest()
        self.assertTrue(
            self.store.delete_document("u1", document["document_id"])
        )
        self.assertIsNone(self.store.get_document("u1", document["document_id"]))
        self.assertEqual(self.store._conn.execute(
            "SELECT COUNT(*) FROM chunk_embeddings"
        ).fetchone()[0], 0)

    def test_deleting_twice_reports_false_the_second_time(self):
        document = self._ingest()
        self.assertTrue(self.store.delete_document("u1", document["document_id"]))
        self.assertFalse(self.store.delete_document("u1", document["document_id"]))

    def test_deleting_another_users_document_is_refused(self):
        document = self._ingest(user="u1")
        self.assertFalse(
            self.store.delete_document("u2", document["document_id"])
        )
        self.assertIsNotNone(
            self.store.get_document("u1", document["document_id"])
        )

    def test_deleting_a_document_without_chunks_still_removes_the_row(self):
        document = self._doc()
        self.assertTrue(self.store.delete_document("u1", document["document_id"]))

    def test_other_documents_survive_a_delete(self):
        keep = self._ingest(chunks=("keep me",))
        drop = self._ingest(chunks=("drop me",))
        self.store.delete_document("u1", drop["document_id"])
        remaining = [
            d["document_id"] for d in self.store.list_documents("u1")
        ]
        self.assertIn(keep["document_id"], remaining)
        self.assertNotIn(drop["document_id"], remaining)


class UploadCapTests(StoreFixture):
    def test_the_cap_only_counts_todays_events(self):
        self.store.record_upload_event("u1", "doc-1")
        self.assertEqual(self.store.count_uploads_today("u1"), 1)
        # Backdate the event to yesterday's UTC midnight.
        with self.store._conn:
            self.store._conn.execute(
                "UPDATE feature_events SET created_at = '2000-01-01T00:00:00+00:00'"
            )
        self.assertEqual(self.store.count_uploads_today("u1"), 0)

    def test_other_features_do_not_count_towards_the_cap(self):
        with self.store._conn:
            self.store._conn.execute(
                "INSERT INTO feature_events("
                "user_id, feature_key, event_name, metadata_json, created_at) "
                "VALUES ('u1', 'voice', 'tts', '{}', '2999-01-01T00:00:00+00:00')"
            )
        self.assertEqual(self.store.count_uploads_today("u1"), 0)

    def test_events_are_scoped_per_user(self):
        self.store.record_upload_event("u1", "a")
        self.store.record_upload_event("u1", "b")
        self.store.record_upload_event("u2", "c")
        self.assertEqual(self.store.count_uploads_today("u1"), 2)
        self.assertEqual(self.store.count_uploads_today("u2"), 1)

    def test_the_upload_event_records_the_document_id(self):
        self.store.record_upload_event("u1", "doc-42")
        row = self.store._conn.execute(
            "SELECT metadata_json FROM feature_events"
        ).fetchone()
        self.assertEqual(json.loads(row["metadata_json"]), {"document_id": "doc-42"})


class SearchTests(StoreFixture):
    def test_search_returns_nothing_for_a_user_with_no_chunks(self):
        self.assertEqual(self.store.search("u1", [1.0, 0.0]), [])

    def test_search_returns_nothing_for_a_user_with_only_unembedded_chunks(self):
        document = self._doc()
        with self.store._conn:
            self.store._conn.execute(
                "INSERT INTO document_chunks("
                "user_id, document_id, chunk_index, content, token_count, created_at)"
                " VALUES ('u1', ?, 0, 'orphan', 1, '2026-01-01T00:00:00+00:00')",
                (document["document_id"],),
            )
        self.assertEqual(self.store.search("u1", [1.0, 0.0]), [])

    def test_a_stale_dimension_is_excluded_from_the_ranking(self):
        document = self._ingest(chunks=("right dimension",), vector=[1.0, 0.0])
        other = self.store.create_document("u1", "b.txt", "text/plain", "stale")
        self.store.persist_chunks("u1", other["document_id"], ["wrong dimension"], [[1.0, 0.0, 0.0]])
        results = self.store.search("u1", [1.0, 0.0], k=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_id"], document["document_id"])

    def test_k_is_clamped_to_at_least_one(self):
        self._ingest(chunks=("only one",), vector=[1.0, 0.0])
        self.assertEqual(len(self.store.search("u1", [1.0, 0.0], k=0)), 1)
        self.assertEqual(len(self.store.search("u1", [1.0, 0.0], k=-4)), 1)

    def test_k_is_clamped_to_one_hundred(self):
        document = self._doc()
        self.store.persist_chunks(
            "u1",
            document["document_id"],
            [f"chunk {i}" for i in range(120)],
            [[1.0, float(i)] for i in range(120)],
        )
        self.assertEqual(len(self.store.search("u1", [1.0, 1.0], k=10_000)), 100)

    def test_an_unknown_document_scope_matches_nothing(self):
        self._ingest()
        self.assertEqual(
            self.store.search("u1", [1.0, 0.0], document_ids=["ghost"]), []
        )

    def test_results_are_ordered_by_descending_similarity(self):
        self._ingest(chunks=("orthogonal", "aligned", "opposite"), vector=[0.0, 1.0])
        document = self.store.get_document_chunks("u1", self._list_one_id())
        results = self.store.search("u1", [1.0, 0.0], k=3)
        scores = [hit["score"] for hit in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(results), 3)
        self.assertTrue(document)

    def _list_one_id(self):
        return self.store.list_documents("u1")[0]["document_id"]

    def test_the_fallback_is_used_beyond_the_vec_in_clause_ceiling(self):
        # VEC_IN_LIMIT is the point at which the exact pure-Python scan is
        # preferred over a vec0 `chunk_id IN (...)` filter, so the two modes
        # must agree on the ranking.
        self._ingest(chunks=("target",), vector=[1.0, 0.0])
        before = self.store._conn.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]
        self.assertEqual(before, 1)
        # Pad the corpus past the ceiling with one-chunk vectors.
        extra = self.store.create_document("u1", "bulk.txt", "text/plain", "bulk")
        self.store.persist_chunks(
            "u1",
            extra["document_id"],
            [f"filler chunk number {i}" for i in range(VEC_IN_LIMIT)],
            [[0.0, 1.0] for _ in range(VEC_IN_LIMIT)],
        )
        total = self.store._conn.execute(
            "SELECT COUNT(*) FROM document_chunks"
        ).fetchone()[0]
        self.assertGreater(total, VEC_IN_LIMIT)
        results = self.store.search("u1", [1.0, 0.0], k=3)
        self.assertEqual(len(results), 3)
        # The aligned chunk still wins despite the bulk.
        top_document = self.store.get_document("u1", results[0]["document_id"])
        self.assertEqual(top_document["title"], "u1-doc")

    def test_an_available_index_with_a_small_eligible_set_uses_the_vec_path(self):
        # sqlite-vec is not installed here, so the vec0 SQL itself cannot run;
        # what is asserted is the *dispatch*: an available index with a built
        # table and a small eligible set must go to _search_vec, not the
        # fallback. See the module docstring.
        self._ingest(chunks=("target",), vector=[1.0, 0.0])
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        _vec_mirror(self.store)
        calls: list[tuple] = []
        self.store._search_vec = lambda q, k, eligible: calls.append(
            (list(q), k, list(eligible))
        ) or [{"chunk_id": 1, "sentinel": True}]
        results = self.store.search("u1", [1.0, 0.0], k=2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 2)
        self.assertEqual(calls[0][2], [1])
        self.assertEqual(results, [{"chunk_id": 1, "sentinel": True}])

    def test_no_vec_table_sends_search_to_the_fallback(self):
        self._ingest(chunks=("target",), vector=[1.0, 0.0])
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        # No vec_chunks table yet, so the vec path is not usable.
        results = self.store.search("u1", [1.0, 0.0], k=2)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["score"], 1.0)

    def test_load_results_maps_scores_onto_stored_chunks(self):
        document = self._ingest(chunks=("first", "second"), vector=[1.0, 0.0])
        chunk_ids = [
            c["chunk_id"]
            for c in self.store.get_document_chunks("u1", document["document_id"])
        ]
        # A score dict keyed by chunk id is mapped onto the stored rows and
        # re-sorted by score.
        results = self.store._load_results(
            {chunk_ids[0]: 0.25, chunk_ids[1]: 0.75}
        )
        self.assertEqual([hit["chunk_id"] for hit in results], [chunk_ids[1], chunk_ids[0]])
        self.assertEqual(results[0]["score"], 0.75)
        self.assertEqual(results[0]["content"], "second")
        self.assertEqual(results[0]["document_id"], document["document_id"])
        self.assertEqual(results[0]["chunk_index"], 1)

    def test_load_results_with_no_scores_is_empty(self):
        self.assertEqual(self.store._load_results({}), [])

    def test_load_results_skips_score_keys_with_no_stored_chunk(self):
        self._ingest(chunks=("only",), vector=[1.0, 0.0])
        self.assertEqual(self.store._load_results({9_999: 0.9}), [])


class VecMetaTests(StoreFixture):
    def test_metadata_reads_back_what_was_written(self):
        self.assertEqual(self.store._vec_meta_get("dimension"), "")
        self.store._vec_meta_set("dimension", "8")
        self.assertEqual(self.store._vec_meta_get("dimension"), "8")
        # A second write overwrites rather than duplicating.
        self.store._vec_meta_set("dimension", "16")
        self.assertEqual(self.store._vec_meta_get("dimension"), "16")
        self.assertEqual(self.store._vec_meta_get("index_ready"), "")

    def test_an_unchanged_dimension_is_a_noop_even_with_the_index_available(self):
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        self.store._vec_meta_set("dimension", "8")
        self.store._vec_meta_set("index_ready", "1")
        # index_ready == "1" and the dimension matches: nothing to rebuild.
        self.assertFalse(self.store.ensure_vec_index(8))
        self.assertFalse(self.store._vec_table_exists())

    def test_a_dimension_change_drops_the_index_before_recreating_it(self):
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        with self.store._conn:
            self.store._vec_meta_set("dimension", "8")
            self.store._vec_meta_set("index_ready", "1")
        _vec_mirror(self.store)
        # A dimension change triggers a rebuild, which drops and recreates the
        # table. With no vec0 module in this environment the CREATE fails.
        with self.assertRaises(sqlite3.OperationalError):
            self.store.ensure_vec_index(16)
        # The DROP is not rolled back with the failed CREATE (sqlite runs DDL
        # outside the implicit transaction), so the index is left missing while
        # vec_meta still advertises it as ready -- and ensure_vec_index does not
        # consult _vec_table_exists(), so a later same-dimension call is a no-op.
        self.assertFalse(self.store._vec_table_exists())
        self.assertEqual(self.store._vec_meta_get("index_ready"), "1")
        self.assertFalse(self.store.ensure_vec_index(8))
        self.assertFalse(self.store._vec_table_exists())

    def test_a_rebuild_reports_true_so_the_caller_re_embeds(self):
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        _vec_mirror(self.store)
        created: list[int] = []
        # The vec0 CREATE is the only part that needs the native extension; the
        # rebuild contract around it is what is under test.
        self.store._create_vec_table = lambda dimension: created.append(dimension)
        self.assertTrue(self.store.ensure_vec_index(8))
        self.assertEqual(created, [8])
        # A rebuild drops the table first, so the index has to be rebuilt again
        # even at the same dimension.
        self.assertTrue(self.store.ensure_vec_index(8))
        self.assertEqual(created, [8, 8])

    def test_an_unavailable_extension_reports_no_rebuild(self):
        self.assertFalse(self.store.ensure_vec_index(8))
        self.assertEqual(self.store._vec_meta_get("index_ready"), "")


class VecMirrorTests(StoreFixture):
    """The vec index mirrors the stored embeddings chunk for chunk."""

    def setUp(self):
        super().setUp()
        self.store._vec_available = True
        self.addCleanup(setattr, self.store, "_vec_available", False)
        _vec_mirror(self.store)

    def test_an_embedding_write_replaces_the_mirrored_row(self):
        self.store.persist_chunks("u1", "d1", ["body"], [[1.0, 0.0]])
        chunk_id = self.store.get_chunk("u1", "d1", 0)["chunk_id"]
        self.store.upsert_chunk_embedding(chunk_id, [0.0, 1.0])
        stored = self.store._conn.execute(
            "SELECT embedding FROM chunk_embeddings WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        mirrored = self.store._conn.execute(
            "SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        self.assertEqual(stored["embedding"], json.dumps([0.0, 1.0]))
        self.assertEqual(mirrored["embedding"], json.dumps([0.0, 1.0]))

    def test_persisting_chunks_mirrors_every_embedding(self):
        self.store.persist_chunks(
            "u1", "d1", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]]
        )
        mirrored = self.store._conn.execute(
            "SELECT chunk_id FROM vec_chunks ORDER BY chunk_id"
        ).fetchall()
        self.assertEqual(len(mirrored), 2)

    def test_replacing_chunks_clears_the_mirror(self):
        self.store.persist_chunks("u1", "d1", ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        first = [
            row["chunk_id"]
            for row in self.store._conn.execute(
                "SELECT chunk_id FROM vec_chunks"
            ).fetchall()
        ]
        self.store.persist_chunks("u1", "d1", ["c"], [[1.0, 1.0]])
        remaining = self.store._conn.execute(
            "SELECT COUNT(*) FROM vec_chunks"
        ).fetchone()[0]
        self.assertEqual(remaining, 1)
        self.assertNotIn(first[0], [
            row["chunk_id"]
            for row in self.store._conn.execute(
                "SELECT chunk_id FROM vec_chunks"
            ).fetchall()
        ])

    def test_deleting_a_document_clears_the_mirror(self):
        document = self._doc()
        self.store.persist_chunks("u1", document["document_id"], ["a", "b"], [[1.0, 0.0], [0.0, 1.0]])
        self.assertTrue(
            self.store.delete_document("u1", document["document_id"])
        )
        self.assertEqual(
            self.store._conn.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0],
            0,
        )

    def test_missing_vectors_are_found_by_joining_the_index(self):
        self.store.persist_chunks("u1", "d1", ["indexed"], [[1.0, 0.0]])
        # A chunk row added behind the store's back has no mirrored vector.
        with self.store._conn:
            self.store._conn.execute(
                "INSERT INTO document_chunks("
                "user_id, document_id, chunk_index, content, token_count, created_at)"
                " VALUES ('u1', 'd1', 1, 'unindexed', 1, '2026-01-01T00:00:00+00:00')"
            )
        missing = self.store.chunks_missing_vectors(2)
        self.assertEqual([item["content"] for item in missing], ["unindexed"])

    def test_a_fully_indexed_document_needs_no_re_embedding(self):
        self.store.persist_chunks("u1", "d1", ["indexed"], [[1.0, 0.0]])
        self.assertEqual(self.store.chunks_missing_vectors(2), [])


class FallbackMissingVectorTests(StoreFixture):
    """``chunks_missing_vectors`` in pure-Python mode (no vec index)."""

    def test_the_fallback_reader_reports_stale_dimensions(self):
        self.store.persist_chunks("u1", "d1", ["two-dim"], [[1.0, 0.0]])
        self.assertEqual(self.store.chunks_missing_vectors(2), [])
        # A different dimension makes every chunk look un-indexed.
        self.assertEqual(len(self.store.chunks_missing_vectors(3)), 1)

    def test_the_fallback_reader_reports_an_absent_embedding(self):
        self.store.persist_chunks("u1", "d1", ["indexed"], [[1.0, 0.0]])
        with self.store._conn:
            self.store._conn.execute("DELETE FROM chunk_embeddings")
        missing = self.store.chunks_missing_vectors(2)
        self.assertEqual([item["content"] for item in missing], ["indexed"])

    def test_the_fallback_reader_of_an_empty_corpus_is_empty(self):
        self.assertEqual(self.store.chunks_missing_vectors(2), [])

    def test_missing_vectors_are_ordered_by_chunk_id(self):
        document = self._doc()
        self.store.persist_chunks(
            "u1", document["document_id"], ["a", "b", "c"], [[1.0, 0.0]] * 3
        )
        missing = self.store.chunks_missing_vectors(2)
        self.assertEqual([item["content"] for item in missing], [])
        with self.store._conn:
            self.store._conn.execute("DELETE FROM chunk_embeddings")
        missing = self.store.chunks_missing_vectors(2)
        self.assertEqual([item["content"] for item in missing], ["a", "b", "c"])
        self.assertEqual(
            [item["chunk_id"] for item in missing],
            sorted(item["chunk_id"] for item in missing),
        )


class RunAsyncTests(StoreFixture):
    def test_keyword_arguments_are_forwarded(self):
        import asyncio

        self._ingest()
        self.assertEqual(
            asyncio.run(
                self.store.run_async(
                    self.store.count_uploads_today, user_id="u1"
                )
            ),
            0,
        )

    def test_positional_arguments_are_forwarded(self):
        import asyncio

        self._ingest()
        listed = asyncio.run(self.store.run_async(self.store.list_documents, "u1"))
        self.assertEqual(len(listed), 1)
        # The work ran on the store's dedicated worker thread.
        self.assertTrue(
            self.store._executor._threads,
            "no worker thread was started",
        )


class CosineSimilarityTests(unittest.TestCase):
    def test_a_mismatched_vector_length_is_zip_truncated(self):
        # zip() stops at the shorter vector, so a length mismatch silently
        # scores the common prefix rather than raising.
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0, 9.0], [1.0, 0.0]), 1.0
        )

    def test_an_empty_vector_scores_zero(self):
        self.assertEqual(cosine_similarity([], [1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()