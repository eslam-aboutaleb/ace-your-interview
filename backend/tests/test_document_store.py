"""Tests for the document store: chunking, recall, fallback parity."""

import math
import os
import re
import tempfile
import unittest

from app.services.document_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_DOCUMENT,
    split_chunks,
)
from app.services.document_store import (
    DocumentStore,
    cosine_similarity,
    vector_index_status,
)

FIXTURE_CONCEPTS = [
    "kubernetes", "postgresql", "redis", "kafka",
    "elasticsearch", "docker", "terraform", "ansible",
    "jenkins", "grafana", "prometheus", "nginx",
    "haproxy", "memcached", "rabbitmq", "cassandra",
    "mongodb", "neo4j", "spark", "hadoop",
    "flink", "zookeeper", "consul", "vault",
    "nomad", "envoy", "istio", "linkerd",
    "knative", "openfaas", "pulumi", "cloudformation",
    "vagrant", "packer", "chef", "puppet",
    "saltstack", "gitlab", "github", "bitbucket",
    "jira", "confluence", "slack", "discord",
    "notion", "obsidian", "logseq", "roam",
    "zettelkasten", "anki",
]


def fixture_document(concept: str) -> str:
    return (
        f"The {concept} is a widely used technology in "
        f"modern software engineering. {concept} provides "
        f"reliable performance, horizontal scaling and "
        f"operational simplicity. Teams adopt {concept} to "
        f"reduce operational toil and to build resilient "
        f"distributed systems. The {concept} documentation "
        f"covers installation, configuration, monitoring and "
        f"troubleshooting. Production deployments of "
        f"{concept} usually combine replication, backups and "
        f"alerting. When interviewing, candidates should be "
        f"able to explain what {concept} is, why it matters, "
        f"and the main tradeoffs compared with alternatives."
    )


def fixture_query(concept: str) -> str:
    return f"What is {concept} and why is {concept} important?"


def _corpus_vocabulary() -> set[str]:
    words: set[str] = set()
    for concept in FIXTURE_CONCEPTS:
        for text in (fixture_document(concept), fixture_query(concept)):
            words.update(re.findall(r"[a-z0-9]+", text.lower()))
    return words


class FixtureEmbedder:
    """Collision-free bag-of-words embedder over a fixed vocabulary.

    Each vocabulary word owns one dimension, so distinct words
    never collide and cosine rankings are exact and stable
    across float32 (vec0) and float64 (pure-Python) math.
    """

    def __init__(self, vocabulary):
        self._index = {
            word: position
            for position, word in enumerate(sorted(vocabulary))
        }
        self.dimension = len(self._index)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            position = self._index.get(word)
            if position is not None:
                vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector


class ChunkingTests(unittest.TestCase):
    def test_empty_text_yields_no_chunks(self):
        self.assertEqual(split_chunks(""), [])
        self.assertEqual(split_chunks("   \n  "), [])

    def test_short_text_is_single_chunk(self):
        text = "The quick brown fox jumps over the lazy dog."
        chunks = split_chunks(text)
        self.assertEqual(chunks, [text])

    def test_chunks_respect_size_limit(self):
        text = "word " * 2000
        chunks = split_chunks(text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), CHUNK_SIZE)

    def test_consecutive_chunks_overlap(self):
        paragraphs = [
            f"Paragraph {i} discusses topic number {i} in "
            f"great detail with plenty of extra words to "
            f"make the text long enough to span several "
            f"chunks when combined together in sequence."
            for i in range(40)
        ]
        chunks = split_chunks("\n\n".join(paragraphs))
        self.assertGreater(len(chunks), 2)
        overlap_found = 0
        for previous, current in zip(chunks, chunks[1:]):
            tail = previous[-CHUNK_OVERLAP:]
            if current.startswith(tail):
                overlap_found += 1
        self.assertGreaterEqual(overlap_found, len(chunks) - 2)

    def test_chunking_is_deterministic(self):
        text = "The quick brown fox. " * 500
        self.assertEqual(split_chunks(text), split_chunks(text))

    def test_max_chunks_cap(self):
        text = "a" * (CHUNK_SIZE * (MAX_CHUNKS_PER_DOCUMENT + 10))
        chunks = split_chunks(text)
        self.assertLessEqual(len(chunks), MAX_CHUNKS_PER_DOCUMENT)

    def test_hard_split_of_unbreakable_text(self):
        text = "x" * (CHUNK_SIZE * 3)
        chunks = split_chunks(text)
        # Unbreakable text is hard-split; with overlap the
        # chunk count grows slightly beyond the naive split.
        self.assertGreaterEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), CHUNK_SIZE)
        self.assertIn("x" * CHUNK_SIZE, "".join(chunks))


class CosineTests(unittest.TestCase):
    def test_identical_vectors(self):
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [0.0, 1.0]),
            0.0,
        )

    def test_opposite_vectors(self):
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [-1.0, 0.0]),
            -1.0,
        )

    def test_zero_vector(self):
        self.assertEqual(
            cosine_similarity([0.0, 0.0], [1.0, 0.0]),
            0.0,
        )


class DocumentStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._tmp.name, "docs.db")
        self.store = DocumentStore(self.db_path)
        self.embedder = FixtureEmbedder(_corpus_vocabulary())

    def tearDown(self):
        self.store._executor.shutdown(wait=False)
        self._tmp.cleanup()

    def _ingest(self, user_id, concept):
        document = self.store.create_document(
            user_id, f"{concept}.txt", "text/plain", concept
        )
        chunks = split_chunks(fixture_document(concept))
        embeddings = [self.embedder.embed(chunk) for chunk in chunks]
        self.store.persist_chunks(
            user_id, document["document_id"], chunks, embeddings
        )
        return document

    def test_document_status_machine(self):
        document = self.store.create_document(
            "u1", "a.txt", "text/plain", "A"
        )
        self.assertEqual(document["status"], "uploaded")
        self.assertEqual(document["chunk_count"], 0)

        processing = self.store.update_document_status(
            "u1", document["document_id"], "processing"
        )
        self.assertEqual(processing["status"], "processing")

        ready = self.store.update_document_status(
            "u1",
            document["document_id"],
            "ready",
            chunk_count=3,
        )
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["chunk_count"], 3)

        failed = self.store.update_document_status(
            "u1",
            document["document_id"],
            "failed",
            error="boom",
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"], "boom")

        with self.assertRaises(ValueError):
            self.store.update_document_status(
                "u1", document["document_id"], "bogus"
            )

    def test_list_and_get_documents(self):
        first = self._ingest("u1", "kubernetes")
        second = self._ingest("u1", "redis")
        listed = self.store.list_documents("u1")
        self.assertEqual(len(listed), 2)
        self.assertEqual(
            {doc["document_id"] for doc in listed},
            {first["document_id"], second["document_id"]},
        )
        detail = self.store.get_document(
            "u1", first["document_id"]
        )
        self.assertEqual(detail["title"], "kubernetes")
        self.assertIsNone(
            self.store.get_document("u1", "missing")
        )
        self.assertIsNone(
            self.store.get_document("u2", first["document_id"])
        )

    def test_delete_removes_chunks_and_vectors(self):
        document = self._ingest("u1", "kubernetes")
        chunks = self.store.get_document_chunks(
            "u1", document["document_id"]
        )
        self.assertGreater(len(chunks), 0)
        self.assertTrue(
            self.store.delete_document(
                "u1", document["document_id"]
            )
        )
        self.assertEqual(
            self.store.get_document_chunks(
                "u1", document["document_id"]
            ),
            [],
        )
        self.assertEqual(
            self.store.search(
                "u1", self.embedder.embed("kubernetes")
            ),
            [],
        )
        self.assertFalse(
            self.store.delete_document(
                "u1", document["document_id"]
            )
        )

    def test_search_recall_at_5_on_fixture_corpus(self):
        for concept in FIXTURE_CONCEPTS:
            self._ingest("u1", concept)

        hits = 0
        for concept in FIXTURE_CONCEPTS:
            query_embedding = self.embedder.embed(
                fixture_query(concept)
            )
            results = self.store.search(
                "u1", query_embedding, k=5
            )
            result_concepts = {
                self.store.get_document(
                    "u1", hit["document_id"]
                )["title"]
                for hit in results
            }
            if concept in result_concepts:
                hits += 1

        recall = hits / len(FIXTURE_CONCEPTS)
        self.assertGreaterEqual(
            recall,
            0.8,
            f"recall@5 {recall:.2f} below 0.8",
        )

    def test_search_ranks_target_document_first(self):
        for concept in FIXTURE_CONCEPTS[:10]:
            self._ingest("u1", concept)
        results = self.store.search(
            "u1", self.embedder.embed(fixture_query("kafka")), k=5
        )
        self.assertTrue(results)
        top_document = self.store.get_document(
            "u1", results[0]["document_id"]
        )
        self.assertEqual(top_document["title"], "kafka")

    def test_search_scopes_to_given_documents(self):
        kubernetes = self._ingest("u1", "kubernetes")
        redis = self._ingest("u1", "redis")
        results = self.store.search(
            "u1",
            self.embedder.embed(fixture_query("redis")),
            k=5,
            document_ids=[redis["document_id"]],
        )
        self.assertTrue(results)
        for hit in results:
            self.assertEqual(hit["document_id"], redis["document_id"])

        scoped = self.store.search(
            "u1",
            self.embedder.embed(fixture_query("redis")),
            k=5,
            document_ids=[kubernetes["document_id"]],
        )
        # Redis query against the kubernetes document only:
        # ranking is weak but must stay scoped to the
        # requested document.
        for hit in scoped:
            self.assertEqual(
                hit["document_id"], kubernetes["document_id"]
            )

    def test_search_isolates_users(self):
        self._ingest("u1", "kubernetes")
        self._ingest("u2", "redis")
        results = self.store.search(
            "u1", self.embedder.embed(fixture_query("kubernetes"))
        )
        for hit in results:
            document = self.store.get_document(
                "u1", hit["document_id"]
            )
            self.assertEqual(document["title"], "kubernetes")

    def test_fallback_cosine_parity_with_vec_index(self):
        for concept in FIXTURE_CONCEPTS[:10]:
            self._ingest("u1", concept)
        # A weighted multi-concept query: every mentioned
        # concept appears only in its own document, so the
        # top-5 similarities are strictly ordered with wide
        # gaps — identical under float32 (vec0) and float64
        # (pure-Python cosine) arithmetic.
        weighted_parts: list[str] = []
        for concept, weight in [
            ("kafka", 5),
            ("redis", 4),
            ("docker", 3),
            ("terraform", 2),
            ("ansible", 1),
        ]:
            weighted_parts.extend([concept] * weight)
        weighted_query = " ".join(weighted_parts)
        query_embedding = self.embedder.embed(weighted_query)

        vec_results = self.store.search(
            "u1", query_embedding, k=5
        )
        vec_titles = [
            self.store.get_document("u1", hit["document_id"])["title"]
            for hit in vec_results
        ]
        self.assertEqual(
            vec_titles,
            ["kafka", "redis", "docker", "terraform", "ansible"],
        )

        if not self.store._vec_available:
            # sqlite-vec unavailable: the fallback is the only
            # mode, so parity is trivially satisfied.
            return

        self.store._vec_available = False
        try:
            fallback_results = self.store.search(
                "u1", query_embedding, k=5
            )
        finally:
            self.store._vec_available = True

        self.assertEqual(
            [hit["chunk_id"] for hit in vec_results],
            [hit["chunk_id"] for hit in fallback_results],
        )
        self.assertEqual(
            [hit["document_id"] for hit in vec_results],
            [hit["document_id"] for hit in fallback_results],
        )

    def test_fallback_mode_still_ranks_correctly(self):
        for concept in FIXTURE_CONCEPTS[:10]:
            self._ingest("u1", concept)
        self.store._vec_available = False
        try:
            results = self.store.search(
                "u1",
                self.embedder.embed(fixture_query("kafka")),
                k=5,
            )
        finally:
            self.store._vec_available = True
        self.assertTrue(results)
        top_document = self.store.get_document(
            "u1", results[0]["document_id"]
        )
        self.assertEqual(top_document["title"], "kafka")

    def test_dimension_change_rebuilds_index(self):
        document = self._ingest("u1", "kubernetes")
        chunks = self.store.get_document_chunks(
            "u1", document["document_id"]
        )
        self.assertTrue(chunks)

        if self.store._vec_available:
            # A different dimension triggers a rebuild that
            # wipes the stored vectors.
            self.assertTrue(self.store.ensure_vec_index(8))
        # In either mode, all chunks now need re-embedding at
        # the new dimension.
        missing = self.store.chunks_missing_vectors(8)
        self.assertEqual(len(missing), len(chunks))
        for item in missing:
            self.store.upsert_chunk_embedding(
                item["chunk_id"], [0.5] * 8
            )
        self.assertEqual(self.store.chunks_missing_vectors(8), [])

        if self.store._vec_available:
            # Same dimension again is a no-op.
            self.assertFalse(self.store.ensure_vec_index(8))

    def test_daily_cap_events(self):
        self.assertEqual(self.store.count_uploads_today("u1"), 0)
        self.store.record_upload_event("u1", "doc-1")
        self.store.record_upload_event("u1", "doc-2")
        self.assertEqual(self.store.count_uploads_today("u1"), 2)
        self.assertEqual(self.store.count_uploads_today("u2"), 0)

    def test_vector_index_status_reflects_probe(self):
        expected = (
            "available" if self.store._vec_available else "fallback"
        )
        self.assertEqual(vector_index_status(), expected)


if __name__ == "__main__":
    unittest.main()
