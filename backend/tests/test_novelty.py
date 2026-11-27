"""Unit tests for the novelty funnel (issue #2)."""

import unittest

from app.services.novelty import (
    NoveltyChecker,
    SeenQuestion,
    concept_signature,
    cosine,
    jaccard,
    question_fingerprint,
)


def _seen(text: str, *, perspective: str = "", embedding=None) -> tuple[SeenQuestion, str]:
    return (
        SeenQuestion(
            fingerprint=question_fingerprint(text),
            signature=concept_signature(text),
            embedding=embedding,
            perspective=perspective,
        ),
        text,
    )


class PrimitiveTests(unittest.TestCase):
    def test_fingerprint_is_normalisation_stable(self):
        a = question_fingerprint("  What is  HTTP? ")
        b = question_fingerprint("what is http?")
        self.assertEqual(a, b)
        self.assertNotEqual(a, question_fingerprint("what is http/2?"))

    def test_signature_stems_and_drops_stopwords(self):
        sig = concept_signature("What are the differences between caches and caching proxies?")
        self.assertIn("differenc", sig)
        self.assertIn("cach", sig)
        self.assertIn("proxy", sig)
        self.assertNotIn("the", sig)
        self.assertNotIn("what", sig)

    def test_jaccard_and_cosine(self):
        self.assertAlmostEqual(jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})), 1 / 3)
        self.assertEqual(jaccard(frozenset(), frozenset({"a"})), 0.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine([1.0], [1.0, 2.0]), 0.0)

    def test_checker_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            NoveltyChecker(embedding_review_threshold=0.95, embedding_threshold=0.9)


class ExactAndCharStageTests(unittest.TestCase):
    def setUp(self):
        self.checker = NoveltyChecker()

    def test_exact_duplicate_rejected(self):
        seen = [_seen("What is the difference between HTTP and HTTPS?")]
        verdict = self.checker.check("What is the difference between HTTP and HTTPS?", seen=seen)
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "exact_duplicate")

    def test_normalised_exact_duplicate_rejected(self):
        seen = [_seen("What is the difference between HTTP and HTTPS?")]
        verdict = self.checker.check("  what  is the difference between http and https? ", seen=seen)
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "exact_duplicate")

    def test_containment_near_duplicate_rejected(self):
        seen = [_seen("Explain how TCP congestion control works in detail")]
        verdict = self.checker.check("Explain how TCP congestion control works", seen=seen)
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "near_duplicate_containment")

    def test_char_ratio_near_duplicate_rejected(self):
        seen = [_seen("How does TLS 1.3 establish a secure session between two hosts?")]
        verdict = self.checker.check("How does TLS 1.3 establish a secure session between two host?", seen=seen)
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "near_duplicate_char")

    def test_empty_candidate_rejected(self):
        verdict = self.checker.check("   ", seen=[])
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "empty_question")

    def test_novel_question_accepted(self):
        seen = [_seen("What is the difference between HTTP and HTTPS?")]
        verdict = self.checker.check("Walk me through the TCP three-way handshake.", seen=seen)
        self.assertEqual(verdict.outcome, "accepted")
        self.assertEqual(verdict.reason, "novel")


class ConceptStageTests(unittest.TestCase):
    def setUp(self):
        self.checker = NoveltyChecker()

    def test_concept_equivalent_rejected(self):
        seen = [_seen("How does HTTPS protect data in transit?")]
        verdict = self.checker.check("How does HTTPS keep data protected in transit?", seen=seen)
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "concept_equivalent")
        self.assertGreaterEqual(verdict.score, 0.65)

    def test_distinct_perspective_allows_same_concept(self):
        seen = [_seen("How does HTTPS protect data in transit?", perspective="conceptual")]
        verdict = self.checker.check(
            "How does HTTPS keep data protected in transit?",
            seen=seen,
            perspective="applied",
        )
        self.assertEqual(verdict.outcome, "accepted")

    def test_same_perspective_still_rejects(self):
        seen = [_seen("How does HTTPS protect data in transit?", perspective="conceptual")]
        verdict = self.checker.check(
            "How does HTTPS keep data protected in transit?",
            seen=seen,
            perspective="conceptual",
        )
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "concept_equivalent")


class EmbeddingStageTests(unittest.TestCase):
    def setUp(self):
        self.checker = NoveltyChecker()

    def test_embedding_similar_rejected(self):
        seen = [_seen(
            "What is the role of the DNS resolver?",
            embedding=(1.0, 0.0, 0.0),
        )]
        verdict = self.checker.check(
            "Describe what a DNS resolver does.",
            seen=seen,
            candidate_embedding=[0.99, 0.01, 0.0],
        )
        self.assertEqual(verdict.outcome, "rejected")
        self.assertEqual(verdict.reason, "embedding_similar")

    def test_embedding_review_band_returns_review(self):
        # cosine ~0.866 between [1,0] and [0.866, 0.5]
        checker = NoveltyChecker(embedding_threshold=0.95, embedding_review_threshold=0.85)
        seen = [_seen("What is the role of the DNS resolver?", embedding=(1.0, 0.0))]
        verdict = checker.check(
            "Describe what a DNS resolver does.",
            seen=seen,
            candidate_embedding=[0.866, 0.5],
        )
        self.assertEqual(verdict.outcome, "review")
        self.assertEqual(verdict.reason, "embedding_review_band")

    def test_embedding_stage_skipped_without_embeddings(self):
        seen = [_seen("What is the role of the DNS resolver?")]
        verdict = self.checker.check(
            "Describe what a DNS resolver does.",
            seen=seen,
            candidate_embedding=[1.0, 0.0],
        )
        # no stored embeddings -> similarity stage has nothing to compare
        self.assertEqual(verdict.outcome, "accepted")


if __name__ == "__main__":
    unittest.main()
