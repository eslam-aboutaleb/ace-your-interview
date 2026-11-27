"""Novelty pipeline for journey question generation (issue #2).

A deterministic, dependency-free funnel that decides whether a generated
candidate question is novel for one user's journey:

1. exact normalized-hash check (sha1 over the shared normaliser),
2. character-level near-duplicate check (containment + sequence ratio),
3. concept equivalence check (content-token Jaccard),
4. calibrated embedding similarity check (cosine, when embeddings are
   provided by the caller), with a review band left for optional LLM
   adjudication by the caller.

The module never calls an LLM or an embedding provider itself; the
generation service computes embeddings and adjudications and passes the
results in. That keeps every decision here reproducible in tests.

Intentionally distinct follow-up perspectives: when both the candidate and
a seen question carry an explicit, different ``perspective``, the concept
equivalence stage is skipped - a conceptual and an applied question about
the same concept are legitimate deepening, not duplicates. Exact and
character-level duplicates are rejected regardless of perspective.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.services.question_text import normalise_question

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "to", "what", "when",
    "why", "with", "would", "you",
}

_WORD_RUN = re.compile(r"[a-z0-9]+")

OUTCOME_ACCEPTED = "accepted"
OUTCOME_REJECTED = "rejected"
OUTCOME_REVIEW = "review"


def question_fingerprint(text: str) -> str:
    """Stable exact-match key: sha1 over the shared normalised text."""
    return hashlib.sha1(normalise_question(text).encode("utf-8")).hexdigest()


def concept_signature(text: str) -> frozenset[str]:
    """Content tokens with light stemming; stopwords and tiny tokens dropped."""
    words = _WORD_RUN.findall(normalise_question(text))
    normalized: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in _STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        # Verb forms: "protected"/"protecting" must meet "protect", and a
        # trailing "e" unifies "cache"/"caching". Light stemming on purpose;
        # the char and embedding stages back it up.
        if word.endswith("ing") and len(word) > 5:
            word = word[:-3]
        elif word.endswith("ed") and len(word) > 4:
            word = word[:-2]
        if word.endswith("e") and len(word) > 3:
            word = word[:-1]
        normalized.add(word)
    return frozenset(normalized)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class SeenQuestion:
    """One previously accepted question in a journey."""

    fingerprint: str
    signature: frozenset[str] = field(default_factory=frozenset)
    embedding: tuple[float, ...] | None = None
    perspective: str = ""


@dataclass(frozen=True)
class NoveltyVerdict:
    outcome: str  # accepted | rejected | review
    reason: str
    score: float = 0.0
    matched_fingerprint: str = ""


class NoveltyChecker:
    """Threshold-calibrated duplicate funnel.

    ``char_threshold`` mirrors the generator's existing near-duplicate
    ratio; ``concept_threshold`` mirrors its token-overlap cutoff, so
    journey generation is at least as strict as the in-session dedup the
    product already ships. ``embedding_threshold`` rejects; the
    [``embedding_review_threshold``, ``embedding_threshold``) band is
    returned as ``review`` for optional LLM adjudication.
    """

    def __init__(
        self,
        *,
        char_threshold: float = 0.87,
        concept_threshold: float = 0.65,
        embedding_threshold: float = 0.92,
        embedding_review_threshold: float = 0.85,
        min_containment_length: int = 24,
    ) -> None:
        if not 0.0 < embedding_review_threshold <= embedding_threshold <= 1.0:
            raise ValueError("embedding review threshold must be <= threshold within (0, 1]")
        self.char_threshold = char_threshold
        self.concept_threshold = concept_threshold
        self.embedding_threshold = embedding_threshold
        self.embedding_review_threshold = embedding_review_threshold
        self.min_containment_length = min_containment_length

    def check(
        self,
        candidate_text: str,
        *,
        seen: list[tuple[SeenQuestion, str]],
        candidate_embedding: list[float] | None = None,
        perspective: str = "",
    ) -> NoveltyVerdict:
        """Full funnel. ``seen`` pairs each accepted question with its text."""
        normalised = normalise_question(candidate_text)
        if not normalised:
            return NoveltyVerdict(OUTCOME_REJECTED, "empty_question")

        fingerprint = question_fingerprint(candidate_text)
        signature = concept_signature(candidate_text)
        perspective = (perspective or "").strip().lower()

        # Stage 1: exact normalized hash.
        for prior, _text in seen:
            if prior.fingerprint == fingerprint:
                return NoveltyVerdict(
                    OUTCOME_REJECTED, "exact_duplicate", 1.0, prior.fingerprint
                )

        # Stage 2: character near-duplicate (containment + sequence ratio).
        best_char = 0.0
        best_char_fp = ""
        for prior, text in seen:
            prior_norm = normalise_question(text)
            if not prior_norm:
                continue
            if (
                normalised in prior_norm or prior_norm in normalised
            ) and min(len(normalised), len(prior_norm)) >= self.min_containment_length:
                return NoveltyVerdict(
                    OUTCOME_REJECTED, "near_duplicate_containment", 1.0, prior.fingerprint
                )
            ratio = SequenceMatcher(None, normalised, prior_norm).ratio()
            if ratio > best_char:
                best_char, best_char_fp = ratio, prior.fingerprint
        if best_char >= self.char_threshold:
            return NoveltyVerdict(
                OUTCOME_REJECTED, "near_duplicate_char", best_char, best_char_fp
            )

        # Stage 3: concept equivalence (skipped for intentionally distinct
        # follow-up perspectives).
        best_concept = 0.0
        best_concept_fp = ""
        for prior, _text in seen:
            if (
                perspective
                and prior.perspective
                and prior.perspective != perspective
            ):
                continue
            score = jaccard(signature, prior.signature)
            if score > best_concept:
                best_concept, best_concept_fp = score, prior.fingerprint
        if best_concept >= self.concept_threshold:
            return NoveltyVerdict(
                OUTCOME_REJECTED, "concept_equivalent", best_concept, best_concept_fp
            )

        # Stage 4: calibrated embedding similarity, when embeddings exist.
        best_embed = 0.0
        best_embed_fp = ""
        if candidate_embedding:
            for prior, _text in seen:
                if not prior.embedding:
                    continue
                score = cosine(candidate_embedding, list(prior.embedding))
                if score > best_embed:
                    best_embed, best_embed_fp = score, prior.fingerprint
            if best_embed >= self.embedding_threshold:
                return NoveltyVerdict(
                    OUTCOME_REJECTED, "embedding_similar", best_embed, best_embed_fp
                )
            if best_embed >= self.embedding_review_threshold:
                return NoveltyVerdict(
                    OUTCOME_REVIEW, "embedding_review_band", best_embed, best_embed_fp
                )

        return NoveltyVerdict(OUTCOME_ACCEPTED, "novel", max(best_char, best_concept, best_embed))
