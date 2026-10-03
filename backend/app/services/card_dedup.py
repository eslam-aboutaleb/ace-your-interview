"""Semantic duplicate detection for flashcards.

Primary path: Jaccard similarity over normalized
token sets of the card text (front + back). This is
the baseline that always works, including when the
sqlite-vec extension is unavailable (the pure-Python
fallback environment).

Accelerator: when an embedding function is supplied
and the vector index is available, cosine similarity
over embeddings refines the candidate pairs. The
accelerator is optional — Jaccard alone produces the
complete result set.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by",
        "for", "from", "has", "how", "in", "is", "it", "of",
        "on", "or", "the", "to", "was", "what", "when", "why",
        "with", "would", "you", "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_THRESHOLD = 0.85


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    cleaned = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return cleaned


def tokenize(text: str) -> set[str]:
    """Tokenize into a stemmed-ish word set."""
    words = _TOKEN_RE.findall(normalize_text(text))
    tokens: set[str] = set()
    for word in words:
        if len(word) <= 2 or word in _STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and len(word) > 3 and not word.endswith(
            ("ss", "us", "is")
        ):
            word = word[:-1]
        tokens.add(word)
    return tokens


def jaccard_similarity(
    tokens_a: set[str], tokens_b: set[str]
) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return intersection / union


def cosine_similarity(
    vector_a: list[float], vector_b: list[float]
) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(vector_a, vector_b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _card_text(card: dict[str, Any]) -> str:
    return f"{card.get('front', '')}\n{card.get('back', '')}"


def find_duplicate_pairs(
    cards: Iterable[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    vectors: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Return similar card pairs above ``threshold``.

    ``cards`` items carry at least ``card_id`` plus
    ``front``/``back`` text. Jaccard over normalized
    text is the primary detector. When ``vectors``
    (one embedding per card, in the same order) is
    supplied — the optional accelerator used when the
    vector index is available — cosine similarity
    refines each Jaccard candidate pair and the pair
    score becomes the max of both signals.
    """
    limit = max(0.0, min(1.0, float(threshold)))
    card_list = [
        card for card in cards if str(card.get("front", "")).strip()
    ]
    token_sets = [tokenize(_card_text(card)) for card in card_list]

    candidates: list[tuple[int, int]] = []
    for i in range(len(card_list)):
        for j in range(i + 1, len(card_list)):
            score = jaccard_similarity(token_sets[i], token_sets[j])
            if score >= limit:
                candidates.append((i, j))

    if not candidates:
        return []

    vectors_usable = bool(
        vectors
        and len(vectors) == len(card_list)
        and all(isinstance(v, list) for v in vectors)
    )

    pairs: list[dict[str, Any]] = []
    for i, j in candidates:
        jaccard = jaccard_similarity(token_sets[i], token_sets[j])
        cosine: float | None = None
        similarity = jaccard
        if vectors_usable:
            cosine = cosine_similarity(vectors[i], vectors[j])
            similarity = max(jaccard, cosine)
        pairs.append(
            _pair_payload(
                card_list[i],
                card_list[j],
                similarity,
                jaccard,
                cosine,
            )
        )
    pairs.sort(key=lambda item: item["similarity"], reverse=True)
    return pairs


def _pair_payload(
    card_a: dict[str, Any],
    card_b: dict[str, Any],
    similarity: float,
    jaccard: float,
    cosine: float | None,
) -> dict[str, Any]:
    return {
        "card_a": {
            "card_id": str(card_a.get("card_id", "")),
            "deck_id": str(card_a.get("deck_id", "")),
            "front": str(card_a.get("front", ""))[:200],
        },
        "card_b": {
            "card_id": str(card_b.get("card_id", "")),
            "deck_id": str(card_b.get("deck_id", "")),
            "front": str(card_b.get("front", ""))[:200],
        },
        "similarity": round(float(similarity), 4),
        "jaccard": round(float(jaccard), 4),
        "cosine": (
            round(float(cosine), 4) if cosine is not None else None
        ),
    }
