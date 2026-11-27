"""Novelty-aware journey question generation (issue #2).

Fills reserved batch slots with generated questions, rejecting duplicates
through the :mod:`app.services.novelty` funnel and regenerating ONLY the
rejected slots. Everything is persisted: per-question fingerprints (hash,
concept signature, optional embedding) survive restarts, and each run is a
``journey_generation_jobs`` row carrying prompt/model/schema versions plus
accepted/rejected counts that roll up onto the batch.

The service is generator-agnostic: the caller injects ``generator_fn``
(production wraps :class:`QuestionGenerator`), plus optional ``embed_fn``
and ``adjudicator_fn``. Without embeddings the similarity stage is skipped;
without an adjudicator the embedding review band rejects conservatively -
regenerating one slot is cheap, a semantic duplicate in a learner's journey
is not.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from app.services.journey_store import JourneyStore
from app.services.novelty import (
    OUTCOME_ACCEPTED,
    OUTCOME_REVIEW,
    NoveltyChecker,
    NoveltyVerdict,
    SeenQuestion,
    concept_signature,
    question_fingerprint,
)

logger = logging.getLogger(__name__)

MAX_EXISTING_QUESTIONS_IN_PROMPT = 200

GeneratorFn = Callable[..., Awaitable[dict[str, Any]]]
EmbedFn = Callable[[str], Awaitable[Optional[list[float]]]]
AdjudicatorFn = Callable[[str, str], Awaitable[bool]]


@dataclass
class SlotResult:
    slot_id: str
    outcome: str  # accepted | rejected
    reason: str
    rounds: int
    score: float = 0.0


class JourneyGenerationError(RuntimeError):
    """Raised when a batch cannot be generated at all."""


class JourneyGenerationService:
    def __init__(
        self,
        store: JourneyStore,
        *,
        checker: NoveltyChecker | None = None,
        embed_fn: EmbedFn | None = None,
        adjudicator_fn: AdjudicatorFn | None = None,
        max_rounds: int = 3,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        self.store = store
        self.checker = checker or NoveltyChecker()
        self.embed_fn = embed_fn
        self.adjudicator_fn = adjudicator_fn
        self.max_rounds = max_rounds

    def _seen_pairs(self, *, user_id: str, journey_id: str) -> list[tuple[SeenQuestion, str]]:
        records = self.store.fingerprints_for_journey(user_id=user_id, journey_id=journey_id)
        return [
            (
                SeenQuestion(
                    fingerprint=rec["fingerprint"],
                    signature=rec["signature"],
                    embedding=rec["embedding"],
                    perspective=rec["perspective"],
                ),
                rec["question_text"],
            )
            for rec in records
        ]

    async def _embed(self, text: str) -> list[float] | None:
        if self.embed_fn is None:
            return None
        try:
            return await self.embed_fn(text)
        except Exception:  # embeddings are best-effort; never fail a batch
            logger.warning("journey generation: embedding failed, continuing without", exc_info=True)
            return None

    async def generate_batch(
        self,
        *,
        user_id: str,
        journey_id: str,
        batch_id: str,
        generator_fn: GeneratorFn,
        prompt_version: str = "",
        model_version: str = "",
        schema_version: str = "",
    ) -> dict[str, Any]:
        journey = self.store.get_journey(user_id=user_id, journey_id=journey_id)
        objectives = {
            obj["id"]: obj
            for phase in journey["phases"]
            for module in phase["modules"]
            for obj in module["objectives"]
        }
        slots = self.store.get_batch_slots(user_id=user_id, journey_id=journey_id, batch_id=batch_id)
        reserved = [slot for slot in slots if slot["status"] == "reserved"]
        if not reserved:
            raise JourneyGenerationError(
                f"batch {batch_id!r} has no reserved slots - already generated?"
            )

        job = self.store.create_generation_job(
            user_id=user_id,
            journey_id=journey_id,
            batch_id=batch_id,
            prompt_version=prompt_version,
            model_version=model_version,
            schema_version=schema_version,
        )

        seen = self._seen_pairs(user_id=user_id, journey_id=journey_id)
        results: list[SlotResult] = []
        max_round_used = 0
        try:
            for slot in reserved:
                result = await self._generate_slot(
                    user_id=user_id,
                    journey=journey,
                    slot=slot,
                    objective=objectives.get(slot["objective_id"], {}),
                    generator_fn=generator_fn,
                    seen=seen,
                )
                max_round_used = max(max_round_used, result.rounds)
                results.append(result)
        except Exception as exc:
            self.store.complete_generation_job(
                user_id=user_id,
                job_id=job["id"],
                status="failed",
                rounds=max_round_used,
                accepted_count=sum(1 for r in results if r.outcome == "accepted"),
                rejected_count=sum(1 for r in results if r.outcome == "rejected"),
                error=str(exc)[:500],
            )
            raise

        accepted = sum(1 for r in results if r.outcome == "accepted")
        rejected = sum(1 for r in results if r.outcome == "rejected")
        job = self.store.complete_generation_job(
            user_id=user_id,
            job_id=job["id"],
            status="completed",
            rounds=max_round_used,
            accepted_count=accepted,
            rejected_count=rejected,
        )
        batch = self.store.set_batch_generation_counts(
            user_id=user_id,
            journey_id=journey_id,
            batch_id=batch_id,
            accepted_count=accepted,
            rejected_count=rejected,
            status="completed" if rejected == 0 else "partial",
        )
        return {
            "job": job,
            "batch": batch,
            "results": [
                {
                    "slot_id": r.slot_id,
                    "outcome": r.outcome,
                    "reason": r.reason,
                    "rounds": r.rounds,
                    "score": r.score,
                }
                for r in results
            ],
        }

    async def _generate_slot(
        self,
        *,
        user_id: str,
        journey: dict[str, Any],
        slot: dict[str, Any],
        objective: dict[str, Any],
        generator_fn: GeneratorFn,
        seen: list[tuple[SeenQuestion, str]],
    ) -> SlotResult:
        slot_id = slot["id"]
        perspective = slot.get("perspective", "")
        last_verdict = NoveltyVerdict(OUTCOME_REVIEW, "not_generated")
        for round_no in range(1, self.max_rounds + 1):
            candidate = await generator_fn(
                journey=journey,
                slot=slot,
                objective=objective,
                existing_questions=[text for _seen, text in seen][
                    -MAX_EXISTING_QUESTIONS_IN_PROMPT:
                ],
                perspective=perspective,
                attempt=round_no,
                retry_reason=last_verdict.reason if round_no > 1 else "",
            )
            question_text = str(candidate.get("question") or "").strip()
            embedding = await self._embed(question_text)
            verdict = self.checker.check(
                question_text,
                seen=seen,
                candidate_embedding=embedding,
                perspective=perspective,
            )
            if verdict.outcome == OUTCOME_REVIEW and self.adjudicator_fn is not None:
                matched_text = next(
                    (text for prior, text in seen if prior.fingerprint == verdict.matched_fingerprint),
                    "",
                )
                try:
                    is_duplicate = await self.adjudicator_fn(question_text, matched_text)
                except Exception:
                    logger.warning("journey generation: adjudicator failed, rejecting conservatively", exc_info=True)
                    is_duplicate = True
                verdict = NoveltyVerdict(
                    "rejected" if is_duplicate else "accepted",
                    "llm_adjudicated_duplicate" if is_duplicate else "llm_adjudicated_novel",
                    verdict.score,
                    verdict.matched_fingerprint,
                )
            elif verdict.outcome == OUTCOME_REVIEW:
                # No adjudicator configured: conservative rejection.
                verdict = NoveltyVerdict(
                    "rejected", f"{verdict.reason}_no_adjudicator", verdict.score, verdict.matched_fingerprint
                )
            last_verdict = verdict
            if verdict.outcome == OUTCOME_ACCEPTED:
                self.store.accept_slot(
                    user_id=user_id,
                    journey_id=journey["id"],
                    slot_id=slot_id,
                    question_ref=question_fingerprint(question_text)[:14],
                    question_json=json.dumps(candidate.get("raw") or {"question": question_text}),
                    question_text=question_text,
                    fingerprint=question_fingerprint(question_text),
                    signature=sorted(concept_signature(question_text)),
                    embedding=embedding,
                    perspective=perspective,
                )
                seen.append(
                    (
                        SeenQuestion(
                            fingerprint=question_fingerprint(question_text),
                            signature=concept_signature(question_text),
                            embedding=tuple(embedding) if embedding else None,
                            perspective=perspective,
                        ),
                        question_text,
                    )
                )
                return SlotResult(slot_id, "accepted", verdict.reason, round_no, verdict.score)
        self.store.reject_slot(
            user_id=user_id,
            journey_id=journey["id"],
            slot_id=slot_id,
            reason=last_verdict.reason,
        )
        return SlotResult(slot_id, "rejected", last_verdict.reason, self.max_rounds, last_verdict.score)


def build_question_generator_fn(
    question_generator: Any,
    parser: Any,
    learning_store: Any,
    *,
    user_identity: dict | None = None,
) -> GeneratorFn:
    """Production adapter over ``QuestionGenerator.generate_v2``.

    One question per slot, aimed at the slot's objective: the objective
    title becomes the ``section_title`` (with the perspective appended when
    set, so intentionally distinct follow-ups also read differently at the
    prompt level) and the objective's source section content becomes the
    grounding ``section_content``. Every accepted journey question is fed
    back through ``existing_questions`` so the in-prompt dedup the
    generator already ships reinforces the persistent funnel.
    """

    def _section_lookup(journey: dict[str, Any], objective: dict[str, Any]) -> tuple[str, str, str]:
        """Return (topic_id, doc_content, section_content)."""
        heading = str(objective.get("source_section") or "")
        if journey["source_type"] == "static_topic":
            topic_id = journey["source_topic_id"]
            detail = parser.get_topic(topic_id) if parser is not None else None
            if detail is None:
                return topic_id, "", ""
            doc_content = parser.get_topic_content(topic_id)
            for section in detail.sections:
                if section.get("heading", "") == heading:
                    return topic_id, doc_content, section.get("content", "")
            return topic_id, doc_content, ""
        # custom_topic: sections live in the learning store
        if learning_store is None:
            return journey["source_topic_id"], "", ""
        custom = learning_store.get_custom_topic(
            user_id=str((user_identity or {}).get("user") or ""),
            topic_id=journey["source_topic_id"],
        )
        if not custom:
            return journey["source_topic_id"], "", ""
        doc_content = str(custom.get("raw_content") or "")
        for section in custom.get("sections") or []:
            if isinstance(section, dict) and str(section.get("heading", "")) == heading:
                return journey["source_topic_id"], doc_content, str(section.get("content", ""))
        return journey["source_topic_id"], doc_content, ""

    async def _generate(
        *,
        journey: dict[str, Any],
        slot: dict[str, Any],
        objective: dict[str, Any],
        existing_questions: list[str],
        perspective: str,
        attempt: int,
        retry_reason: str,
    ) -> dict[str, Any]:
        topic_id, doc_content, section_content = _section_lookup(journey, objective)
        objective_title = str(objective.get("title") or slot.get("objective_title") or "")
        section_title = objective_title
        if perspective:
            section_title = f"{objective_title} ({perspective} perspective)"
        response = await question_generator.generate_v2(
            topic_id=topic_id,
            topic_title=journey["title"],
            doc_content=doc_content,
            count=1,
            existing_questions=existing_questions,
            level=journey.get("level") or None,
            user_identity=user_identity,
            section_title=section_title,
            section_content=section_content,
        )
        if not response.questions:
            raise JourneyGenerationError("generator returned no question for slot")
        item = response.questions[0]
        raw = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        return {
            "question": raw.get("question", ""),
            "raw": raw,
            "provider": getattr(response, "provider_used", ""),
            "model": getattr(response, "model_used", ""),
            "generation_version": getattr(response, "generation_version", ""),
        }

    return _generate
