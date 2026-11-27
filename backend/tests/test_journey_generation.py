"""Service tests for novelty-aware journey generation (issue #2).

All tests are deterministic: the LLM generator, embedding provider and
adjudicator are injected callables. Golden multi-batch fixtures cover the
acceptance criteria: zero exact duplicates, no semantically equivalent
duplicates, legitimate deepening preserved, retries limited to rejected
slots, and fingerprints surviving a store restart.
"""

import asyncio
import os
import tempfile
import unittest

from app.services.journey_generation import (
    JourneyGenerationError,
    JourneyGenerationService,
)
from app.services.journey_store import JourneyStore


_OBJECTIVE_NAMES = [
    "HTTP", "DNS", "TLS", "Caching", "Proxies", "Queues",
    "Routing", "Hashing", "Indexing", "Sharding", "Replication", "Partitioning",
]


def _plan(n_objectives: int) -> dict:
    assert n_objectives <= len(_OBJECTIVE_NAMES)
    return {
        "phases": [
            {
                "title": "Phase 1",
                "description": "",
                "modules": [
                    {
                        "title": "Module",
                        "description": "",
                        "objectives": [
                            {
                                "title": _OBJECTIVE_NAMES[i],
                                "kind": "concept",
                                "source_section": "Module",
                            }
                            for i in range(n_objectives)
                        ],
                    }
                ],
            }
        ]
    }


class _Store:
    def __init__(self):
        self._td = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self._td.name, "learning.db")

    def __enter__(self):
        self.store = JourneyStore(self.db_path)
        return self

    def reopen(self) -> JourneyStore:
        """Simulate a process restart: a fresh store over the same file."""
        self.store = JourneyStore(self.db_path)
        return self.store

    def __exit__(self, *exc_info):
        self._td.cleanup()
        return False


def _create_journey(store, n_objectives=3, user_id="alice"):
    return store.create_journey(
        user_id=user_id,
        title="Backend journey",
        goal="ace the interview",
        source_type="static_topic",
        source_topic_id="01-backend-fundamentals-and-http",
        track="backend",
        level="junior",
        plan=_plan(n_objectives),
    )


_STEMS = [
    "How does {t} handle its first production outage?",
    "Walk me through the consistency guarantees of {t}.",
    "What trade-offs does {t} make under heavy write load?",
    "Explain {t} to a junior engineer with a concrete analogy.",
    "Design an audit trail around {t} failures for a regulated shop.",
    "Compare {t} with a naive alternative in operational cost terms.",
    "Which metrics would you alert on for {t}, and why those?",
    "Describe a migration path that avoids downtime for {t}.",
]


def _make_generator(script: dict[str, list[str]] | None = None):
    """Deterministic generator_fn.

    ``script`` maps objective title -> list of candidate texts consumed one
    per attempt; when exhausted (or absent) a unique fresh question is
    produced. Records every call for retry assertions.
    """
    calls: list[dict] = []
    counters: dict[str, int] = {}

    async def _generate(*, journey, slot, objective, existing_questions, perspective, attempt, retry_reason):
        title = objective.get("title", slot["id"])
        idx = counters.get(title, 0)
        counters[title] = idx + 1
        seq = len(calls)  # global call index: default stems stay distinct
        calls.append(
            {
                "slot_id": slot["id"],
                "objective": title,
                "attempt": attempt,
                "retry_reason": retry_reason,
                "perspective": perspective,
            }
        )
        scripted = (script or {}).get(title, [])
        if idx < len(scripted):
            text = scripted[idx]
        else:
            # a real LLM varies wording; uniform templates would trip the
            # near-duplicate char stage, so stems differ in shape and tokens
            stem = _STEMS[seq % len(_STEMS)]
            text = stem.format(t=f"{title} v{idx}")
        return {"question": text, "raw": {"question": text}}

    return _generate, calls


def _run(coro):
    return asyncio.run(coro)


class GoldenMultiBatchTests(unittest.TestCase):
    """Acceptance: batches 1..N, zero exact/semantic duplicates, counts right."""

    def test_three_batches_all_accepted_no_duplicates(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=6)
            service = JourneyGenerationService(store)
            accepted_texts: list[str] = []
            # one generator across batches: a real LLM keeps varying its
            # wording, so the deterministic fixture draws fresh stems from
            # a shared sequence instead of replaying the same templates
            generator, _calls = _make_generator()
            for n in range(3):
                batch, created = store.create_batch(
                    user_id="alice", journey_id=journey["id"],
                    idempotency_key=f"batch-{n}", slot_count=2,
                )
                self.assertTrue(created)
                outcome = _run(
                    service.generate_batch(
                        user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                        generator_fn=generator,
                        prompt_version="questions@v3", model_version="test-model",
                        schema_version="v2",
                    )
                )
                self.assertEqual(outcome["job"]["status"], "completed")
                self.assertEqual(outcome["job"]["accepted_count"], 2)
                self.assertEqual(outcome["job"]["rejected_count"], 0)
                self.assertEqual(outcome["job"]["prompt_version"], "questions@v3")
                self.assertEqual(outcome["batch"]["accepted_count"], 2)
            fps = store.fingerprints_for_journey(user_id="alice", journey_id=journey["id"])
            self.assertEqual(len(fps), 6)
            fingerprints = {f["fingerprint"] for f in fps}
            self.assertEqual(len(fingerprints), 6, "zero exact duplicates")
            accepted_texts = [f["question_text"] for f in fps]
            self.assertEqual(len(set(accepted_texts)), 6)

    def test_duplicate_candidate_rejected_and_only_that_slot_retried(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=2
            )
            # HTTP is slot position 0 and generates first; its accepted
            # text is the first stem. The DNS slot's first candidate
            # duplicates it verbatim; the retry must be fresh.
            script = {
                "DNS": ["How does HTTP v0 handle its first production outage?"],
            }
            generator, calls = _make_generator(script)
            service = JourneyGenerationService(store)
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            self.assertEqual(outcome["job"]["accepted_count"], 2)
            obj0_calls = [c for c in calls if c["objective"] == "HTTP"]
            obj1_calls = [c for c in calls if c["objective"] == "DNS"]
            self.assertEqual(len(obj1_calls), 2, "rejected slot retried")
            self.assertEqual(obj1_calls[1]["retry_reason"], "exact_duplicate")
            self.assertEqual(len(obj0_calls), 1, "accepted slot never regenerated")
            retry_result = next(
                r for r in outcome["results"] if r["slot_id"] == obj1_calls[0]["slot_id"]
            )
            self.assertEqual(retry_result["outcome"], "accepted")
            self.assertEqual(retry_result["rounds"], 2)

    def test_slot_rejected_after_bounded_rounds(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=1)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=1
            )
            # every candidate is the same text -> exact dup from round 2 on,
            # and the first candidate collides with itself across rounds
            script = {"HTTP": ["Same question always?"] * 5}
            generator, calls = _make_generator(script)
            service = JourneyGenerationService(store, max_rounds=2)
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            # round 1 accepts (nothing seen yet); to force rejection the
            # generator must repeat within one slot - so pre-seed instead:
            self.assertEqual(outcome["job"]["accepted_count"], 1)

            # second batch: generator keeps producing the same text
            store2 = store
            # retire nothing; add a fresh objective via a new journey instead
            journey2 = _create_journey(store2, n_objectives=1)
            # seed journey2 with an accepted question, then always duplicate it
            batch2, _ = store2.create_batch(
                user_id="alice", journey_id=journey2["id"], idempotency_key="x", slot_count=1
            )
            gen2, _ = _make_generator({"HTTP": ["Seeded question?"]})
            _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey2["id"], batch_id=batch2["id"],
                    generator_fn=gen2,
                )
            )
            # a second journey batch over the same objective set: refresh trick
            # instead, verify bounded rounds on a journey whose generator only dupes
            journey3 = _create_journey(store2, n_objectives=2)
            batch3, _ = store2.create_batch(
                user_id="alice", journey_id=journey3["id"], idempotency_key="y", slot_count=2
            )
            gen3, calls3 = _make_generator(
                {
                    "HTTP": ["Constant text?"],
                    "DNS": ["Constant text?", "Constant text?", "Constant text?"],
                }
            )
            service2 = JourneyGenerationService(store2, max_rounds=3)
            outcome3 = _run(
                service2.generate_batch(
                    user_id="alice", journey_id=journey3["id"], batch_id=batch3["id"],
                    generator_fn=gen3,
                )
            )
            self.assertEqual(outcome3["job"]["accepted_count"], 1)
            self.assertEqual(outcome3["job"]["rejected_count"], 1)
            self.assertEqual(outcome3["batch"]["rejected_count"], 1)
            obj1_calls = [c for c in calls3 if c["objective"] == "DNS"]
            self.assertEqual(len(obj1_calls), 3, "bounded at max_rounds")
            rejected = next(r for r in outcome3["results"] if r["outcome"] == "rejected")
            self.assertEqual(rejected["reason"], "exact_duplicate")

    def test_regenerate_call_on_finished_batch_conflicts(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=1)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=1
            )
            generator, _ = _make_generator()
            service = JourneyGenerationService(store)
            _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            with self.assertRaises(JourneyGenerationError):
                _run(
                    service.generate_batch(
                        user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                        generator_fn=generator,
                    )
                )


class PersistenceTests(unittest.TestCase):
    def test_fingerprints_survive_restart(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            service = JourneyGenerationService(store)
            batch1, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=1
            )
            gen1, _ = _make_generator({"HTTP": ["What is a reverse proxy?"]})
            _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch1["id"],
                    generator_fn=gen1,
                )
            )
            # restart
            store = env.reopen()
            service = JourneyGenerationService(store)
            batch2, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b2", slot_count=1
            )
            gen2, calls2 = _make_generator({"DNS": ["What is a reverse proxy?"]})
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch2["id"],
                    generator_fn=gen2,
                )
            )
            # the duplicate of the pre-restart question is rejected, retried,
            # and the scripted-exhaustion fallback (fresh text) is accepted
            self.assertEqual(outcome["job"]["accepted_count"], 1)
            self.assertEqual(len(calls2), 2)
            self.assertEqual(calls2[1]["retry_reason"], "exact_duplicate")
            fps = store.fingerprints_for_journey(user_id="alice", journey_id=journey["id"])
            self.assertEqual(len(fps), 2)


class PerspectiveTests(unittest.TestCase):
    def test_distinct_perspective_deepening_preserved(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1",
                slot_count=2, perspectives=["conceptual", "applied"],
            )
            slots = store.get_batch_slots(user_id="alice", journey_id=journey["id"], batch_id=batch["id"])
            self.assertEqual(slots[0]["perspective"], "conceptual")
            self.assertEqual(slots[1]["perspective"], "applied")
            # both slots yield near-identical concept text: legitimate deepening
            script = {
                "HTTP": ["How does HTTPS protect data in transit?"],
                "DNS": ["How does HTTPS keep data protected in transit?"],
            }
            generator, _ = _make_generator(script)
            service = JourneyGenerationService(store)
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            self.assertEqual(outcome["job"]["accepted_count"], 2)
            reasons = {r["reason"] for r in outcome["results"]}
            self.assertEqual(reasons, {"novel"})

    def test_same_perspective_concept_dup_rejected(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=2
            )
            script = {
                "HTTP": ["How does HTTPS protect data in transit?"],
                "DNS": ["How does HTTPS keep data protected in transit?"],
            }
            generator, calls = _make_generator(script)
            service = JourneyGenerationService(store, max_rounds=2)
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            obj1_calls = [c for c in calls if c["objective"] == "DNS"]
            # first dupe rejected (concept_equivalent), retry consumes scripted
            # dupes, final fallback fresh text accepted within rounds
            self.assertGreaterEqual(len(obj1_calls), 2)
            first_retry = obj1_calls[1]
            self.assertEqual(first_retry["retry_reason"], "concept_equivalent")
            self.assertEqual(outcome["job"]["accepted_count"], 2)


class EmbeddingAndAdjudicationTests(unittest.TestCase):
    def test_embedding_similar_rejected_without_adjudicator(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=2
            )
            script = {
                "HTTP": ["What is the role of the DNS resolver?"],
                "DNS": ["Describe what a DNS resolver does."] * 3,
            }
            generator, calls = _make_generator(script)

            async def embed(text: str):
                # every question lands on nearly the same vector
                return [1.0, 0.01]

            service = JourneyGenerationService(store, embed_fn=embed, max_rounds=2)
            _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            obj1_calls = [c for c in calls if c["objective"] == "DNS"]
            self.assertGreaterEqual(len(obj1_calls), 2)
            self.assertEqual(obj1_calls[1]["retry_reason"], "embedding_similar")

    def test_review_band_adjudicated(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=2
            )
            script = {
                "HTTP": ["What happens during the alpha migration window?"],
                "DNS": ["Explain the beta rollback procedure step by step?"],
            }
            generator, _ = _make_generator(script)
            adjudicator_calls: list[tuple[str, str]] = []

            async def embed(text: str):
                # cosine([1,0], [0.9,0.436]) = 0.9 -> inside the review band
                return [1.0, 0.0] if "alpha" in text else [0.9, 0.436]

            async def adjudicate(candidate: str, matched: str) -> bool:
                adjudicator_calls.append((candidate, matched))
                return False  # LLM says: legitimately different

            from app.services.novelty import NoveltyChecker
            checker = NoveltyChecker(embedding_threshold=0.95, embedding_review_threshold=0.85)
            service = JourneyGenerationService(
                store, checker=checker, embed_fn=embed, adjudicator_fn=adjudicate
            )
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            self.assertEqual(outcome["job"]["accepted_count"], 2)
            self.assertEqual(len(adjudicator_calls), 1)
            reasons = {r["reason"] for r in outcome["results"]}
            self.assertIn("llm_adjudicated_novel", reasons)

    def test_embedding_failure_degrades_gracefully(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=1)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=1
            )
            generator, _ = _make_generator()

            async def embed(text: str):
                raise RuntimeError("provider down")

            service = JourneyGenerationService(store, embed_fn=embed)
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            self.assertEqual(outcome["job"]["accepted_count"], 1)


class GeneratorFailureTests(unittest.TestCase):
    def test_generator_error_fails_job_and_leaves_slots_reserved(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=1)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=1
            )

            async def failing(**kwargs):
                raise RuntimeError("LLM quota exhausted")

            service = JourneyGenerationService(store)
            with self.assertRaises(RuntimeError):
                _run(
                    service.generate_batch(
                        user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                        generator_fn=failing,
                    )
                )
            jobs = store._conn.execute(
                "SELECT * FROM journey_generation_jobs WHERE batch_id = ?", (batch["id"],)
            ).fetchall()
            self.assertEqual(jobs[0]["status"], "failed")
            self.assertIn("quota", jobs[0]["error"])
            # slots stay reserved so a later call can retry the batch
            slots = store.get_batch_slots(user_id="alice", journey_id=journey["id"], batch_id=batch["id"])
            self.assertEqual(slots[0]["status"], "reserved")
            # and the retry path works
            generator, _ = _make_generator()
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            self.assertEqual(outcome["job"]["accepted_count"], 1)


if __name__ == "__main__":
    unittest.main()


class ProductionAdapterTests(unittest.TestCase):
    """build_question_generator_fn over the real DocParser + stub generator."""

    def test_static_topic_section_lookup_and_generate(self):
        from app.services.doc_parser import DocParser
        from app.services.journey_generation import build_question_generator_fn

        parser = DocParser()
        topics = parser.list_topics()
        self.assertTrue(topics)
        detail = parser.get_topic(topics[0].id)
        heading = detail.sections[0].get("heading", "")

        class _Gen:
            def __init__(self):
                self.kwargs = None

            async def generate_v2(self, **kwargs):
                self.kwargs = kwargs

                class _Item:
                    question = "What is the request/response lifecycle?"

                    def model_dump(self):
                        return {"question": self.question}

                class _Resp:
                    questions = [_Item()]
                    provider_used = "stub"
                    model_used = "stub"
                    generation_version = "stub@v1"

                return _Resp()

        gen = _Gen()
        fn = build_question_generator_fn(gen, parser, None, user_identity={"user": "u"})
        journey = {
            "id": "jrn_x",
            "title": detail.title,
            "source_type": "static_topic",
            "source_topic_id": topics[0].id,
            "level": "junior",
        }
        slot = {"id": "slot_x", "objective_title": "Obj", "perspective": "applied"}
        objective = {"title": "Obj", "source_section": heading}
        result = _run(
            fn(
                journey=journey, slot=slot, objective=objective,
                existing_questions=["old?"], perspective="applied",
                attempt=1, retry_reason="",
            )
        )
        self.assertEqual(result["question"], "What is the request/response lifecycle?")
        self.assertEqual(gen.kwargs["count"], 1)
        self.assertEqual(gen.kwargs["existing_questions"], ["old?"])
        self.assertEqual(gen.kwargs["section_title"], "Obj (applied perspective)")
        self.assertTrue(gen.kwargs["section_content"], "section content must ground generation")
        self.assertTrue(gen.kwargs["doc_content"])
        self.assertEqual(result["generation_version"], "stub@v1")

    def test_static_unknown_heading_and_unknown_topic_degrade(self):
        from app.services.doc_parser import DocParser
        from app.services.journey_generation import build_question_generator_fn

        parser = DocParser()
        topics = parser.list_topics()

        class _Gen:
            async def generate_v2(self, **kwargs):
                self.kwargs = kwargs

                class _Item:
                    question = "Q?"

                    def model_dump(self):
                        return {"question": "Q?"}

                class _Resp:
                    questions = [_Item()]
                    provider_used = ""
                    model_used = ""
                    generation_version = ""

                return _Resp()

        gen = _Gen()
        fn = build_question_generator_fn(gen, parser, None)
        base = {
            "id": "jrn_x", "title": "T", "source_type": "static_topic",
            "source_topic_id": topics[0].id, "level": "",
        }
        slot = {"id": "s", "objective_title": "O", "perspective": ""}
        result = _run(fn(journey=base, slot=slot, objective={"title": "O", "source_section": "No such heading"}, existing_questions=[], perspective="", attempt=1, retry_reason=""))
        self.assertEqual(result["question"], "Q?")
        self.assertEqual(gen.kwargs["section_content"], "")
        unknown = dict(base, source_topic_id="topic-nope")
        result = _run(fn(journey=unknown, slot=slot, objective={"title": "O", "source_section": ""}, existing_questions=[], perspective="", attempt=1, retry_reason=""))
        self.assertEqual(result["question"], "Q?")
        self.assertEqual(gen.kwargs["doc_content"], "")

    def test_custom_topic_lookup_paths(self):
        from app.services.doc_parser import DocParser
        from app.services.journey_generation import build_question_generator_fn

        class _LearningStore:
            def __init__(self, custom):
                self._custom = custom

            def get_custom_topic(self, *, user_id, topic_id):
                return self._custom

        class _Gen:
            async def generate_v2(self, **kwargs):
                self.kwargs = kwargs

                class _Item:
                    question = "Q?"

                    def model_dump(self):
                        return {"question": "Q?"}

                class _Resp:
                    questions = [_Item()]
                    provider_used = ""
                    model_used = ""
                    generation_version = ""

                return _Resp()

        journey = {
            "id": "jrn_c", "title": "Custom", "source_type": "custom_topic",
            "source_topic_id": "custom-1", "level": "",
        }
        slot = {"id": "s", "objective_title": "O", "perspective": ""}
        objective = {"title": "O", "source_section": "Deep dive"}

        custom = {
            "raw_content": "RAW",
            "sections": [{"heading": "Deep dive", "content": "SECTION"}],
        }
        gen = _Gen()
        fn = build_question_generator_fn(gen, DocParser(), _LearningStore(custom), user_identity={"user": "u"})
        _run(fn(journey=journey, slot=slot, objective=objective, existing_questions=[], perspective="", attempt=1, retry_reason=""))
        self.assertEqual(gen.kwargs["doc_content"], "RAW")
        self.assertEqual(gen.kwargs["section_content"], "SECTION")

        gen2 = _Gen()
        fn = build_question_generator_fn(gen2, DocParser(), _LearningStore(None), user_identity={"user": "u"})
        _run(fn(journey=journey, slot=slot, objective=objective, existing_questions=[], perspective="", attempt=1, retry_reason=""))
        self.assertEqual(gen2.kwargs["doc_content"], "")

        gen3 = _Gen()
        fn = build_question_generator_fn(gen3, DocParser(), None, user_identity={"user": "u"})
        _run(fn(journey=journey, slot=slot, objective=objective, existing_questions=[], perspective="", attempt=1, retry_reason=""))
        self.assertEqual(gen3.kwargs["doc_content"], "")

    def test_empty_generator_response_raises(self):
        from app.services.doc_parser import DocParser
        from app.services.journey_generation import (
            JourneyGenerationError,
            build_question_generator_fn,
        )

        class _Gen:
            async def generate_v2(self, **kwargs):
                class _Resp:
                    questions = []
                    provider_used = ""
                    model_used = ""
                    generation_version = ""

                return _Resp()

        fn = build_question_generator_fn(_Gen(), DocParser(), None)
        journey = {
            "id": "j", "title": "T", "source_type": "static_topic",
            "source_topic_id": "nope", "level": "",
        }
        with self.assertRaises(JourneyGenerationError):
            _run(fn(journey=journey, slot={"id": "s", "objective_title": "O", "perspective": ""}, objective={"title": "O"}, existing_questions=[], perspective="", attempt=1, retry_reason=""))

    def test_adjudicator_failure_rejects_conservatively(self):
        with _Store() as env:
            store = env.store
            journey = _create_journey(store, n_objectives=2)
            batch, _ = store.create_batch(
                user_id="alice", journey_id=journey["id"], idempotency_key="b1", slot_count=2
            )
            script = {
                "HTTP": ["What happens during the alpha migration window?"],
                "DNS": ["Explain the beta rollback procedure step by step?"],
            }
            generator, _ = _make_generator(script)

            async def embed(text: str):
                return [1.0, 0.0] if "alpha" in text else [0.9, 0.436]

            async def adjudicate(candidate: str, matched: str) -> bool:
                raise RuntimeError("adjudicator down")

            from app.services.novelty import NoveltyChecker
            checker = NoveltyChecker(embedding_threshold=0.95, embedding_review_threshold=0.85)
            service = JourneyGenerationService(
                store, checker=checker, embed_fn=embed, adjudicator_fn=adjudicate, max_rounds=1
            )
            outcome = _run(
                service.generate_batch(
                    user_id="alice", journey_id=journey["id"], batch_id=batch["id"],
                    generator_fn=generator,
                )
            )
            rejected = [r for r in outcome["results"] if r["outcome"] == "rejected"]
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]["reason"], "llm_adjudicated_duplicate")

    def test_service_requires_positive_rounds(self):
        with _Store() as env:
            with self.assertRaises(ValueError):
                JourneyGenerationService(env.store, max_rounds=0)
