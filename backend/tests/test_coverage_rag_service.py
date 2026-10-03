"""Coverage for app/services/rag_service.py.

Complements ``tests/test_rag_service.py`` (citation precision, the repair retry,
injection neutrality) with the citation *parser* fallbacks, the degenerate-input
branches of ``ask``, the vector-index re-embedding path, and the chunk-context
ceiling.
"""

import asyncio
import json
import unittest

from app.services.rag_service import (
    CHUNK_CONTEXT_MAX_CHARS,
    RETRIEVAL_TOP_K,
    RagService,
    split_answer_and_citations,
)


def _fence(payload) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


def _raw_fence(body: str) -> str:
    """A citation fence whose body is *not* JSON-encoded first."""
    return "```json\n" + body + "\n```"


def _run(coro):
    return asyncio.run(coro)


class CitationParserTests(unittest.TestCase):
    def test_a_json_fence_with_no_citations_key_yields_nothing(self):
        answer, citations = split_answer_and_citations(
            "Answer.\n\n" + _fence({"other": 1})
        )
        self.assertEqual(answer, "Answer.")
        self.assertEqual(citations, [])

    def test_a_non_list_citations_value_is_ignored(self):
        _answer, citations = split_answer_and_citations(
            "Answer.\n\n" + _fence({"citations": {"document_id": "d"}})
        )
        self.assertEqual(citations, [])

    def test_a_json_array_body_is_not_an_object(self):
        _answer, citations = split_answer_and_citations(
            "Answer.\n\n" + _fence([1, 2, 3])
        )
        self.assertEqual(citations, [])

    def test_non_dict_citation_entries_are_skipped(self):
        _answer, citations = split_answer_and_citations(
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        "not an object",
                        None,
                        {"document_id": "d1", "chunk_index": 0, "quote": "q"},
                    ]
                }
            )
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["document_id"], "d1")

    def test_missing_citation_fields_default_to_empty(self):
        _answer, citations = split_answer_and_citations(
            "Answer.\n\n" + _fence({"citations": [{}]})
        )
        self.assertEqual(citations, [{"document_id": "", "chunk_index": None, "quote": ""}])

    def test_a_json_object_wrapped_in_prose_is_recovered(self):
        body = (
            'Here you go: {"citations": [{"document_id": "d1", '
            '"chunk_index": 3, "quote": "q"}]} hope that helps'
        )
        _answer, citations = split_answer_and_citations(_raw_fence(body))
        self.assertEqual(citations[0]["chunk_index"], 3)
        self.assertEqual(citations[0]["document_id"], "d1")

    def test_prose_without_any_object_yields_nothing(self):
        _answer, citations = split_answer_and_citations(
            _raw_fence("there is no json object in here at all")
        )
        self.assertEqual(citations, [])

    def test_prose_with_an_unterminated_object_yields_nothing(self):
        _answer, citations = split_answer_and_citations(
            _raw_fence('{"citations": [{"document_id": "d1"')
        )
        self.assertEqual(citations, [])

    def test_a_closed_brace_before_the_opening_brace_yields_nothing(self):
        _answer, citations = split_answer_and_citations(_raw_fence("} then {"))
        self.assertEqual(citations, [])

    def test_braces_that_bound_but_are_not_valid_json_yield_nothing(self):
        # The brace slice is well formed but its contents are not JSON, so the
        # second parse fails too.
        _answer, citations = split_answer_and_citations(
            _raw_fence("{ not: valid json }")
        )
        self.assertEqual(citations, [])

    def test_a_json_array_between_braces_is_not_an_object(self):
        _answer, citations = split_answer_and_citations(
            _raw_fence('{ "citations": [1, 2] }')
        )
        self.assertEqual(len(citations), 0)

    def test_an_empty_fence_body_yields_nothing(self):
        _answer, citations = split_answer_and_citations("Answer.\n\n```json\n\n```")
        self.assertEqual(citations, [])

    def test_a_fence_without_the_json_info_string_is_ignored(self):
        answer, citations = split_answer_and_citations(
            'Answer.\n\n```\n{"citations": []}\n```'
        )
        self.assertEqual(answer, 'Answer.\n\n```\n{"citations": []}\n```')
        self.assertEqual(citations, [])

    def test_none_text_is_handled(self):
        self.assertEqual(split_answer_and_citations(None), ("", []))


class _FakeStore:
    """Minimal document store for the RAG branches the sqlite store cannot reach."""

    def __init__(self, *, hits=(), rebuilt=False, missing=()):
        self.hits = list(hits)
        self.rebuilt = rebuilt
        self.missing = list(missing)
        self.calls: list[tuple] = []
        self.upserts: list[tuple[int, list[float]]] = []
        self.embeddings: list[tuple[list[str], object]] = []
        self._embed_calls = 0
        self.embedding_client = None

    async def run_async(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    def ensure_vec_index(self, dimension):
        self.calls.append(("ensure_vec_index", dimension))
        return self.rebuilt

    def chunks_missing_vectors(self, dimension):
        self.calls.append(("chunks_missing_vectors", dimension))
        return self.missing

    def upsert_chunk_embedding(self, chunk_id, embedding):
        self.calls.append(("upsert", chunk_id))
        self.upserts.append((chunk_id, embedding))

    def search(self, *, user_id, query_embedding, k, document_ids=None):
        self.calls.append(("search", k, document_ids))
        return self.hits

    def get_chunk(self, user_id, document_id, chunk_index):
        self.calls.append(("get_chunk", document_id, chunk_index))
        for chunk in self.missing:
            if chunk_index == chunk.get("chunk_index"):
                return chunk
        return None


class _FakeEmbedder:
    def __init__(self, dimension=4, vectors=None, empty=False):
        self.dimension = dimension
        self._vectors = vectors
        self._empty = empty
        self.batches: list[list[str]] = []

    async def embed(self, texts, user_identity=None):
        self.batches.append(list(texts))
        if self._empty:
            return []
        if self._vectors is not None:
            return self._vectors
        return [[1.0] * self.dimension for _ in texts]


class _FakeLLM:
    def __init__(self, *results):
        self.results = list(results)
        self.prompts: list[str] = []

    async def completion(self, prompt, llm_config=None, user_identity=None, task=None):
        self.prompts.append(prompt)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if isinstance(result, dict):
            return result
        return {
            "success": True,
            "analysis": result,
            "metadata": {},
            "error": "",
        }


def _hit(content="chunk body", document_id="d1", chunk_index=0):
    return {
        "chunk_id": 1,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": content,
        "score": 0.9,
    }


class AskDegenerateInputTests(unittest.TestCase):
    def _service(self, store=None, embedder=None, llm=None):
        return RagService(
            store or _FakeStore(hits=[_hit()]),
            llm or _FakeLLM("answer"),
            embedder or _FakeEmbedder(),
        )

    def test_an_empty_message_is_rejected_before_anything_else(self):
        service = self._service()
        with self.assertRaises(ValueError):
            _run(service.ask(user_id="u1", message=""))

    def test_no_query_vector_is_reported_without_calling_the_model(self):
        store = _FakeStore(hits=[_hit()])
        llm = _FakeLLM("unused")
        service = self._service(store=store, embedder=_FakeEmbedder(empty=True), llm=llm)
        result = _run(service.ask(user_id="u1", message="what is caching?"))
        self.assertIn("could not process your question", result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertTrue(result["conversation_id"])
        # Retrieval and the model are both skipped.
        self.assertEqual([c[0] for c in store.calls], ["ensure_vec_index"])
        self.assertEqual(llm.prompts, [])

    def test_an_empty_corpus_never_calls_the_model(self):
        store = _FakeStore(hits=[])
        llm = _FakeLLM("unused")
        service = self._service(store=store, llm=llm)
        result = _run(service.ask(user_id="u1", message="what is caching?"))
        self.assertIn("couldn't find relevant content", result["answer"])
        self.assertEqual(llm.prompts, [])

    def test_retrieval_is_scoped_to_the_requested_documents(self):
        store = _FakeStore(hits=[_hit()])
        service = self._service(store=store)
        _run(
            service.ask(
                user_id="u1",
                message="q",
                document_ids=["d1", "  ", "", "d2"],
            )
        )
        search_call = next(c for c in store.calls if c[0] == "search")
        self.assertEqual(search_call[1], RETRIEVAL_TOP_K)
        # An empty-string id is dropped; a whitespace-only id survives the
        # truthiness filter but is stripped to "" by the scope builder.
        self.assertEqual(search_call[2], ["d1", "", "d2"])

    def test_a_whitespace_only_scope_is_still_sent_as_a_scope(self):
        store = _FakeStore(hits=[_hit()])
        service = self._service(store=store)
        _run(service.ask(user_id="u1", message="q", document_ids=["  "]))
        search_call = next(c for c in store.calls if c[0] == "search")
        # The scope survives as [""] rather than collapsing to None, so the
        # store is asked for chunks of a document with an empty id.
        self.assertEqual(search_call[2], [""])

    def test_an_empty_scope_list_means_no_scope(self):
        store = _FakeStore(hits=[_hit()])
        service = self._service(store=store)
        _run(service.ask(user_id="u1", message="q", document_ids=[]))
        search_call = next(c for c in store.calls if c[0] == "search")
        self.assertIsNone(search_call[2])

    def test_an_llm_result_without_a_message_is_reported(self):
        store = _FakeStore(hits=[_hit()])
        # No "error" key at all, so the service has nothing to report.
        llm = _FakeLLM({"success": True, "analysis": "", "metadata": {}})
        service = self._service(store=store, llm=llm)
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertIn("Unknown error", result["answer"])
        self.assertEqual(result["citations"], [])

    def test_an_llm_result_with_a_blank_error_still_reports_it(self):
        store = _FakeStore(hits=[_hit()])
        llm = _FakeLLM(
            {"success": True, "analysis": "", "metadata": {}, "error": ""}
        )
        service = self._service(store=store, llm=llm)
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertIn("Sorry, I couldn't process that request.", result["answer"])

    def test_a_blank_conversation_id_is_replaced(self):
        service = self._service()
        result = _run(service.ask(user_id="u1", message="q", conversation_id="   "))
        self.assertTrue(result["conversation_id"].startswith("doc-chat:"))


class CitationVerificationTests(unittest.TestCase):
    def _service(self, store, *results):
        return RagService(store, _FakeLLM(*results), _FakeEmbedder())

    def test_a_citation_missing_required_fields_is_dropped(self):
        store = _FakeStore(
            hits=[_hit(content="the exact body")],
            missing=[{"chunk_index": 0, "content": "the exact body"}],
        )
        store.get_chunk = lambda user_id, document_id, chunk_index: {
            "content": "the exact body"
        }
        llm = _FakeLLM(
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        {"chunk_index": 0, "quote": "the exact body"},
                        {"document_id": "d1", "quote": "the exact body"},
                        {"document_id": "d1", "chunk_index": 0},
                    ]
                }
            )
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(result["citations"], [])

    def test_a_non_integer_chunk_index_is_dropped(self):
        store = _FakeStore(hits=[_hit()])
        seen: list[tuple] = []

        def get_chunk(user_id, document_id, chunk_index):
            seen.append((document_id, chunk_index))
            return {"content": "body"}

        store.get_chunk = get_chunk
        llm = _FakeLLM(
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        {"document_id": "d1", "chunk_index": "not-an-int", "quote": "body"}
                    ]
                }
            )
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(result["citations"], [])
        # The malformed index never reached the store.
        self.assertEqual(seen, [])

    def test_a_citation_for_a_deleted_chunk_is_dropped(self):
        store = _FakeStore(hits=[_hit()])
        store.get_chunk = lambda user_id, document_id, chunk_index: None
        llm = _FakeLLM(
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        {"document_id": "d1", "chunk_index": 0, "quote": "body"}
                    ]
                }
            )
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(result["citations"], [])

    def test_a_repeated_citation_appears_once(self):
        store = _FakeStore(hits=[_hit(content="the body")])
        store.get_chunk = lambda user_id, document_id, chunk_index: {
            "content": "the body"
        }
        entry = {"document_id": "d1", "chunk_index": 0, "quote": "the body"}
        llm = _FakeLLM("Answer.\n\n" + _fence({"citations": [entry, dict(entry)]}))
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(len(result["citations"]), 1)

    def test_quote_matching_is_whitespace_sensitive(self):
        store = _FakeStore(hits=[_hit(content="the body")])
        store.get_chunk = lambda user_id, document_id, chunk_index: {
            "content": "the body"
        }
        llm = _FakeLLM(
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        {
                            "document_id": "d1",
                            "chunk_index": 0,
                            "quote": "the  body",
                        }
                    ]
                }
            )
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        # A quote that differs only in whitespace is not an exact substring.
        self.assertEqual(result["citations"], [])


class RepairFallbackTests(unittest.TestCase):
    def _store(self):
        store = _FakeStore(hits=[_hit(content="the body")])
        store.get_chunk = lambda user_id, document_id, chunk_index: {
            "content": "the body"
        }
        return store

    def _bad_citation_block(self):
        return (
            "Answer.\n\n"
            + _fence(
                {
                    "citations": [
                        {
                            "document_id": "d1",
                            "chunk_index": 0,
                            "quote": "not the stored text",
                        }
                    ]
                }
            )
        )

    def test_a_repair_answer_with_no_verified_citations_is_discarded(self):
        store = self._store()
        llm = _FakeLLM(
            self._bad_citation_block(),
            # The repair quotes the same wrong text, so nothing verifies.
            "Repaired answer.\n\n" + self._bad_citation_block().split("\n\n", 1)[1],
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        # The original (unverified) answer and its empty citation list stand.
        self.assertEqual(result["citations"], [])
        self.assertNotIn("Repaired answer", result["answer"])

    def test_a_failed_repair_call_is_discarded(self):
        store = self._store()
        llm = _FakeLLM(
            self._bad_citation_block(),
            {"success": False, "analysis": "", "metadata": {}, "error": "nope"},
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(result["citations"], [])
        self.assertEqual(len(llm.prompts), 2)

    def test_a_repair_answer_without_a_body_is_discarded(self):
        store = self._store()
        llm = _FakeLLM(
            self._bad_citation_block(),
            "Repaired with no citation block at all",
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertEqual(result["citations"], [])
        # The repair produced no citations, so verification yields nothing and
        # the original answer is kept.
        self.assertIn("Answer.", result["answer"])

    def test_a_successful_repair_replaces_the_answer_and_citations(self):
        store = self._store()
        good = {
            "citations": [
                {"document_id": "d1", "chunk_index": 0, "quote": "the body"}
            ]
        }
        llm = _FakeLLM(
            self._bad_citation_block(),
            "Fully repaired.\n\n" + _fence(good),
        )
        service = RagService(store, llm, _FakeEmbedder())
        result = _run(service.ask(user_id="u1", message="q"))
        self.assertIn("Fully repaired.", result["answer"])
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["quote"], "the body")

    def test_the_repair_prompt_lists_the_offending_citations(self):
        store = self._store()
        llm = _FakeLLM(
            self._bad_citation_block(),
            "Repaired.\n\n"
            + _fence(
                {
                    "citations": [
                        {"document_id": "d1", "chunk_index": 0, "quote": "the body"}
                    ]
                }
            ),
        )
        service = RagService(store, llm, _FakeEmbedder())
        _run(service.ask(user_id="u1", message="q"))
        repair_prompt = llm.prompts[1]
        self.assertIn("Citations that failed verification", repair_prompt)
        self.assertIn("not the stored text", repair_prompt)
        self.assertIn("Repeat the previous answer", repair_prompt)


class IndexRebuildTests(unittest.TestCase):
    def _service(self, store, embedder):
        return RagService(store, _FakeLLM("answer"), embedder)

    def test_a_unchanged_index_skips_the_re_embedding_pass(self):
        store = _FakeStore(hits=[_hit()], rebuilt=False, missing=[{"chunk_id": 1}])
        embedder = _FakeEmbedder()
        _run(self._service(store, embedder).ask(user_id="u1", message="q"))
        self.assertEqual(
            [call[0] for call in store.calls if call[0].startswith("ensure")],
            ["ensure_vec_index"],
        )
        # Only the query itself was embedded.
        self.assertEqual(len(embedder.batches), 1)

    def test_a_rebuilt_index_with_nothing_to_do_skips_the_re_embedding_pass(self):
        store = _FakeStore(hits=[_hit()], rebuilt=True, missing=[])
        embedder = _FakeEmbedder()
        _run(self._service(store, embedder).ask(user_id="u1", message="q"))
        self.assertIn(("chunks_missing_vectors", 4), store.calls)
        self.assertEqual(store.upserts, [])

    def test_a_rebuilt_index_re_embeds_the_missing_chunks(self):
        store = _FakeStore(
            hits=[_hit()],
            rebuilt=True,
            missing=[
                {"chunk_id": 11, "content": "chunk eleven"},
                {"chunk_id": 12, "content": "chunk twelve"},
            ],
        )
        embedder = _FakeEmbedder()
        _run(self._service(store, embedder).ask(user_id="u1", message="q"))
        # The chunk texts are embedded as a batch before the query.
        self.assertEqual(
            embedder.batches[0], ["chunk eleven", "chunk twelve"]
        )
        self.assertEqual(
            store.upserts,
            [(11, [1.0, 1.0, 1.0, 1.0]), (12, [1.0, 1.0, 1.0, 1.0])],
        )

    def test_a_partial_embedding_response_only_upserts_what_arrived(self):
        store = _FakeStore(
            hits=[_hit()],
            rebuilt=True,
            missing=[
                {"chunk_id": 11, "content": "eleven"},
                {"chunk_id": 12, "content": "twelve"},
            ],
        )
        embedder = _FakeEmbedder(vectors=[[0.5, 0.5, 0.5, 0.5]])
        _run(self._service(store, embedder).ask(user_id="u1", message="q"))
        # zip() stops at the shorter list, so a short provider response can
        # never index-shift the upserts.
        self.assertEqual(store.upserts, [(11, [0.5, 0.5, 0.5, 0.5])])

    def test_the_identity_is_forwarded_to_the_embedding_calls(self):
        store = _FakeStore(hits=[_hit()])
        embedder = _FakeEmbedder()
        service = self._service(store, embedder)

        captured: list[object] = []

        async def embed(texts, user_identity=None):
            captured.append(user_identity)
            return [[1.0] * 4 for _ in texts]

        embedder.embed = embed
        identity = {"user": "u1", "provider": "google"}
        _run(
            service.ask(
                user_id="u1", message="q", user_identity=identity
            )
        )
        self.assertEqual(captured, [identity])


class PromptShapeTests(unittest.TestCase):
    def test_the_chunk_context_is_capped_at_the_documented_ceiling(self):
        big = "z" * (CHUNK_CONTEXT_MAX_CHARS + 5_000)
        store = _FakeStore(hits=[_hit(content=big, document_id="d1", chunk_index=0)])
        llm = _FakeLLM("answer")
        service = RagService(store, llm, _FakeEmbedder())
        _run(service.ask(user_id="u1", message="q"))
        prompt = llm.prompts[0]
        self.assertIn(f"[truncated at {CHUNK_CONTEXT_MAX_CHARS} characters]", prompt)
        self.assertLess(len(prompt), CHUNK_CONTEXT_MAX_CHARS + 4_000)

    def test_every_retrieved_chunk_is_labelled_with_its_coordinates(self):
        store = _FakeStore(
            hits=[
                _hit(content="alpha body", document_id="doc-a", chunk_index=0),
                _hit(content="beta body", document_id="doc-b", chunk_index=7),
            ]
        )
        llm = _FakeLLM("answer")
        service = RagService(store, llm, _FakeEmbedder())
        _run(service.ask(user_id="u1", message="q"))
        prompt = llm.prompts[0]
        self.assertIn("=== Chunk 0 (document_id: doc-a, chunk_index: 0) ===", prompt)
        self.assertIn("=== Chunk 1 (document_id: doc-b, chunk_index: 7) ===", prompt)
        self.assertIn("alpha body", prompt)
        self.assertIn("beta body", prompt)

    def test_the_question_is_interpolated_verbatim(self):
        store = _FakeStore(hits=[_hit()])
        llm = _FakeLLM("answer")
        service = RagService(store, llm, _FakeEmbedder())
        _run(service.ask(user_id="u1", message="  What is a WAL?  "))
        self.assertIn("User's question: What is a WAL?", llm.prompts[0])

    def test_the_prompt_carries_the_untrusted_clause(self):
        store = _FakeStore(hits=[_hit()])
        llm = _FakeLLM("answer")
        service = RagService(store, llm, _FakeEmbedder())
        _run(service.ask(user_id="u1", message="q"))
        self.assertIn("<untrusted_input", llm.prompts[0])
        self.assertIn("untrusted context", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()