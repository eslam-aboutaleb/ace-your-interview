"""Coverage for ``app.routers.chat`` — topic resolution, memory, RAG ask gate.

The LLM, the parser and the learning store are all in-process fakes, so no
provider call and no database file is involved. The RAG gate is exercised
through a patched settings object for ``STUDY_ENABLE_RAG_V1``.
"""

import unittest
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import get_settings
from app.dependencies import require_auth
from app.routers import chat

BASE_BODY: dict[str, Any] = {
    "word": "idempotency",
    "context_question": "What makes a write idempotent?",
    "context_answer": "Repeating the request has no additional effect.",
    "user_message": "How do I make my retry safe?",
}


class RecordingLLM:
    def __init__(self, result=None):
        self.calls: list[dict[str, Any]] = []
        self.result = result or {
            "success": True,
            "analysis": "Use an idempotency key.",
            "metadata": {"provider": "groq", "model": "llama"},
            "error": "",
        }

    async def completion(self, prompt, llm_config=None, user_identity=None, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return dict(self.result)


class FakeTopic:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self, mode=None):
        return dict(self._payload)


class FakeParser:
    def __init__(self, topics=None):
        self.topics = topics or {}
        self.requested: list[str] = []

    def get_topic(self, topic_id):
        self.requested.append(topic_id)
        topic = self.topics.get(topic_id)
        return FakeTopic(topic) if topic else None


class FakeLearningStore:
    """Minimal LearningStore surface the router actually calls."""

    def __init__(self, *, custom_topic=None, resolved=None, memory=None):
        self.custom_topic = custom_topic
        self.resolved = resolved or {}
        self.memory = memory
        self.resolve_calls: list[dict] = []
        self.memory_calls: list[dict] = []
        self.upserts: list[dict] = []

    async def run_async(self, fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    def get_custom_topic(self, *, user_id, topic_id):
        return self.custom_topic

    def resolve_topic_ai_settings(self, *, user_id, topic_id, topic_detail):
        self.resolve_calls.append(
            {"user_id": user_id, "topic_id": topic_id, "topic_detail": topic_detail}
        )
        return self.resolved

    def get_assistant_memory(self, *, user_id, conversation_id, flow):
        self.memory_calls.append(
            {"user_id": user_id, "conversation_id": conversation_id, "flow": flow}
        )
        return self.memory

    def upsert_assistant_memory(self, **kwargs):
        self.upserts.append(kwargs)
        return {}


class FakeRagService:
    def __init__(self):
        self.calls: list[dict] = []

    async def ask(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": "The cited answer.",
            "citations": [
                {
                    "document_id": "doc-1",
                    "chunk_index": 2,
                    "quote": "cited text",
                }
            ],
            "conversation_id": "conv-1",
        }


class ChatRouterTestBase(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM()
        self.parser = None
        self.store = None
        self.rag = None
        self._prev = (
            chat._llm_client,
            chat._parser,
            chat._learning_store,
            chat._mcp_gateway,
            chat._rag_service,
        )
        chat.init(self.llm)
        chat.init_rag_service(None)

        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.app.dependency_overrides[require_auth] = lambda: {
            "user": "learner@example.com",
            "provider": "google",
        }
        self.client = TestClient(self.app)

    def tearDown(self):
        (
            chat._llm_client,
            chat._parser,
            chat._learning_store,
            chat._mcp_gateway,
            chat._rag_service,
        ) = self._prev
        self.app.dependency_overrides.clear()

    def _wire(self, *, parser=None, store=None, rag=None):
        chat.init(self.llm, parser, store)
        self.parser = parser
        self.store = store
        if rag is not None:
            self.rag = rag
            chat.init_rag_service(rag)


class MemoryAndHistoryBlockTests(unittest.TestCase):
    def test_memory_block_is_empty_for_non_dict_input(self):
        self.assertEqual(chat._memory_prompt_block(None), "")
        self.assertEqual(chat._memory_prompt_block("not-a-dict"), "")
        self.assertEqual(chat._memory_prompt_block({}), "")

    def test_memory_block_fences_the_assistant_summary(self):
        block = chat._memory_prompt_block({"summary": {"summary": "prior context"}})
        self.assertIn("prior context", block)
        self.assertIn("Conversation memory", block)

    def test_history_block_skips_blank_messages(self):
        block = chat._history_block(
            [
                type("M", (), {"role": "user", "content": "  "})(),
                type("M", (), {"role": "assistant", "content": ""})(),
            ]
        )
        self.assertEqual(block, "")

    def test_history_block_labels_roles(self):
        block = chat._history_block(
            [
                type("M", (), {"role": "user", "content": "ask"})(),
                type("M", (), {"role": "assistant", "content": "answer"})(),
                type("M", (), {"role": "tool", "content": "meta"})(),
            ]
        )
        self.assertIn("User: ask", block)
        self.assertIn("Assistant: answer", block)
        self.assertIn("Assistant: meta", block)

    def test_history_block_is_empty_for_missing_history(self):
        self.assertEqual(chat._history_block(None), "")
        self.assertEqual(chat._history_block([]), "")

    def test_conversation_id_prefers_the_client_value_then_a_derived_fallback(self):
        explicit = chat.ChatFollowUpRequest(
            **BASE_BODY, conversation_id="  custom-id  "
        )
        self.assertEqual(chat._conversation_id(explicit), "custom-id")

        derived = chat.ChatFollowUpRequest(
            **BASE_BODY, topic_id="01-http", section_title="Basics"
        )
        self.assertEqual(
            chat._conversation_id(derived), "01-http:basics:idempotency"
        )

        # Without a topic or section the highlighted word alone is the key.
        bare = chat.ChatFollowUpRequest(**BASE_BODY)
        self.assertEqual(chat._conversation_id(bare), "idempotency")


class FollowUpServiceGuardTests(ChatRouterTestBase):
    def test_missing_llm_client_is_reported_as_unavailable(self):
        chat._llm_client = None
        response = self.client.post("/api/chat/follow-up", json=BASE_BODY)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Service not initialised")


class FollowUpTopicResolutionTests(ChatRouterTestBase):
    def test_static_topic_from_the_parser_is_preferred_and_resolved(self):
        parser = FakeParser({"01-http": {"id": "01-http", "title": "HTTP"}})
        store = FakeLearningStore(
            resolved={
                "response_detail": "detailed",
                "preferred_language": "python",
                "requires_programming": True,
            }
        )
        self._wire(parser=parser, store=store)

        response = self.client.post(
            "/api/chat/follow-up", json={**BASE_BODY, "topic_id": "01-http"}
        )

        self.assertEqual(response.status_code, 200)
        prompt = self.llm.calls[0]["prompt"]
        # The resolved profile drives the instruction set.
        self.assertIn("very detailed responses", prompt)
        self.assertIn("```python", prompt)
        self.assertEqual(store.resolve_calls[0]["topic_detail"]["title"], "HTTP")

    def test_custom_topic_is_used_when_the_parser_has_no_static_topic(self):
        parser = FakeParser({})
        store = FakeLearningStore(
            custom_topic={"id": "custom-1", "title": "Rust"},
            resolved={"response_detail": "concise"},
        )
        self._wire(parser=parser, store=store)

        response = self.client.post(
            "/api/chat/follow-up",
            json={**BASE_BODY, "topic_id": "custom-1", "requires_programming": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(store.resolve_calls[0]["topic_detail"]["title"], "Rust")
        prompt = self.llm.calls[0]["prompt"]
        self.assertIn("concise responses", prompt)
        self.assertIn("Avoid code blocks", prompt)

    def test_explicit_request_fields_win_over_the_resolved_topic_profile(self):
        parser = FakeParser({"01-http": {"id": "01-http", "title": "HTTP"}})
        store = FakeLearningStore(
            resolved={
                "response_detail": "detailed",
                "preferred_language": "go",
                "requires_programming": True,
            }
        )
        self._wire(parser=parser, store=store)

        self.client.post(
            "/api/chat/follow-up",
            json={
                **BASE_BODY,
                "topic_id": "01-http",
                "response_detail": "concise",
                "preferred_language": "python",
                "requires_programming": False,
            },
        )

        prompt = self.llm.calls[0]["prompt"]
        self.assertIn("concise responses", prompt)
        self.assertIn("Avoid code blocks", prompt)

    def test_topic_metadata_lines_are_fenced_into_the_prompt(self):
        self._wire()
        self.client.post(
            "/api/chat/follow-up",
            json={
                **BASE_BODY,
                "topic_id": "01-http",
                "topic_title": "HTTP",
                "topic_track": "backend",
                "section_title": "Basics",
                "mode": "study",
            },
        )
        prompt = self.llm.calls[0]["prompt"]
        self.assertIn("- Topic ID: 01-http", prompt)
        self.assertIn("- Topic Title: HTTP", prompt)
        self.assertIn("- Topic Track: backend", prompt)
        self.assertIn("- Section: Basics", prompt)
        self.assertIn("- Learning Mode: study", prompt)

    def test_missing_topic_metadata_is_stated_explicitly(self):
        self._wire()
        self.client.post("/api/chat/follow-up", json=BASE_BODY)
        self.assertIn("Topic metadata not provided", self.llm.calls[0]["prompt"])

    def test_programming_request_without_a_language_emits_no_language_rule(self):
        self._wire()
        self.client.post(
            "/api/chat/follow-up",
            json={**BASE_BODY, "requires_programming": True},
        )

        prompt = self.llm.calls[0]["prompt"]
        # Neither a fenced-example instruction nor the "avoid code" clause:
        # the client asked for code but named no language to write it in.
        self.assertNotIn("fenced code example in", prompt)
        self.assertNotIn("Avoid code blocks", prompt)
        self.assertIn("You may include fenced code blocks when code clarifies", prompt)


class FollowUpMemoryTests(ChatRouterTestBase):
    def test_use_memory_persists_a_bounded_summary_after_a_successful_reply(self):
        store = FakeLearningStore(memory={"summary": {"summary": "earlier: greeting"}})
        self._wire(store=store)

        response = self.client.post(
            "/api/chat/follow-up",
            json={**BASE_BODY, "topic_id": "01-http", "topic_title": "HTTP",
                  "use_memory": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(store.memory_calls)
        upsert = store.upserts[-1]
        self.assertEqual(upsert["flow"], "chat")
        self.assertIn("earlier: greeting", upsert["summary"]["summary"])
        self.assertIn("User asked: How do I make my retry safe?", upsert["summary"]["summary"])
        self.assertIn("Assistant: Use an idempotency key.", upsert["summary"]["summary"])
        self.assertEqual(upsert["summary"]["updated_by"], "chat_follow_up")

    def test_existing_memory_summary_is_injected_as_a_fenced_block(self):
        store = FakeLearningStore(memory={"summary": {"summary": "user cares about retries"}})
        self._wire(store=store)

        self.client.post(
            "/api/chat/follow-up", json={**BASE_BODY, "use_memory": True}
        )

        prompt = self.llm.calls[0]["prompt"]
        self.assertIn("user cares about retries", prompt)
        self.assertIn("Conversation memory", prompt)

    def test_memory_is_not_touched_when_use_memory_is_off(self):
        store = FakeLearningStore(memory={"summary": {"summary": "ignored"}})
        self._wire(store=store)

        self.client.post("/api/chat/follow-up", json={**BASE_BODY, "use_memory": False})

        self.assertEqual(store.memory_calls, [])
        self.assertEqual(store.upserts, [])

    def test_llm_failure_degrades_to_an_apology_without_touching_memory(self):
        self.llm.result = {
            "success": False,
            "analysis": "",
            "metadata": {},
            "error": "provider unavailable",
        }
        store = FakeLearningStore()
        self._wire(store=store)

        response = self.client.post(
            "/api/chat/follow-up", json={**BASE_BODY, "use_memory": True}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("couldn't process that request", response.json()["reply"])
        self.assertIn("provider unavailable", response.json()["reply"])
        self.assertEqual(store.upserts, [])

    def test_budget_exceeded_is_reported_as_a_retryable_429(self):
        self.llm.result = {
            "success": False,
            "analysis": "",
            "metadata": {},
            "error": "LLM call budget exceeded.",
            "error_code": "llm_budget_exceeded",
        }
        self._wire()

        response = self.client.post("/api/chat/follow-up", json=BASE_BODY)

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["detail"]["code"], "llm_budget_exceeded")


class AskEndpointTests(ChatRouterTestBase):
    def test_missing_rag_service_is_reported_as_unavailable(self):
        self._wire()
        with patch_settings(enable_rag_v1=True):
            response = self.client.post("/api/chat/ask", json={"message": "what is rag?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Service not initialised")

    def test_rag_disabled_is_reported_as_a_feature_flag_503(self):
        self._wire(rag=FakeRagService())
        with patch_settings(enable_rag_v1=False):
            response = self.client.post("/api/chat/ask", json={"message": "what is rag?"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("STUDY_ENABLE_RAG_V1=false", response.json()["detail"])

    def test_enabled_rag_returns_the_answer_with_citations(self):
        rag = FakeRagService()
        self._wire(rag=rag)
        with patch_settings(enable_rag_v1=True):
            response = self.client.post(
                "/api/chat/ask",
                json={
                    "message": "how do we page?",
                    "document_ids": ["doc-1"],
                    "topic_id": "01-http",
                    "conversation_id": "conv-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"], "The cited answer.")
        self.assertEqual(payload["conversation_id"], "conv-1")
        self.assertEqual(len(payload["citations"]), 1)
        self.assertEqual(rag.calls[0]["user_id"], "learner@example.com")
        self.assertEqual(rag.calls[0]["document_ids"], ["doc-1"])
        self.assertEqual(rag.calls[0]["topic_id"], "01-http")

    def test_ask_rejects_an_empty_message(self):
        self._wire(rag=FakeRagService())
        with patch_settings(enable_rag_v1=True):
            response = self.client.post("/api/chat/ask", json={"message": ""})
        self.assertEqual(response.status_code, 422)


class _SettingsPatch:
    """Temporarily override one ``Settings`` field behind the lru_cache."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self._prev = None
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        self._prev = get_settings()
        for key, value in self.overrides.items():
            self._saved[key] = getattr(self._prev, key)
            setattr(self._prev, key, value)
        return self._prev

    def __exit__(self, *exc_info):
        for key, value in self._saved.items():
            setattr(self._prev, key, value)
        return False


def patch_settings(**overrides):
    return _SettingsPatch(**overrides)


if __name__ == "__main__":
    unittest.main()