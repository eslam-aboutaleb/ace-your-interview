import asyncio
import os
import tempfile
import unittest

from app.routers import chat
from app.schemas.models import ChatFollowUpRequest
from app.services.llm_policy import LLMServiceApprovalRequiredError
from app.services.learning_store import LearningStore


class FakeLLM:
    def __init__(self, success: bool = True, blocked: bool = False, error_code: str = ""):
        self.success = success
        self.blocked = blocked
        self.error_code = error_code
        self.last_prompt = ""
        self.last_system = ""
        self.last_kwargs: dict = {}

    async def completion(
        self,
        prompt,
        llm_config=None,
        user_identity=None,
        task=None,
        *,
        system="",
        structured=False,
        max_tokens_cap=None,
        tools=None,
        tool_choice=None,
    ):
        self.last_prompt = prompt
        self.last_system = system
        self.last_kwargs = {
            "task": task,
            "structured": structured,
            "max_tokens_cap": max_tokens_cap,
        }
        if self.blocked:
            return {
                "success": False,
                "analysis": "",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "approval required",
                "error_code": "llm_service_approval_required",
            }
        if self.success:
            return {
                "success": True,
                "analysis": "### Meaning\nA concise explanation.\n\n- Practical takeaway",
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "",
                "error_code": "",
            }
        return {
            "success": False,
            "analysis": "",
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "provider_down",
            "error_code": self.error_code,
        }


class ChatRouterTests(unittest.TestCase):
    def test_follow_up_returns_markdown_reply_string(self):
        prev_llm = chat._llm_client
        fake = FakeLLM(success=True)
        chat.init(fake)
        try:
            payload = ChatFollowUpRequest(
                word="idempotency",
                context_question="What is idempotency?",
                context_answer="Idempotency means repeated requests have the same effect.",
                user_message="Why does this matter in retries?",
                history=[],
            )
            res = asyncio.run(chat.chat_follow_up(payload, user={"user": "u1", "provider": "google"}))
            self.assertTrue(isinstance(res.reply, str))
            self.assertIn("### Meaning", res.reply)
            self.assertIn("adaptive structure", fake.last_prompt)
            self.assertIn("valid GFM table syntax", fake.last_prompt)
            self.assertEqual(res.provider_used, "openai")
        finally:
            chat._llm_client = prev_llm

    def test_follow_up_failure_returns_safe_error_reply(self):
        prev_llm = chat._llm_client
        fake = FakeLLM(success=False)
        chat.init(fake)
        try:
            payload = ChatFollowUpRequest(
                word="cache",
                context_question="What is cache?",
                context_answer="Cache stores data for faster reads.",
                user_message="Explain tradeoffs.",
                history=[],
            )
            res = asyncio.run(chat.chat_follow_up(payload, user={"user": "u1", "provider": "google"}))
            self.assertIn("Sorry, I couldn't process that request.", res.reply)
            self.assertIn("provider_down", res.reply)
        finally:
            chat._llm_client = prev_llm

    def test_follow_up_blocked_raises_policy_error(self):
        prev_llm = chat._llm_client
        fake = FakeLLM(success=False, blocked=True)
        chat.init(fake)
        try:
            payload = ChatFollowUpRequest(
                word="cache",
                context_question="What is cache?",
                context_answer="Cache stores data for faster reads.",
                user_message="Explain tradeoffs.",
                history=[],
            )
            with self.assertRaises(LLMServiceApprovalRequiredError):
                asyncio.run(chat.chat_follow_up(payload, user={"user": "u1", "provider": "google"}))
        finally:
            chat._llm_client = prev_llm

    def test_follow_up_uses_memory_context_when_available(self):
        prev_llm = chat._llm_client
        prev_store = chat._learning_store
        fake = FakeLLM(success=True)
        with tempfile.TemporaryDirectory() as td:
            store = LearningStore(os.path.join(td, "learning.db"))
            store.upsert_assistant_memory(
                user_id="u1",
                conversation_id="conv-xyz",
                flow="chat",
                summary={"summary": "Earlier we discussed idempotency key storage tradeoffs."},
            )
            chat.init(fake, learning_store=store)
            try:
                payload = ChatFollowUpRequest(
                    word="idempotency",
                    context_question="What is idempotency?",
                    context_answer="Idempotency means repeated requests have the same effect.",
                    user_message="How do I store the key?",
                    conversation_id="conv-xyz",
                    history=[],
                )
                _ = asyncio.run(chat.chat_follow_up(payload, user={"user": "u1", "provider": "google"}))
                self.assertIn("Conversation memory", fake.last_prompt)
            finally:
                chat._llm_client = prev_llm
                chat._learning_store = prev_store


if __name__ == "__main__":
    unittest.main()
