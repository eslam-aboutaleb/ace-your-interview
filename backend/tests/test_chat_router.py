import asyncio
import unittest

from app.routers import chat
from app.schemas.models import ChatFollowUpRequest
from app.services.llm_policy import LLMServiceApprovalRequiredError


class FakeLLM:
    def __init__(self, success: bool = True, blocked: bool = False):
        self.success = success
        self.blocked = blocked
        self.last_prompt = ""

    async def completion(self, prompt, llm_config=None, user_identity=None):
        self.last_prompt = prompt
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
            }
        return {
            "success": False,
            "analysis": "",
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "provider_down",
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


if __name__ == "__main__":
    unittest.main()
