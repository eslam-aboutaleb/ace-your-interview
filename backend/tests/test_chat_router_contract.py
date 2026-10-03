import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.dependencies import require_auth
from app.routers import chat


class FakeLLM:
    def __init__(self):
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
        return {
            "success": True,
            "analysis": "### Meaning\nContext-aware explanation.\n\n- Practical takeaway",
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


class ChatRouterContractTests(unittest.TestCase):
    def setUp(self):
        self.prev_llm = chat._llm_client
        self.fake = FakeLLM()
        chat.init(self.fake)
        self.app = FastAPI()
        self.app.include_router(chat.router)
        self.app.dependency_overrides[require_auth] = lambda: {"user": "contract-user", "provider": "google"}
        self.client = TestClient(self.app)

    def tearDown(self):
        chat._llm_client = self.prev_llm
        self.app.dependency_overrides.clear()

    def test_follow_up_accepts_topic_metadata_and_injects_prompt_context(self):
        res = self.client.post(
            "/api/chat/follow-up",
            json={
                "word": "reconciliation",
                "context_question": "What is virtual DOM reconciliation?",
                "context_answer": "Reconciliation compares trees to compute minimal DOM updates.",
                "topic_id": "frontend-react",
                "topic_title": "React Rendering",
                "topic_track": "frontend",
                "section_title": "Rendering Pipeline",
                "mode": "quiz",
                "user_message": "Why does this matter for performance?",
                "history": [
                    {"role": "user", "content": "Define reconciliation briefly."},
                    {"role": "assistant", "content": "It is React's tree diff process."},
                ],
            },
        )

        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertIn("reply", payload)
        self.assertEqual(payload["provider_used"], "openai")
        self.assertIn("Topic ID: frontend-react", self.fake.last_prompt)
        self.assertIn("Topic Title: React Rendering", self.fake.last_prompt)
        self.assertIn("Topic Track: frontend", self.fake.last_prompt)
        self.assertIn("Section: Rendering Pipeline", self.fake.last_prompt)
        self.assertIn("Learning Mode: quiz", self.fake.last_prompt)
        self.assertIn("fenced Mermaid diagrams", self.fake.last_prompt)
        self.assertIn("fenced code blocks", self.fake.last_prompt)

    def test_follow_up_accepts_legacy_payload_without_topic_metadata(self):
        res = self.client.post(
            "/api/chat/follow-up",
            json={
                "word": "idempotency",
                "context_question": "What is idempotency?",
                "context_answer": "Repeated requests produce the same outcome.",
                "user_message": "Why is this useful for retries?",
                "history": [],
            },
        )

        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertIn("reply", payload)
        self.assertIn("Topic metadata not provided.", self.fake.last_prompt)


if __name__ == "__main__":
    unittest.main()
