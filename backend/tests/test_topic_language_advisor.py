import asyncio
import json
import unittest

from app.schemas.models import TopicDetail
from app.services.topic_language_advisor import TopicLanguageAdvisor


class FakeLLM:
    def __init__(self, analysis: str, success: bool = True):
        self.analysis = analysis
        self.success = success

    async def completion(self, prompt, llm_config=None, user_identity=None):
        return {
            "success": self.success,
            "analysis": self.analysis,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "" if self.success else "failed",
        }


class TopicLanguageAdvisorTests(unittest.TestCase):
    def test_advisor_accepts_valid_llm_payload(self):
        llm = FakeLLM(
            json.dumps(
                {
                    "requires_programming": True,
                    "language_options": ["TypeScript", "javascript", "Go", "ts"],
                }
            )
        )
        advisor = TopicLanguageAdvisor(llm)
        topic = TopicDetail(
            id="custom-react",
            title="React Interview Roadmap",
            description="Frontend topic",
            track="frontend",
            levels=["junior", "mid", "senior"],
            sections=[{"heading": "State", "content": "State management."}],
            raw_content="# React",
        )
        out = asyncio.run(advisor.advise_topic(topic=topic))
        self.assertTrue(out["requires_programming"])
        self.assertEqual(out["language_options"][:3], ["typescript", "javascript", "go"])
        self.assertEqual(out["source"], "llm")

    def test_advisor_falls_back_when_llm_invalid(self):
        llm = FakeLLM("not-json", success=True)
        advisor = TopicLanguageAdvisor(llm)
        topic = TopicDetail(
            id="topic-system-design",
            title="System Design Foundations",
            description="Distributed systems tradeoffs.",
            track="system_design",
            levels=["junior", "mid", "senior"],
            sections=[{"heading": "Tradeoffs", "content": "CAP and consistency."}],
            raw_content="# SD",
        )
        out = asyncio.run(advisor.advise_topic(topic=topic))
        self.assertIn("requires_programming", out)
        self.assertIn("language_options", out)
        self.assertEqual(out["source"], "fallback")


if __name__ == "__main__":
    unittest.main()
