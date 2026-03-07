import asyncio
import json
import unittest

from app.services.custom_topic_generator import CustomTopicGenerator


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def completion(self, prompt, llm_config=None, user_identity=None):
        self.calls += 1
        if self.responses:
            return {
                "success": True,
                "analysis": self.responses.pop(0),
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "",
            }
        return {"success": False, "analysis": "", "metadata": {}, "error": "no_response"}


class CustomTopicGeneratorTests(unittest.TestCase):
    def test_generate_topic_uses_valid_payload(self):
        payload = {
            "title": "Java Interview Roadmap",
            "description": "Master Java from fundamentals to architecture.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {
                    "heading": f"Java Topic {i}",
                    "content": "Concise notes for interviews with practical focus.",
                }
                for i in range(1, 101)
            ],
        }
        llm = FakeLLM([json.dumps(payload)])
        generator = CustomTopicGenerator(llm)

        out = asyncio.run(generator.generate_topic(topic="Java", target_sections=100))
        self.assertEqual(out.id, "custom-java")
        self.assertEqual(out.title, "Java Interview Roadmap")
        self.assertEqual(out.track, "backend")
        self.assertEqual(len(out.sections), 100)
        self.assertTrue(out.sections[0]["heading"].startswith("Junior:"))
        self.assertIn("Java Topic", out.sections[0]["heading"])
        self.assertIn("## Junior:", out.raw_content)

    def test_generate_topic_falls_back_when_payload_invalid(self):
        llm = FakeLLM(["not-json", '{"oops":true}', "[]", ""])
        generator = CustomTopicGenerator(llm)

        out = asyncio.run(generator.generate_topic(topic="Kafka", target_sections=120))
        self.assertEqual(out.id, "custom-kafka")
        self.assertEqual(len(out.sections), 120)
        self.assertTrue(out.sections[0]["heading"].startswith("Junior:"))
        self.assertIn("Kafka", out.sections[0]["heading"])
        self.assertTrue(out.description)

    def test_generate_topic_enforces_core_coverage_dimensions(self):
        payload = {
            "title": "Java Interview Roadmap",
            "description": "Java preparation path.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {
                    "heading": f"Java fundamentals lesson {i}",
                    "content": "Fundamental basics and terminology only.",
                }
                for i in range(1, 101)
            ],
        }
        llm = FakeLLM([json.dumps(payload)])
        generator = CustomTopicGenerator(llm)

        out = asyncio.run(generator.generate_topic(topic="Java", target_sections=100))
        headings_corpus = " ".join(str(s.get("heading", "")).lower() for s in out.sections)

        self.assertIn("workflow, tooling, and delivery", headings_corpus)
        self.assertIn("implementation patterns and integration", headings_corpus)
        self.assertIn("debugging and root cause analysis", headings_corpus)
        self.assertIn("testing and quality verification", headings_corpus)
        self.assertIn("performance and scalability", headings_corpus)
        self.assertIn("security and threat mitigation", headings_corpus)
        self.assertIn("reliability and failure recovery", headings_corpus)
        self.assertIn("architecture tradeoffs and system design", headings_corpus)
        self.assertIn("operations, observability, and runbooks", headings_corpus)


if __name__ == "__main__":
    unittest.main()
