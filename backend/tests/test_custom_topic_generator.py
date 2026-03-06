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
        self.assertIn("## Java Topic 1", out.raw_content)

    def test_generate_topic_falls_back_when_payload_invalid(self):
        llm = FakeLLM(["not-json", '{"oops":true}', "[]", ""])
        generator = CustomTopicGenerator(llm)

        out = asyncio.run(generator.generate_topic(topic="Kafka", target_sections=120))
        self.assertEqual(out.id, "custom-kafka")
        self.assertEqual(len(out.sections), 120)
        self.assertTrue(out.sections[0]["heading"].startswith("Kafka Module"))
        self.assertTrue(out.description)


if __name__ == "__main__":
    unittest.main()
