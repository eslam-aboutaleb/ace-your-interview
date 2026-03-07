import asyncio
import json
import unittest

from app.services.problem_solving_generator import ProblemSolvingGenerator


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if self.responses:
            return {
                "success": True,
                "analysis": self.responses.pop(0),
                "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
                "error": "",
            }
        return {"success": False, "analysis": "", "metadata": {}, "error": "no_response"}


class ProblemSolvingGeneratorTests(unittest.TestCase):
    def test_generate_topic_with_valid_payload(self):
        payload = {
            "title": "Problem Solving and Algorithms (java)",
            "description": "Java interview roadmap from basics to advanced patterns.",
            "track": "backend",
            "levels": ["junior", "mid", "senior"],
            "sections": [
                {
                    "heading": f"Pattern {i}",
                    "content": (
                        "Identify constraints, derive strategy, analyze complexity, "
                        "and explain edge cases clearly in interviews."
                    ),
                }
                for i in range(1, 121)
            ],
        }
        llm = FakeLLM([json.dumps(payload)])
        generator = ProblemSolvingGenerator(llm)

        topic = asyncio.run(
            generator.generate_topic(preferred_language="java", target_sections=120)
        )
        self.assertEqual(topic.id, "00-problem-solving-and-algorithms")
        self.assertTrue(topic.content_ready)
        self.assertEqual(topic.selected_language, "java")
        self.assertEqual(len(topic.sections), 120)

    def test_generate_topic_falls_back_when_invalid(self):
        llm = FakeLLM(["not-json", "{}", "[]", ""])
        generator = ProblemSolvingGenerator(llm)

        topic = asyncio.run(
            generator.generate_topic(preferred_language="python", target_sections=120)
        )
        self.assertEqual(topic.selected_language, "python")
        self.assertEqual(len(topic.sections), 120)
        self.assertTrue(topic.sections[0]["heading"].startswith("Junior:"))


if __name__ == "__main__":
    unittest.main()

