import asyncio
import unittest

from app.services.interview_generator import InterviewGenerator


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if self._responses:
            return {"success": True, "analysis": self._responses.pop(0), "metadata": {}}
        return {"success": True, "analysis": "{}", "metadata": {}}


class FakeParser:
    def list_topics(self, track=None, level=None, q=None):
        return []


class InterviewGeneratorTests(unittest.TestCase):
    def test_retries_duplicate_question_then_returns_unique(self):
        llm = FakeLLM(
            [
                '{"question":"Explain REST.","competency_focus":"fundamentals","expected_signals":["x","y"]}',
                '{"question":"How do you handle API versioning safely?","competency_focus":"api evolution","expected_signals":["compatibility","deprecation"]}',
            ]
        )
        gen = InterviewGenerator(llm, FakeParser())

        session = {
            "session_id": "s1",
            "track": "backend",
            "level": "mid",
            "interview_type": "technical",
            "turn_count": 5,
            "turns_completed": 1,
            "target_role": "Backend Engineer",
            "focus_areas": ["api design"],
            "job_description_text": "",
            "resume_summary_text": "",
            "asked_questions": ["Explain REST."],
        }

        out = asyncio.run(gen.generate_question(session=session, turns=[]))
        self.assertIn("versioning", out["question"].lower())

    def test_eval_payload_repair_after_invalid(self):
        llm = FakeLLM(
            [
                '{"rubric": {"technical_accuracy": 4}, "strengths": [], "improvements": []}',
                '{"rubric":{"technical_accuracy":4,"reasoning_depth":3,"communication_clarity":4,"completeness":3,"confidence_signal":3,"overall":74},"strengths":["Good structure"],"improvements":["Add metrics"],"follow_up_note":"Mention one production tradeoff next time."}',
            ]
        )
        gen = InterviewGenerator(llm, FakeParser())

        session = {
            "session_id": "s2",
            "track": "system_design",
            "level": "senior",
            "interview_type": "system_design",
            "target_role": "Staff Engineer",
        }

        out = asyncio.run(
            gen.evaluate_answer(
                session=session,
                question="Design a rate limiter.",
                user_answer="Use token bucket and Redis.",
                turn_index=1,
            )
        )
        self.assertEqual(out["rubric"]["overall"], 74)
        self.assertGreaterEqual(len(out["strengths"]), 1)

    def test_evaluate_prompt_requests_markdown_follow_up_note(self):
        prompt = InterviewGenerator._evaluate_prompt(
            session={
                "track": "backend",
                "level": "mid",
                "interview_type": "technical",
                "target_role": "Backend Engineer",
            },
            question="How would you design retries?",
            answer="Use bounded retries with idempotency keys.",
            turn_index=1,
        )
        self.assertIn("follow_up_note", prompt)
        self.assertIn("follow_up_note should use adaptive markdown structure", prompt)
        self.assertIn("default to concise coaching prose", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("Keep valid JSON string escaping", prompt)
        self.assertIn("No extra keys.", prompt)

    def test_build_report_summary_is_markdown_ready(self):
        gen = InterviewGenerator(FakeLLM([]), FakeParser())
        report = gen.build_report(
            session={"track": "backend"},
            turns=[
                {
                    "turn_index": 1,
                    "question": "Q1",
                    "user_answer": "A1",
                    "rubric": {
                        "technical_accuracy": 4,
                        "reasoning_depth": 3,
                        "communication_clarity": 4,
                        "completeness": 3,
                        "confidence_signal": 3,
                        "overall": 74,
                    },
                    "strengths": ["Clear structure"],
                    "improvements": ["Add production metric"],
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
        )
        self.assertIn("### Interview Summary", report["summary"])
        self.assertIn("| Dimension | Average |", report["summary"])
        self.assertIn("#### Focus Areas", report["summary"])


if __name__ == "__main__":
    unittest.main()
