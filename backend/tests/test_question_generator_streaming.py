import asyncio
import json
import unittest

from app.services.question_generator import QuestionGenerator


def _question_payload(question: str) -> dict:
    return {
        "question": question,
        "answer": (
            "This answer explains the concept in practical terms, maps it to implementation "
            "constraints, and clarifies tradeoffs for interview-ready reasoning."
        ),
        "difficulty": "medium",
        "learning_objective": "Understand and apply the documented concept.",
        "source_section": "Core Concepts",
        "source_quote": "Core concepts define behavior and tradeoffs.",
        "misconception_trap": "Assuming a default without validating constraints.",
        "reasoning_summary": "Start from requirements, then evaluate tradeoffs.",
        "target_level": "mid",
    }


class FakeLLM:
    def __init__(self, analyses: list[str]):
        self._analyses = analyses
        self._idx = 0

    async def completion(self, prompt, llm_config=None, user_identity=None):
        if self._idx < len(self._analyses):
            analysis = self._analyses[self._idx]
        else:
            analysis = self._analyses[-1]
        self._idx += 1
        return {
            "success": True,
            "analysis": analysis,
            "metadata": {"provider": "openai", "model": "gpt-4o-mini"},
            "error": "",
            "error_code": "",
        }


async def _collect_events(generator):
    events = []
    async for event in generator:
        events.append(event)
    return events


class QuestionGeneratorStreamingTests(unittest.TestCase):
    def test_generate_v2_stream_emits_incremental_questions_then_done(self):
        llm = FakeLLM(
            [
                json.dumps([_question_payload("What is idempotency in APIs?")]),
                json.dumps([_question_payload("How do you design retry-safe endpoints?")]),
                json.dumps([_question_payload("When should optimistic concurrency be used?")]),
            ]
        )
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="topic-api",
                    topic_title="API Design",
                    doc_content="API docs content",
                    count=3,
                    level="mid",
                )
            )
        )

        self.assertEqual(events[0]["type"], "start")
        question_events = [e for e in events if e.get("type") == "question"]
        self.assertEqual(len(question_events), 3)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["generated_count"], 3)

    def test_generate_v2_stream_filters_duplicates_and_recovers_full_count(self):
        duplicate = json.dumps([_question_payload("Explain consistency models in distributed systems.")])
        unique = json.dumps([_question_payload("When should quorum reads be preferred over leader-only reads?")])
        llm = FakeLLM([duplicate, duplicate, unique])
        generator = QuestionGenerator(llm)

        events = asyncio.run(
            _collect_events(
                generator.generate_v2_stream(
                    topic_id="topic-consistency",
                    topic_title="Distributed Systems",
                    doc_content="Distributed systems docs content",
                    count=2,
                    level="mid",
                )
            )
        )

        question_events = [e for e in events if e.get("type") == "question"]
        done_event = events[-1]
        self.assertEqual(done_event["type"], "done")
        self.assertEqual(len(question_events), 2)
        self.assertEqual(done_event["generated_count"], 2)
        self.assertGreater(done_event["retries_used"], 0)


if __name__ == "__main__":
    unittest.main()
