import unittest

from app.services.question_generator import (
    _build_prompt,
    _build_quiz_prompt,
    _validate_question_item,
    _validate_quiz_item,
)


class QuestionGeneratorValidationTests(unittest.TestCase):
    def test_question_item_requires_grounding_fields(self):
        item = {
            "question": "What does this service do?",
            "answer": "It handles user sessions and validates signed cookies for auth.",
            "difficulty": "medium",
            "learning_objective": "Understand auth session responsibilities.",
            "source_section": "Auth",
            "source_quote": "Validate the session cookie and return user payload.",
            "misconception_trap": "Assuming session validation is done in frontend.",
            "reasoning_summary": "Auth must be enforced server-side per request.",
            "target_level": "mid",
        }
        valid, issue = _validate_question_item(item, "topic-auth", "medium", "mid")
        self.assertTrue(valid, issue)
        self.assertEqual(item["topic_id"], "topic-auth")

    def test_quiz_item_strict_choice_rules(self):
        item = {
            "question": "Which option is correct?",
            "type": "mcq",
            "choices": [
                {"label": "A", "text": "One"},
                {"label": "B", "text": "Two"},
                {"label": "C", "text": "Three"},
                {"label": "D", "text": "Four"},
            ],
            "correct_answer": "C",
            "explanation": "Only one option matches the documented behavior exactly.",
            "difficulty": "hard",
            "topic_id": "topic-1",
            "source_quote": "Only one branch performs strict validation.",
            "reasoning_summary": "Read the decision path and eliminate incorrect branches.",
            "target_level": "senior",
        }
        valid, issue = _validate_quiz_item(
            item,
            allowed_topics={"topic-1"},
            allowed_types={"mcq", "true_false"},
            difficulty="hard",
            level="senior",
        )
        self.assertTrue(valid, issue)

        broken = dict(item)
        broken["choices"] = [
            {"label": "A", "text": "True"},
            {"label": "B", "text": "False"},
        ]
        broken["type"] = "mcq"
        valid2, issue2 = _validate_quiz_item(
            broken,
            allowed_topics={"topic-1"},
            allowed_types={"mcq", "true_false"},
            difficulty="hard",
            level="senior",
        )
        self.assertFalse(valid2)
        self.assertEqual(issue2, "mcq_labels_must_be_abcd")

    def test_question_level_mismatch_rejected(self):
        item = {
            "question": "How would you design retries for this API?",
            "answer": "Use bounded retries with backoff and jitter while keeping idempotency guarantees and observability metrics.",
            "difficulty": "medium",
            "learning_objective": "Evaluate reliability reasoning.",
            "source_section": "Retries",
            "source_quote": "Retries must be bounded to avoid overload.",
            "misconception_trap": "Assuming retries are always safe.",
            "reasoning_summary": "Retries need limits and idempotency.",
            "target_level": "senior",
        }
        valid, issue = _validate_question_item(item, "topic-reliability", "medium", "junior")
        self.assertFalse(valid)
        self.assertEqual(issue, "level_mismatch")

    def test_question_prompt_requires_adaptive_markdown_answers(self):
        prompt = _build_prompt(
            topic_id="topic-api",
            topic_title="API Design",
            doc_content="Use pagination and stable ordering for list endpoints.",
            count=3,
            level="mid",
        )
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("use bullets only when listing steps/checklists/categories", prompt)
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("valid JSON, escape newlines", prompt)
        self.assertIn("Return exactly 3 items", prompt)

    def test_quiz_prompt_requires_adaptive_markdown_explanations(self):
        prompt = _build_quiz_prompt(
            topics_content=[{"id": "api", "title": "API Design", "content": "Use ETags and cache control."}],
            count=3,
            question_types=["mcq"],
            level="mid",
        )
        self.assertIn("short readable paragraphs", prompt)
        self.assertIn("use bullets only when listing steps/checklists/categories", prompt)
        self.assertIn("valid GFM table syntax", prompt)
        self.assertIn("fenced code blocks", prompt)
        self.assertIn("fenced Mermaid diagrams", prompt)
        self.assertIn("valid JSON, escape newlines", prompt)
        self.assertIn("Return exactly 3 items", prompt)


if __name__ == "__main__":
    unittest.main()
