import unittest

from app.services.question_generator import _validate_question_item, _validate_quiz_item


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
        }
        valid, issue = _validate_question_item(item, "topic-auth", "medium")
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
        }
        valid, issue = _validate_quiz_item(
            item,
            allowed_topics={"topic-1"},
            allowed_types={"mcq", "true_false"},
            difficulty="hard",
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
        )
        self.assertFalse(valid2)
        self.assertEqual(issue2, "mcq_labels_must_be_abcd")


if __name__ == "__main__":
    unittest.main()

