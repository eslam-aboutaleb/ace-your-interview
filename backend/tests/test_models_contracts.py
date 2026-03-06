import unittest

from app.schemas.models import (
    CreateCustomTopicRequest,
    GenerateQuestionsRequest,
    GenerateQuizRequest,
)


class ModelContractTests(unittest.TestCase):
    def test_generate_questions_level_defaults_to_mid(self):
        req = GenerateQuestionsRequest(topic_id="backend-core")
        self.assertEqual(req.level, "mid")

    def test_generate_quiz_level_defaults_to_mid(self):
        req = GenerateQuizRequest(topic_ids=["backend-core"])
        self.assertEqual(req.level, "mid")

    def test_custom_topic_target_sections_defaults_to_120(self):
        req = CreateCustomTopicRequest(topic="Java")
        self.assertEqual(req.target_sections, 120)


if __name__ == "__main__":
    unittest.main()
