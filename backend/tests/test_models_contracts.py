import unittest

from app.schemas.models import (
    CreateCustomTopicRequest,
    CreateInterviewSessionRequest,
    FeedbackModeEnum,
    GenerateQuestionsRequest,
    GenerateQuizRequest,
    GenerateTopicContentRequest,
    InterviewerStyleEnum,
    InterviewTrendsResponse,
    InterviewTypeEnum,
    StudyPlanResponse,
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

    def test_dynamic_topic_content_target_sections_defaults_to_120(self):
        req = GenerateTopicContentRequest(preferred_language="python")
        self.assertEqual(req.target_sections, 120)

    def test_interview_type_supports_coding(self):
        self.assertEqual(InterviewTypeEnum.CODING.value, "coding")

    def test_interview_session_request_defaults_style_and_feedback(self):
        req = CreateInterviewSessionRequest(track="backend")
        self.assertEqual(req.interviewer_style, InterviewerStyleEnum.NEUTRAL)
        self.assertEqual(req.feedback_mode, FeedbackModeEnum.CONCISE)

    def test_study_plan_response_contract(self):
        payload = StudyPlanResponse(
            generated_at="2026-03-07T10:00:00+00:00",
            days=7,
            daily_items=3,
            total_tasks=1,
            days_plan=[
                {
                    "day_index": 1,
                    "label": "Today",
                    "date": "2026-03-07",
                    "tasks": [
                        {
                            "task_type": "review",
                            "topic_id": "topic-a",
                            "title": "Review topic-a",
                            "reason": "Due now",
                            "estimated_minutes": 20,
                            "cta_route": "/topics/topic-a",
                        }
                    ],
                }
            ],
        )
        self.assertEqual(payload.days_plan[0].tasks[0].task_type.value, "review")

    def test_interview_trends_response_contract(self):
        payload = InterviewTrendsResponse(
            points=[
                {
                    "session_id": "is_1",
                    "completed_at": "2026-03-07T10:00:00+00:00",
                    "overall_score": 74,
                    "rubric_averages": {
                        "technical_accuracy": 3.8,
                        "reasoning_depth": 3.5,
                        "communication_clarity": 3.6,
                        "completeness": 3.4,
                        "confidence_signal": 3.2,
                        "overall": 74,
                    },
                    "track": "backend",
                    "level": "mid",
                    "interview_type": "coding",
                    "readiness_label": "On Track",
                }
            ],
            summary={
                "latest_score": 74,
                "previous_score": 0,
                "delta": 0,
                "session_count": 1,
            },
        )
        self.assertEqual(payload.points[0].interview_type.value, "coding")


if __name__ == "__main__":
    unittest.main()
