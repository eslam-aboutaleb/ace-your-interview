"""Pydantic schemas for the study-app API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── LLM Provider Enum ───────────────────────────────────────
class LLMProviderEnum(str, Enum):
    DEFAULT = "default"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    GITHUB = "github"


class TrackEnum(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    SYSTEM_DESIGN = "system_design"
    AI_STACK = "ai_stack"


class LevelEnum(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class InterviewTypeEnum(str, Enum):
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    SYSTEM_DESIGN = "system_design"
    AI_FUNDAMENTALS = "ai_fundamentals"
    CODING = "coding"
    MIXED = "mixed"


class InterviewerStyleEnum(str, Enum):
    SUPPORTIVE = "supportive"
    NEUTRAL = "neutral"
    CHALLENGING = "challenging"


class FeedbackModeEnum(str, Enum):
    CONCISE = "concise"
    DEEP = "deep"


class ResponseDetailEnum(str, Enum):
    CONCISE = "concise"
    VERY_DETAILED = "very_detailed"


# Maps REST enum string → proto int
PROVIDER_TO_PROTO: dict[str, int] = {
    "default": 0,
    "openai": 1,
    "anthropic": 2,
    "google": 3,
    "groq": 4,
    "ollama": 5,
    "github": 6,
}

MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "google": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
    ],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "ollama": ["llama3.2", "llama3.1", "mistral", "codellama", "phi3"],
    "github": ["gpt-4o-mini", "gpt-4o"],
}
USER_SETTINGS_PROVIDER_WHITELIST = ["google", "openai", "anthropic", "groq"]


# ── Request / Response Models ───────────────────────────────
class LLMConfigRequest(BaseModel):
    provider: LLMProviderEnum = LLMProviderEnum.DEFAULT
    model: str = ""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=0, ge=0)


class GenerateQuestionsRequest(BaseModel):
    topic_id: str
    count: int = Field(default=5, ge=1, le=100)
    requested_total_count: Optional[int] = Field(default=None, ge=1, le=100)
    existing_questions: list[str] = Field(default_factory=list, max_length=200)
    difficulty: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")
    level: Optional[str] = Field(default="mid", pattern="^(junior|mid|senior)$")
    response_detail: Optional[ResponseDetailEnum] = None
    preferred_language: Optional[str] = Field(default=None, max_length=60)
    llm_config: Optional[LLMConfigRequest] = None
    section_title: Optional[str] = None
    section_content: Optional[str] = None


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    difficulty: str = "medium"


class GenerateQuestionsResponse(BaseModel):
    topic_id: str
    topic_title: str
    questions: list[QuestionAnswer]
    provider_used: str = ""
    model_used: str = ""


class QuestionAnswerV2(BaseModel):
    question_id: str
    topic_id: str
    question: str
    answer: str
    difficulty: str = "medium"
    learning_objective: str = ""
    source_section: str = ""
    source_quote: str = ""
    misconception_trap: str = ""
    reasoning_summary: str = ""


class GenerateQuestionsV2Response(BaseModel):
    topic_id: str
    topic_title: str
    questions: list[QuestionAnswerV2]
    provider_used: str = ""
    model_used: str = ""
    retries_used: int = 0
    malformed_items_dropped: int = 0


class TopicSummary(BaseModel):
    id: str
    title: str
    description: str = ""
    track: str = ""
    levels: list[str] = Field(default_factory=list)
    section_count: int = 0
    estimated_questions: int = 0


class TopicDetail(BaseModel):
    id: str
    title: str
    description: str = ""
    track: str = ""
    levels: list[str] = Field(default_factory=list)
    sections: list[dict[str, str]]  # [{heading, content}]
    raw_content: str = ""
    requires_programming: bool = False
    language_options: list[str] = Field(default_factory=list)
    selected_language: str = ""
    response_detail: ResponseDetailEnum = ResponseDetailEnum.CONCISE
    is_dynamic_topic: bool = False
    content_ready: bool = True


class CreateCustomTopicRequest(BaseModel):
    topic: str = Field(..., min_length=2, max_length=120)
    target_sections: int = Field(default=120, ge=100, le=150)
    llm_config: Optional[LLMConfigRequest] = None


class GenerateTopicContentRequest(BaseModel):
    preferred_language: str = Field(..., min_length=2, max_length=40)
    target_sections: int = Field(default=120, ge=100, le=150)
    force_regenerate: bool = False
    llm_config: Optional[LLMConfigRequest] = None


class TopicPreferencesResponse(BaseModel):
    topic_id: str
    response_detail: ResponseDetailEnum = ResponseDetailEnum.CONCISE
    preferred_language: str = ""
    requires_programming: bool = False
    language_options: list[str] = Field(default_factory=list)


class TopicPreferencesUpdateRequest(BaseModel):
    response_detail: Optional[ResponseDetailEnum] = None
    preferred_language: Optional[str] = Field(default=None, max_length=60)


class ProviderStatus(BaseModel):
    name: str
    available: bool
    models: list[str] = []
    backend: str = ""  # "llm-chain" or "cli-agent"


class LLMProvidersResponse(BaseModel):
    providers: list[ProviderStatus]


class HealthStatus(BaseModel):
    llm_chain: bool = False
    cli_agent: bool = False
    llm_chain_version: str = ""
    cli_agent_version: str = ""


class UserAuthModeEnum(str, Enum):
    API_KEY = "api_key"
    ACCOUNT = "account"


class UserLLMSourceEnum(str, Enum):
    PERSONAL = "personal"
    STUDY_APP = "study_app"


class UserPreferences(BaseModel):
    provider: str = Field(default="openai", pattern="^(google|openai|anthropic|groq)$")
    model: str = ""
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=0, ge=0)
    auth_mode: UserAuthModeEnum = UserAuthModeEnum.API_KEY
    llm_source: UserLLMSourceEnum = UserLLMSourceEnum.PERSONAL
    require_answer_reveal: bool = False


class UserPreferencesUpdateRequest(UserPreferences):
    pass


class UserApiKeyUpdateRequest(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=500)


class ProviderConnectionStatus(BaseModel):
    provider: str
    models: list[str] = []
    supports_account_connect: bool = False
    api_key_connected: bool = False
    account_connected: bool = False
    using_backend_fallback: bool = False
    backend_fallback_eligible: bool = False


class StudyAppAssignment(BaseModel):
    provider: str
    model: str
    updated_at: str = ""


class UserSettingsResponse(BaseModel):
    has_saved_preferences: bool = False
    preferences: UserPreferences
    providers: list[ProviderConnectionStatus] = []
    study_app_available: bool = False
    study_app_assignment: StudyAppAssignment | None = None


class LLMUserAssignmentUpdateRequest(BaseModel):
    provider: str = Field(..., pattern="^(google|openai|anthropic|groq)$")
    model: str = Field(default="", min_length=1, max_length=200)


class LLMUserAssignmentItem(BaseModel):
    login_provider: str
    identifier: str
    identity_key: str
    is_backend_approved: bool = False
    assignment: StudyAppAssignment | None = None


class LLMUserAssignmentsResponse(BaseModel):
    users: list[LLMUserAssignmentItem] = []
    provider_models: dict[str, list[str]] = {}


class LLMMyAssignmentResponse(BaseModel):
    login_provider: str
    identifier: str
    identity_key: str
    is_backend_approved: bool = False
    assignment: StudyAppAssignment | None = None
    provider_models: dict[str, list[str]] = {}


# ── Chat Follow-Up Models ────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatFollowUpRequest(BaseModel):
    word: str
    context_question: str = ""
    context_answer: str = ""
    topic_id: str = ""
    topic_title: str = ""
    topic_track: str = ""
    section_title: str = ""
    mode: str = ""
    response_detail: Optional[ResponseDetailEnum] = None
    preferred_language: Optional[str] = Field(default=None, max_length=60)
    requires_programming: Optional[bool] = None
    user_message: str
    history: list[ChatMessage] = []
    llm_config: Optional[LLMConfigRequest] = None


class ChatFollowUpResponse(BaseModel):
    reply: str
    provider_used: str = ""
    model_used: str = ""


# ── Quiz Models ──────────────────────────────────────────────
class QuizQuestionType(str, Enum):
    MCQ = "mcq"
    TRUE_FALSE = "true_false"


class QuizChoice(BaseModel):
    label: str
    text: str


class QuizQuestion(BaseModel):
    question: str
    type: QuizQuestionType
    choices: list[QuizChoice]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"
    topic_id: str = ""


class GenerateQuizRequest(BaseModel):
    topic_ids: list[str]
    count: int = Field(default=10, ge=1, le=100)
    question_types: list[QuizQuestionType] = [QuizQuestionType.MCQ, QuizQuestionType.TRUE_FALSE]
    difficulty: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")
    level: Optional[str] = Field(default="mid", pattern="^(junior|mid|senior)$")
    response_detail: Optional[ResponseDetailEnum] = None
    preferred_language: Optional[str] = Field(default=None, max_length=60)
    llm_config: Optional[LLMConfigRequest] = None


class GenerateQuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topics_used: list[str] = []
    provider_used: str = ""
    model_used: str = ""


class QuizQuestionV2(BaseModel):
    question_id: str
    question: str
    type: QuizQuestionType
    choices: list[QuizChoice]
    correct_answer: str
    explanation: str
    difficulty: str = "medium"
    topic_id: str = ""
    source_quote: str = ""
    reasoning_summary: str = ""


class GenerateQuizV2Response(BaseModel):
    questions: list[QuizQuestionV2]
    topics_used: list[str] = []
    provider_used: str = ""
    model_used: str = ""
    retries_used: int = 0
    malformed_items_dropped: int = 0


class LearningMode(str, Enum):
    STUDY = "study"
    QUIZ = "quiz"


class LearningAttemptRequest(BaseModel):
    question_id: str = Field(..., min_length=2, max_length=200)
    topic_id: str = Field(..., min_length=1, max_length=200)
    user_answer: str = Field(default="", max_length=8000)
    is_correct: bool
    confidence: int = Field(default=3, ge=1, le=5)
    response_time_ms: int = Field(default=0, ge=0)
    mode: LearningMode


class LearningAttemptResponse(BaseModel):
    attempt_id: int
    topic_id: str
    question_id: str
    mastery_score: float
    due_at: str
    review_bucket: int


class ReviewQueueItem(BaseModel):
    topic_id: str
    question_id: str
    due_at: str
    mastery_score: float
    last_confidence: int
    review_bucket: int
    attempts: int


class ReviewQueueResponse(BaseModel):
    items: list[ReviewQueueItem]
    total_due: int = 0


class WeakAreaItem(BaseModel):
    topic_id: str
    attempts: int
    accuracy: float
    avg_confidence: float
    mastery_score: float
    due_count: int


class WeakAreasResponse(BaseModel):
    weak_areas: list[WeakAreaItem]


class TopicMasteryItem(BaseModel):
    topic_id: str
    mastery_score: float
    attempts: int


class TopicMasteryResponse(BaseModel):
    topics: list[TopicMasteryItem]


class StudyPlanTaskType(str, Enum):
    REVIEW = "review"
    TOPIC_STUDY = "topic_study"
    QUIZ = "quiz"


class StudyPlanTask(BaseModel):
    task_type: StudyPlanTaskType
    topic_id: str
    title: str
    reason: str
    estimated_minutes: int = Field(default=20, ge=5, le=120)
    cta_route: str


class StudyPlanDay(BaseModel):
    day_index: int = Field(ge=1, le=31)
    label: str
    date: str
    tasks: list[StudyPlanTask] = Field(default_factory=list)


class StudyPlanResponse(BaseModel):
    generated_at: str
    days: int = Field(default=7, ge=1, le=31)
    daily_items: int = Field(default=3, ge=1, le=10)
    total_tasks: int = 0
    days_plan: list[StudyPlanDay] = Field(default_factory=list)


# ── Mock Interview Models ────────────────────────────────────
class RubricScore(BaseModel):
    technical_accuracy: int = Field(default=3, ge=0, le=5)
    reasoning_depth: int = Field(default=3, ge=0, le=5)
    communication_clarity: int = Field(default=3, ge=0, le=5)
    completeness: int = Field(default=3, ge=0, le=5)
    confidence_signal: int = Field(default=3, ge=0, le=5)
    overall: int = Field(default=60, ge=0, le=100)


class RubricAverages(BaseModel):
    technical_accuracy: float = 0.0
    reasoning_depth: float = 0.0
    communication_clarity: float = 0.0
    completeness: float = 0.0
    confidence_signal: float = 0.0
    overall: float = 0.0


class InterviewSession(BaseModel):
    session_id: str
    track: TrackEnum
    level: LevelEnum
    interview_type: InterviewTypeEnum
    turn_count: int
    turns_completed: int
    status: str
    target_role: str = ""
    interviewer_style: InterviewerStyleEnum = InterviewerStyleEnum.NEUTRAL
    feedback_mode: FeedbackModeEnum = FeedbackModeEnum.CONCISE
    focus_areas: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    current_question: str = ""
    report_ready: bool = False


class InterviewTurn(BaseModel):
    session_id: str
    turn_index: int
    question: str
    user_answer: str = ""
    rubric: RubricScore
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    follow_up_note: str = ""
    response_time_ms: int = 0
    created_at: str


class InterviewReport(BaseModel):
    session_id: str
    overall_score: float
    readiness_label: str
    completed_turns: int
    rubric_averages: RubricAverages
    weak_competencies: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    recommended_topic_ids: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: str
    updated_at: str


class CreateInterviewSessionRequest(BaseModel):
    track: TrackEnum
    level: LevelEnum = LevelEnum.MID
    interview_type: InterviewTypeEnum = InterviewTypeEnum.MIXED
    turn_count: int = Field(default=5, ge=1, le=20)
    target_role: str = Field(default="", max_length=200)
    interviewer_style: InterviewerStyleEnum = InterviewerStyleEnum.NEUTRAL
    feedback_mode: FeedbackModeEnum = FeedbackModeEnum.CONCISE
    job_description_text: str = Field(default="", max_length=20000)
    resume_summary_text: str = Field(default="", max_length=20000)
    focus_areas: list[str] = Field(default_factory=list, max_length=30)
    llm_config: Optional[LLMConfigRequest] = None


class SubmitInterviewAnswerRequest(BaseModel):
    user_answer: str = Field(..., min_length=1, max_length=20000)
    response_time_ms: int = Field(default=0, ge=0)
    llm_config: Optional[LLMConfigRequest] = None


class NextInterviewQuestionRequest(BaseModel):
    llm_config: Optional[LLMConfigRequest] = None


class InterviewSessionResponse(BaseModel):
    session: InterviewSession
    turns: list[InterviewTurn] = Field(default_factory=list)


class InterviewTurnResponse(BaseModel):
    session: InterviewSession
    turn: InterviewTurn
    report_ready: bool = False


class InterviewQuestionResponse(BaseModel):
    session_id: str
    turn_index: int
    question: str
    competency_focus: str = ""
    expected_signals: list[str] = Field(default_factory=list)


class InterviewReportResponse(BaseModel):
    session: InterviewSession
    report: InterviewReport


class InterviewSessionsListResponse(BaseModel):
    sessions: list[InterviewSession] = Field(default_factory=list)
    total: int = 0


class InterviewStatsResponse(BaseModel):
    total_sessions: int = 0
    completed_sessions: int = 0
    interview_readiness_score: float = 0.0


class InterviewTrendPoint(BaseModel):
    session_id: str
    completed_at: str
    overall_score: float = 0.0
    rubric_averages: RubricAverages = Field(default_factory=RubricAverages)
    track: TrackEnum
    level: LevelEnum
    interview_type: InterviewTypeEnum
    readiness_label: str = ""


class InterviewTrendsSummary(BaseModel):
    latest_score: float = 0.0
    previous_score: float = 0.0
    delta: float = 0.0
    session_count: int = 0


class InterviewTrendsResponse(BaseModel):
    points: list[InterviewTrendPoint] = Field(default_factory=list)
    summary: InterviewTrendsSummary = Field(default_factory=InterviewTrendsSummary)


# ── Ollama Models ────────────────────────────────────────────
class OllamaModelInfo(BaseModel):
    name: str
    size: str = ""
    modified_at: str = ""


class OllamaModelsResponse(BaseModel):
    connected: bool = False
    downloaded_models: list[OllamaModelInfo] = []
    cloud_models: list[str] = []


class OllamaTestResponse(BaseModel):
    connected: bool = False
    version: str = ""
