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
    "google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "ollama": ["llama3.2", "llama3.1", "mistral", "codellama", "phi3"],
    "github": ["gpt-4o-mini", "gpt-4o"],
}


# ── Request / Response Models ───────────────────────────────
class LLMConfigRequest(BaseModel):
    provider: LLMProviderEnum = LLMProviderEnum.DEFAULT
    model: str = ""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=0, ge=0)


class GenerateQuestionsRequest(BaseModel):
    topic_id: str
    count: int = Field(default=5, ge=1, le=100)
    difficulty: Optional[str] = Field(default=None, pattern="^(easy|medium|hard)$")
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


class TopicSummary(BaseModel):
    id: str
    title: str
    description: str = ""
    section_count: int = 0
    estimated_questions: int = 0


class TopicDetail(BaseModel):
    id: str
    title: str
    description: str = ""
    sections: list[dict[str, str]]  # [{heading, content}]
    raw_content: str = ""


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


# ── Chat Follow-Up Models ────────────────────────────────────
class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatFollowUpRequest(BaseModel):
    word: str
    context_question: str = ""
    context_answer: str = ""
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
    llm_config: Optional[LLMConfigRequest] = None


class GenerateQuizResponse(BaseModel):
    questions: list[QuizQuestion]
    topics_used: list[str] = []
    provider_used: str = ""
    model_used: str = ""


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
