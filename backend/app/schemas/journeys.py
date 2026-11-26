"""Pydantic contracts for the durable learning-journey API (issue #4).

These models version the journey surface independently of the legacy
``app.schemas.models`` module so later learning-journey stages (typed answer
sections, novelty generation, reader UI) can evolve without breaking older
clients. All identifiers are server-assigned; clients never invent IDs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

JOURNEY_API_VERSION = "v1"


class JourneyStatusEnum(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class JourneySourceTypeEnum(str, Enum):
    STATIC_TOPIC = "static_topic"
    CUSTOM_TOPIC = "custom_topic"


class ObjectiveStatusEnum(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    # Retired objectives were removed from the source curriculum by a replan.
    # They are never deleted so recorded progress is never silently lost.
    RETIRED = "retired"


class BatchStatusEnum(str, Enum):
    PLANNED = "planned"
    GENERATING = "generating"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestionSlotStatusEnum(str, Enum):
    RESERVED = "reserved"
    GENERATED = "generated"
    FAILED = "failed"


class JourneyCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    goal: str = Field(default="", max_length=2000)
    source_type: JourneySourceTypeEnum
    source_topic_id: str = Field(min_length=1, max_length=200)
    level: str = Field(default="", max_length=50)


class JourneySummary(BaseModel):
    id: str
    title: str
    goal: str = ""
    source_type: JourneySourceTypeEnum
    source_topic_id: str = ""
    track: str = ""
    level: str = ""
    status: JourneyStatusEnum = JourneyStatusEnum.ACTIVE
    plan_version: int = 1
    objective_count: int = 0
    batch_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class JourneyListResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    journeys: list[JourneySummary] = Field(default_factory=list)


class JourneyObjective(BaseModel):
    id: str
    module_id: str
    position: int
    title: str
    kind: str = "concept"
    source_section: str = ""
    status: ObjectiveStatusEnum = ObjectiveStatusEnum.PENDING


class JourneyModule(BaseModel):
    id: str
    phase_id: str
    position: int
    title: str
    description: str = ""
    objectives: list[JourneyObjective] = Field(default_factory=list)


class JourneyPhase(BaseModel):
    id: str
    position: int
    title: str
    description: str = ""
    modules: list[JourneyModule] = Field(default_factory=list)


class JourneyDetail(JourneySummary):
    phases: list[JourneyPhase] = Field(default_factory=list)


class JourneyCreateResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    journey: JourneyDetail


class JourneyRefreshResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    journey: JourneyDetail
    plan_changed: bool = False
    added_objectives: int = 0
    retired_objectives: int = 0


class BatchCreateRequest(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=128)
    slot_count: int = Field(default=10, ge=1, le=50)


class JourneyBatch(BaseModel):
    id: str
    journey_id: str
    batch_number: int
    idempotency_key: str = ""
    plan_version: int = 1
    prompt_version: str = ""
    model_version: str = ""
    schema_version: str = ""
    status: BatchStatusEnum = BatchStatusEnum.PLANNED
    accepted_count: int = 0
    rejected_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class BatchCreateResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    batch: JourneyBatch
    # False when the idempotency key replayed an earlier request.
    created: bool = True


class BatchListResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    batches: list[JourneyBatch] = Field(default_factory=list)


class JourneyQuestionSlot(BaseModel):
    id: str
    batch_id: str
    position: int
    objective_id: str = ""
    objective_title: str = ""
    question_ref: str = ""
    status: QuestionSlotStatusEnum = QuestionSlotStatusEnum.RESERVED


class CheckpointState(BaseModel):
    last_batch_id: str = ""
    last_question_id: str = ""
    last_section_id: str = ""
    scroll_offset: float = 0.0
    revision: int = 0
    updated_at: str = ""


class ReaderStateResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    journey_id: str
    checkpoint: CheckpointState
    latest_batch: JourneyBatch | None = None
    questions: list[JourneyQuestionSlot] = Field(default_factory=list)


class CheckpointUpdateRequest(BaseModel):
    expected_revision: int = Field(ge=0)
    mutation_id: str = Field(min_length=1, max_length=128)
    last_batch_id: str | None = Field(default=None, max_length=64)
    last_question_id: str | None = Field(default=None, max_length=200)
    last_section_id: str | None = Field(default=None, max_length=200)
    scroll_offset: float | None = Field(default=None, ge=0.0)


class CheckpointResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    checkpoint: CheckpointState


# ---------------------------------------------------------------------------
# Journey sources (issue #14)
# ---------------------------------------------------------------------------


class JourneySourceKind(str, Enum):
    GITHUB_REPO = "github_repo"
    UPLOADED_DOC = "uploaded_doc"


class JourneySourceAttachRequest(BaseModel):
    kind: JourneySourceKind
    locator: str = Field(min_length=1, max_length=1000)


class JourneySourceResponse(BaseModel):
    id: str
    journey_id: str
    kind: str
    locator: str
    revision_hash: str = ""
    status: str = "indexed"
    file_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class JourneySourceDiff(BaseModel):
    changed: list[str] = Field(default_factory=list)
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)


class JourneySourceIngestResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    source: JourneySourceResponse
    diff: JourneySourceDiff


class JourneySourceListResponse(BaseModel):
    api_version: str = JOURNEY_API_VERSION
    sources: list[JourneySourceResponse] = Field(default_factory=list)
