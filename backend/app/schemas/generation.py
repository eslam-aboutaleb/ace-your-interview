"""Versioned contracts for the generation job orchestration API (issue #16)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

GENERATION_JOBS_API_VERSION = "v1"


class GenerationJobCreateRequest(BaseModel):
    idempotency_key: str = Field(min_length=1, max_length=200)
    outline: list[str] = Field(default_factory=list, max_length=50)


class GenerationJobResponse(BaseModel):
    id: str
    status: str
    idempotency_key: str
    outline: list[str] = Field(default_factory=list)
    accepted: list[dict[str, Any]] = Field(default_factory=list)
    failed: list[dict[str, Any]] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    cost_units: int = 0
    attempt_count: int = 0
    created_at: str = ""
    updated_at: str = ""


class GenerationJobCreateResponse(BaseModel):
    api_version: str = GENERATION_JOBS_API_VERSION
    created: bool
    job: GenerationJobResponse


class GenerationJobGetResponse(BaseModel):
    api_version: str = GENERATION_JOBS_API_VERSION
    job: GenerationJobResponse


class GenerationJobListResponse(BaseModel):
    api_version: str = GENERATION_JOBS_API_VERSION
    jobs: list[GenerationJobResponse] = Field(default_factory=list)
