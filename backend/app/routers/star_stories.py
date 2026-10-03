"""STAR story bank router — CRUD + suggestions."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import get_settings
from app.dependencies import require_auth
from app.schemas.models import (
    StarStory,
    StarStoryCreateRequest,
    StarStoryListResponse,
    StarStorySuggestionsResponse,
    StarStoryUpdateRequest,
)
from app.services.star_store import StarStore, StoryNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/star-stories", tags=["star-stories"])

_store: StarStore | None = None


def init(store: StarStore):
    global _store
    _store = store


def _ensure_enabled() -> None:
    if not get_settings().enable_interview_plus_v1:
        raise HTTPException(status_code=404, detail="Interview personalization disabled")


def _ensure_ready() -> StarStore:
    if _store is None:
        raise HTTPException(status_code=503, detail="STAR story store not initialised")
    return _store


def _to_story_model(story: dict) -> StarStory:
    return StarStory(
        story_id=str(story.get("story_id", "")),
        title=str(story.get("title", "")),
        situation=str(story.get("situation", "")),
        task=str(story.get("task", "")),
        action=str(story.get("action", "")),
        result=str(story.get("result", "")),
        tags=list(story.get("tags") or []),
        created_at=str(story.get("created_at", "")),
        updated_at=str(story.get("updated_at", "")),
    )


@router.post("", response_model=StarStory, status_code=201)
async def create_story(
    body: StarStoryCreateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    story = store.create_story(
        user["user"],
        title=body.title,
        situation=body.situation,
        task=body.task,
        action=body.action,
        result=body.result,
        tags=body.tags,
    )
    return _to_story_model(story)


@router.get("", response_model=StarStoryListResponse)
async def list_stories(
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    stories = store.list_stories(user["user"])
    return StarStoryListResponse(
        stories=[_to_story_model(s) for s in stories],
        total=len(stories),
    )


@router.get("/suggest", response_model=StarStorySuggestionsResponse)
async def suggest_stories(
    q: str = Query(default="", max_length=2000),
    limit: int = Query(default=3, ge=1, le=10),
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    stories = store.suggest_stories(user["user"], q, limit=limit)
    return StarStorySuggestionsResponse(
        stories=[_to_story_model(s) for s in stories],
    )


@router.get("/{story_id}", response_model=StarStory)
async def get_story(
    story_id: str,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    story = store.get_story(user["user"], story_id)
    if not story:
        raise HTTPException(status_code=404, detail="STAR story not found")
    return _to_story_model(story)


@router.put("/{story_id}", response_model=StarStory)
async def update_story(
    story_id: str,
    body: StarStoryUpdateRequest,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    try:
        story = store.update_story(
            user["user"],
            story_id,
            title=body.title,
            situation=body.situation,
            task=body.task,
            action=body.action,
            result=body.result,
            tags=body.tags,
        )
    except StoryNotFoundError:
        raise HTTPException(status_code=404, detail="STAR story not found")
    return _to_story_model(story)


@router.delete("/{story_id}", status_code=204)
async def delete_story(
    story_id: str,
    user: dict = Depends(require_auth),
):
    _ensure_enabled()
    store = _ensure_ready()
    deleted = store.delete_story(user["user"], story_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="STAR story not found")
    return None
