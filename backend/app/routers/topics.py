"""Topics router — list and retrieve documentation topics."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.models import TopicDetail, TopicSummary
from app.services.doc_parser import DocParser

router = APIRouter(prefix="/api/topics", tags=["topics"])

# Singleton – initialised once on first import (app startup)
_parser: DocParser | None = None


def get_parser() -> DocParser:
    global _parser
    if _parser is None:
        _parser = DocParser()
    return _parser


@router.get("", response_model=list[TopicSummary])
async def list_topics():
    """Return all study topics with metadata."""
    return get_parser().list_topics()


@router.get("/{topic_id}", response_model=TopicDetail)
async def get_topic(topic_id: str):
    """Return full topic content including sections and raw markdown."""
    topic = get_parser().get_topic(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail=f"Topic '{topic_id}' not found")
    return topic
