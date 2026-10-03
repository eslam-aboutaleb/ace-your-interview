"""STAR story bank — user-owned behavioral stories with retrieval.

Stories are stored per user and retrieved by token-overlap
(Jaccard) similarity against a query (the question or the
session's focus areas). Embedding-based retrieval will replace
the Jaccard ranking when the RAG plan's vector store lands.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

#: Number of stories auto-suggested for behavioral sessions.
STAR_SUGGESTION_LIMIT = 3


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_story_id() -> str:
    return f"star_{uuid.uuid4().hex[:12]}"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", str(text or "").lower()).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _normalise(text).split() if len(t) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    return len(intersection) / len(union)


class StoryNotFoundError(Exception):
    """The requested story does not exist for this user."""


class StarStore:
    """CRUD + similarity retrieval for STAR stories."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS star_stories (
                    user_id TEXT NOT NULL, story_id TEXT NOT NULL,
                    title TEXT NOT NULL, situation TEXT NOT NULL, task TEXT NOT NULL,
                    action TEXT NOT NULL, result TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, story_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_star_stories_user
                ON star_stories(user_id, updated_at DESC)
                """
            )

    @staticmethod
    def _dumps(items: list[str]) -> str:
        return json.dumps([str(x) for x in items], ensure_ascii=True)

    @staticmethod
    def _loads_list(raw: str) -> list[str]:
        try:
            data = json.loads(raw or "[]")
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        return [str(x).strip() for x in data if str(x).strip()]

    def _row_to_story(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "story_id": row["story_id"],
            "title": row["title"],
            "situation": row["situation"],
            "task": row["task"],
            "action": row["action"],
            "result": row["result"],
            "tags": self._loads_list(row["tags_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def list_stories(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM star_stories WHERE user_id = ? "
                "ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [self._row_to_story(r) for r in rows]

    def get_story(self, user_id: str, story_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM star_stories WHERE user_id = ? AND story_id = ?",
                (user_id, story_id),
            ).fetchone()
        return self._row_to_story(row) if row else None

    def create_story(
        self,
        user_id: str,
        *,
        title: str,
        situation: str,
        task: str,
        action: str,
        result: str,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        story_id = new_story_id()
        now = _utc_now_iso()
        clean_tags = [str(t).strip()[:60] for t in (tags or []) if str(t).strip()][:20]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO star_stories(
                        user_id, story_id, title, situation, task, action,
                        result, tags_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        story_id,
                        title.strip()[:200],
                        situation.strip()[:5000],
                        task.strip()[:5000],
                        action.strip()[:5000],
                        result.strip()[:5000],
                        self._dumps(clean_tags),
                        now,
                        now,
                    ),
                )
        saved = self.get_story(user_id, story_id)
        assert saved is not None
        return saved

    def update_story(
        self,
        user_id: str,
        story_id: str,
        *,
        title: str | None = None,
        situation: str | None = None,
        task: str | None = None,
        action: str | None = None,
        result: str | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        existing = self.get_story(user_id, story_id)
        if not existing:
            raise StoryNotFoundError(f"Story {story_id} not found")

        next_title = title.strip()[:200] if title is not None else existing["title"]
        next_situation = (
            situation.strip()[:5000] if situation is not None else existing["situation"]
        )
        next_task = task.strip()[:5000] if task is not None else existing["task"]
        next_action = action.strip()[:5000] if action is not None else existing["action"]
        next_result = result.strip()[:5000] if result is not None else existing["result"]
        next_tags = (
            [str(t).strip()[:60] for t in tags if str(t).strip()][:20]
            if tags is not None
            else existing["tags"]
        )
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE star_stories
                    SET title = ?, situation = ?, task = ?, action = ?,
                        result = ?, tags_json = ?, updated_at = ?
                    WHERE user_id = ? AND story_id = ?
                    """,
                    (
                        next_title,
                        next_situation,
                        next_task,
                        next_action,
                        next_result,
                        self._dumps(next_tags),
                        now,
                        user_id,
                        story_id,
                    ),
                )
        saved = self.get_story(user_id, story_id)
        assert saved is not None
        return saved

    def delete_story(self, user_id: str, story_id: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM star_stories WHERE user_id = ? AND story_id = ?",
                    (user_id, story_id),
                )
        return bool(cursor.rowcount)

    def suggest_stories(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = STAR_SUGGESTION_LIMIT,
    ) -> list[dict[str, Any]]:
        """Top stories by Jaccard similarity to ``query``.

        Falls back to most-recently-updated stories when the
        query shares no tokens with any story.
        """
        stories = self.list_stories(user_id)
        if not stories:
            return []

        query_tokens = _tokens(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for story in stories:
            story_tokens: set[str] = set()
            for field in ("title", "situation", "task", "action", "result"):
                story_tokens |= _tokens(str(story.get(field) or ""))
            story_tokens |= _tokens(" ".join(story.get("tags") or []))
            scored.append((_jaccard(query_tokens, story_tokens), story))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [story for score, story in scored if score > 0][:limit]
        if not top:
            # No lexical overlap: suggest the most recent stories.
            top = stories[:limit]
        return top
