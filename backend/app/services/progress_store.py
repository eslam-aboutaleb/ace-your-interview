"""SQLite-backed per-topic learner progress: users, asked questions, AI summaries."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar

# The generator's own normaliser, so stored history and client-supplied lists
# de-duplicate against each other exactly the way live generation does.
from app.services.question_generator import _normalise_question as normalise_question_text

T = TypeVar("T")

# Cap on the cumulative per-topic question history. The prompt only ever shows a
# slice of this (see question_generator._MAX_PROMPT_EXISTING_QUESTIONS); the cap
# exists so one long-lived topic cannot grow the row without bound.
MAX_QUESTIONS_ASKED = 200

# Durable invariant for stored summaries: the AI summarizer also caps, this is
# the backstop so no writer can store an unbounded summary.
MAX_SUMMARY_CHARS = 2000


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _loads_list(raw: str) -> list[Any]:
    try:
        data = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return data


def _loads_dict(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class ProgressStore:
    """Persist one cumulative progress document per (user_id, topic_id)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="progress-store")
        # Learning, planner and progress stores share this database file with
        # independent connections, so a busy timeout keeps a concurrent write on
        # another connection from surfacing "database is locked" to the caller.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
        """Run synchronous SQLite work on a dedicated worker thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor,
                lambda: fn(*args, **kwargs),
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL DEFAULT '',
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    session_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_progress (
                    user_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    topic_title TEXT NOT NULL,
                    summary_text TEXT NOT NULL DEFAULT '',
                    summary_status TEXT NOT NULL DEFAULT 'empty',
                    summary_error TEXT NOT NULL DEFAULT '',
                    questions_asked_json TEXT NOT NULL DEFAULT '[]',
                    sections_json TEXT NOT NULL DEFAULT '[]',
                    attempt_stats_json TEXT NOT NULL DEFAULT '{}',
                    preferred_language TEXT NOT NULL DEFAULT '',
                    question_count INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,
                    provider_used TEXT NOT NULL DEFAULT '',
                    model_used TEXT NOT NULL DEFAULT '',
                    generation_source TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, topic_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_topic_progress_user_updated
                ON topic_progress(user_id, updated_at DESC)
                """
            )
            existing_columns = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(topic_progress)").fetchall()
            }
            required_columns = {
                "summary_text": "TEXT NOT NULL DEFAULT ''",
                "summary_status": "TEXT NOT NULL DEFAULT 'empty'",
                "generation_source": "TEXT NOT NULL DEFAULT ''",
                "question_count": "INTEGER NOT NULL DEFAULT 0",
                "revision": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, ddl in required_columns.items():
                if column not in existing_columns:
                    self._conn.execute(
                        f"ALTER TABLE topic_progress ADD COLUMN {column} {ddl}"
                    )

    # ── internals ────────────────────────────────────────────────

    @staticmethod
    def _merge_questions(stored_raw: str, incoming: list[str] | None) -> list[str]:
        """Return a newest-first, de-duplicated, capped question list.

        ``incoming`` arrives in generation order (oldest first) and is reversed
        so the freshest question lands at the head of the stored list; anything
        already recorded is dropped by the shared normaliser.
        """
        merged: list[str] = []
        seen: set[str] = set()
        candidates = [*reversed(list(incoming or [])), *_loads_list(stored_raw)]
        for text in candidates:
            clean = str(text or "").strip()
            if not clean:
                continue
            norm = normalise_question_text(clean)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged.append(clean)
            if len(merged) >= MAX_QUESTIONS_ASKED:
                break
        return merged

    @staticmethod
    def _clean_sections(sections: list[str] | None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in sections or []:
            text = str(raw or "").strip()
            if not text:
                continue
            norm = text.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out.append(text)
        return out

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "user_id": str(row["user_id"]),
            "topic_id": str(row["topic_id"]),
            "topic_title": str(row["topic_title"]),
            "summary_text": str(row["summary_text"]),
            "summary_status": str(row["summary_status"]),
            "summary_error": str(row["summary_error"]),
            "sections": [str(s) for s in _loads_list(row["sections_json"])],
            "attempt_stats": _loads_dict(row["attempt_stats_json"]),
            "preferred_language": str(row["preferred_language"]),
            "question_count": int(row["question_count"]),
            "revision": int(row["revision"]),
            "provider_used": str(row["provider_used"]),
            "model_used": str(row["model_used"]),
            "generation_source": str(row["generation_source"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "questions_asked": [str(q) for q in _loads_list(row["questions_asked_json"])],
        }

    def _read_progress(self, *, user_id: str, topic_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM topic_progress WHERE user_id = ? AND topic_id = ?",
            (user_id, topic_id),
        ).fetchone()
        return self._row_to_document(row) if row else None

    def _read_progress_row(self, *, user_id: str, topic_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM topic_progress WHERE user_id = ? AND topic_id = ?",
            (user_id, topic_id),
        ).fetchone()

    def _touch_user_locked(self, *, user_id: str, provider: str, count_session: bool) -> None:
        now = _utc_now_iso()
        # A blank provider must never overwrite a known one (generate-call capture
        # passes no provider, the save path passes the identity provider), so the
        # conflict branch keeps the stored value.
        self._conn.execute(
            """
            INSERT INTO users(user_id, provider, first_seen_at, last_seen_at, session_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                provider = CASE WHEN excluded.provider != '' THEN excluded.provider ELSE users.provider END,
                last_seen_at = excluded.last_seen_at,
                session_count = users.session_count + ?
            """,
            (
                user_id,
                str(provider or "").strip(),
                now,
                now,
                1 if count_session else 0,
                1 if count_session else 0,
            ),
        )

    # ── writes ───────────────────────────────────────────────────

    def touch_user(self, *, user_id: str, provider: str = "", count_session: bool = False) -> None:
        """Create or refresh the learner row.

        ``count_session`` is only set by the save/autosave path: a saved
        progress document is what counts as a study session, whereas the
        generate-call capture path touches the row on every generation.
        """
        with self._lock:
            with self._conn:
                self._touch_user_locked(
                    user_id=user_id, provider=provider, count_session=count_session
                )

    def upsert_topic_progress(
        self,
        *,
        user_id: str,
        topic_id: str,
        topic_title: str,
        questions: list[str] | None = None,
        sections: list[str] | None = None,
        attempt_stats: dict[str, Any] | None = None,
        preferred_language: str = "",
        provider: str = "",
    ) -> dict[str, Any]:
        """Write the durable raw progress document for a topic.

        Only the raw fields are touched here. The AI summary is owned by
        ``set_summary`` so a summary write can never be lost to a raw write.
        """
        now = _utc_now_iso()
        # A caller that omits `topic_title` must not clobber the stored human
        # title with the opaque topic id, so the ON CONFLICT CASE guard below
        # is only reachable when we pass the *raw* title through. The insert
        # still needs a non-empty value for a brand-new row, hence the
        # `or topic_id` applied only at the parameter site.
        clean_title = str(topic_title or "").strip()
        clean_language = str(preferred_language or "").strip()
        stats = attempt_stats if isinstance(attempt_stats, dict) else {}
        with self._lock:
            with self._conn:
                self._touch_user_locked(
                    user_id=user_id, provider=provider, count_session=True
                )
                existing = self._read_progress_row(user_id=user_id, topic_id=topic_id)
                merged = self._merge_questions(
                    existing["questions_asked_json"] if existing else "[]",
                    questions,
                )
                # Seed a brand-new row with the topic id, but pass '' for an
                # update so the SQL CASE guard preserves the stored title.
                insert_title = clean_title if existing else (clean_title or topic_id)
                self._conn.execute(
                    """
                    INSERT INTO topic_progress(
                        user_id, topic_id, topic_title, summary_text, summary_status,
                        summary_error, questions_asked_json, sections_json,
                        attempt_stats_json, preferred_language, question_count, revision,
                        provider_used, model_used, generation_source, created_at, updated_at
                    ) VALUES (?, ?, ?, '', 'empty', '', ?, ?, ?, ?, ?, 1, '', '', '', ?, ?)
                    ON CONFLICT(user_id, topic_id) DO UPDATE SET
                        topic_title = CASE
                            WHEN excluded.topic_title != '' THEN excluded.topic_title
                            ELSE topic_progress.topic_title
                        END,
                        questions_asked_json = excluded.questions_asked_json,
                        sections_json = excluded.sections_json,
                        attempt_stats_json = excluded.attempt_stats_json,
                        preferred_language = CASE
                            WHEN excluded.preferred_language != '' THEN excluded.preferred_language
                            ELSE topic_progress.preferred_language
                        END,
                        question_count = excluded.question_count,
                        revision = topic_progress.revision + 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        topic_id,
                        insert_title,
                        _dump(merged),
                        _dump(self._clean_sections(sections)),
                        _dump(stats),
                        clean_language,
                        len(merged),
                        now,
                        now,
                    ),
                )
                return self._read_progress(user_id=user_id, topic_id=topic_id) or {}

    def record_asked_questions(
        self,
        *,
        user_id: str,
        topic_id: str,
        topic_title: str = "",
        questions: list[str] | None = None,
        section_title: str = "",
        preferred_language: str = "",
        provider: str = "",
    ) -> dict[str, Any]:
        """Union freshly generated question text into the stored history.

        Zero-frontend-change capture: the router calls this after every
        generation so the server's list is a superset of anything the client
        sends. It deliberately leaves ``summary_status`` untouched — the
        summary is refreshed by the save/autosave path, and flipping to
        ``pending`` on every generation would re-trigger a summary LLM call
        per generation instead of per session.
        """
        now = _utc_now_iso()
        clean_title = str(topic_title or "").strip()
        clean_language = str(preferred_language or "").strip()
        clean_section = str(section_title or "").strip()
        with self._lock:
            with self._conn:
                self._touch_user_locked(
                    user_id=user_id, provider=provider, count_session=False
                )
                existing = self._read_progress_row(user_id=user_id, topic_id=topic_id)
                merged = self._merge_questions(
                    existing["questions_asked_json"] if existing else "[]",
                    questions,
                )
                sections = self._clean_sections(
                    [*_loads_list(existing["sections_json"] if existing else "[]"), clean_section]
                )
                created_at = existing["created_at"] if existing else now
                title_value = clean_title or (existing["topic_title"] if existing else topic_id)
                self._conn.execute(
                    """
                    INSERT INTO topic_progress(
                        user_id, topic_id, topic_title, summary_text, summary_status,
                        summary_error, questions_asked_json, sections_json,
                        attempt_stats_json, preferred_language, question_count, revision,
                        provider_used, model_used, generation_source, created_at, updated_at
                    ) VALUES (?, ?, ?, '', 'empty', '', ?, ?, '{}', ?, ?, 1, '', '', '', ?, ?)
                    ON CONFLICT(user_id, topic_id) DO UPDATE SET
                        topic_title = CASE
                            WHEN excluded.topic_title != '' THEN excluded.topic_title
                            ELSE topic_progress.topic_title
                        END,
                        questions_asked_json = excluded.questions_asked_json,
                        sections_json = excluded.sections_json,
                        preferred_language = CASE
                            WHEN excluded.preferred_language != '' THEN excluded.preferred_language
                            ELSE topic_progress.preferred_language
                        END,
                        question_count = excluded.question_count,
                        revision = topic_progress.revision + 1,
                        updated_at = excluded.updated_at
                    """,
                    (
                        user_id,
                        topic_id,
                        title_value,
                        _dump(merged),
                        _dump(sections),
                        clean_language,
                        len(merged),
                        created_at,
                        now,
                    ),
                )
                return self._read_progress(user_id=user_id, topic_id=topic_id) or {}

    def set_summary(
        self,
        *,
        user_id: str,
        topic_id: str,
        text: str,
        provider_used: str = "",
        model_used: str = "",
        source: str = "ai",
    ) -> dict[str, Any] | None:
        """Store a finished summary (``source`` is ``ai`` or ``fallback``)."""
        return self._update_summary(
            user_id=user_id,
            topic_id=topic_id,
            summary_text=str(text or "")[:MAX_SUMMARY_CHARS],
            summary_status="ready",
            summary_error="",
            provider_used=provider_used,
            model_used=model_used,
            generation_source=source,
        )

    def mark_summary_pending(self, *, user_id: str, topic_id: str) -> dict[str, Any] | None:
        """Flag a summary as owed. Repaired by the next generation for the topic."""
        return self._update_summary(
            user_id=user_id,
            topic_id=topic_id,
            summary_text=None,
            summary_status="pending",
            summary_error="",
        )

    def mark_summary_failed(
        self, *, user_id: str, topic_id: str, error: str
    ) -> dict[str, Any] | None:
        return self._update_summary(
            user_id=user_id,
            topic_id=topic_id,
            summary_text=None,
            summary_status="failed",
            summary_error=str(error or "")[:500],
        )

    def _update_summary(
        self,
        *,
        user_id: str,
        topic_id: str,
        summary_text: str | None,
        summary_status: str,
        summary_error: str,
        provider_used: str = "",
        model_used: str = "",
        generation_source: str = "",
    ) -> dict[str, Any] | None:
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE topic_progress SET
                        summary_text = CASE
                            WHEN ? IS NULL THEN summary_text
                            ELSE ?
                        END,
                        summary_status = ?,
                        summary_error = ?,
                        provider_used = CASE WHEN ? != '' THEN ? ELSE provider_used END,
                        model_used = CASE WHEN ? != '' THEN ? ELSE model_used END,
                        generation_source = CASE WHEN ? != '' THEN ? ELSE generation_source END,
                        updated_at = ?
                    WHERE user_id = ? AND topic_id = ?
                    """,
                    (
                        summary_text,
                        summary_text,
                        summary_status,
                        summary_error,
                        provider_used,
                        provider_used,
                        model_used,
                        model_used,
                        generation_source,
                        generation_source,
                        now,
                        user_id,
                        topic_id,
                    ),
                )
                return self._read_progress(user_id=user_id, topic_id=topic_id)

    # ── reads ────────────────────────────────────────────────────

    def get_progress(self, *, user_id: str, topic_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._read_progress(user_id=user_id, topic_id=topic_id)

    def list_progress(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        capped = max(1, min(int(limit or 50), 200))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM topic_progress
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, capped),
            ).fetchall()
            return [self._row_to_document(row) for row in rows]

    def get_asked_questions(self, *, user_id: str, topic_id: str, limit: int = 200) -> list[str]:
        """Newest-first question texts for prompt injection."""
        capped = max(1, min(int(limit or MAX_QUESTIONS_ASKED), MAX_QUESTIONS_ASKED))
        with self._lock:
            row = self._conn.execute(
                "SELECT questions_asked_json FROM topic_progress WHERE user_id = ? AND topic_id = ?",
                (user_id, topic_id),
            ).fetchone()
            if not row:
                return []
            return [str(q) for q in _loads_list(row["questions_asked_json"])][:capped]
