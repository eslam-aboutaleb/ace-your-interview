"""Company interview packs — seeded per-company style configuration.

A pack captures an interviewer style, question tendencies, and
a difficulty bias for a company. Packs are seeded by
``scripts/seed_company_packs.py`` and are extensible by admins
(``upsert``). When a session names a company with no pack, the
generator falls back to the generic style (existing behavior).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class CompanyPackStore:
    """SQLite-backed company pack registry."""

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
                CREATE TABLE IF NOT EXISTS company_packs (
                    company TEXT PRIMARY KEY,
                    track TEXT NOT NULL,
                    style_config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_pack(row: sqlite3.Row) -> dict[str, Any]:
        try:
            style_config = json.loads(row["style_config_json"] or "{}")
        except json.JSONDecodeError:
            style_config = {}
        return {
            "company": row["company"],
            "track": row["track"],
            "style_config": style_config if isinstance(style_config, dict) else {},
            "updated_at": row["updated_at"],
        }

    def get_pack(self, company: str) -> dict[str, Any] | None:
        key = str(company or "").strip().lower()
        if not key:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM company_packs WHERE company = ?",
                (key,),
            ).fetchone()
        return self._row_to_pack(row) if row else None

    def list_packs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM company_packs ORDER BY company ASC",
            ).fetchall()
        return [self._row_to_pack(r) for r in rows]

    def upsert_pack(
        self,
        company: str,
        *,
        track: str,
        style_config: dict[str, Any],
    ) -> dict[str, Any]:
        key = str(company or "").strip().lower()
        if not key:
            raise ValueError("Company name is required")
        clean_track = str(track or "").strip().lower() or "general"
        clean_config = style_config if isinstance(style_config, dict) else {}
        now = _utc_now_iso()
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO company_packs(company, track, style_config_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(company) DO UPDATE SET
                        track = excluded.track,
                        style_config_json = excluded.style_config_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        key,
                        clean_track,
                        json.dumps(clean_config, ensure_ascii=True),
                        now,
                    ),
                )
        saved = self.get_pack(key)
        assert saved is not None
        return saved

    def delete_pack(self, company: str) -> bool:
        key = str(company or "").strip().lower()
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM company_packs WHERE company = ?",
                    (key,),
                )
        return bool(cursor.rowcount)


#: Default packs seeded at launch (top-10 companies). The seed
#: script writes these into the database; they are also used as
#: the in-code fallback set so the API works before seeding.
DEFAULT_COMPANY_PACKS: tuple[dict[str, Any], ...] = (
    {
        "company": "google",
        "track": "general",
        "style_config": {
            "interviewer_style": "neutral",
            "question_tendencies": [
                "system design depth",
                "coding fundamentals with optimal complexity",
                "behavioral questions framed around ambiguity",
            ],
            "difficulty_bias": "mid-to-senior",
            "followup_style": "deep-dive on tradeoffs",
        },
    },
    {
        "company": "amazon",
        "track": "general",
        "style_config": {
            "interviewer_style": "challenging",
            "question_tendencies": [
                "leadership-principle behavioral questions",
                "coding with edge-case rigor",
                "bar-raiser style probing",
            ],
            "difficulty_bias": "mid",
            "followup_style": "dig into failures and ownership",
        },
    },
    {
        "company": "meta",
        "track": "general",
        "style_config": {
            "interviewer_style": "neutral",
            "question_tendencies": [
                "product-sense questions",
                "coding with clear constraints",
                "data-structure fundamentals",
            ],
            "difficulty_bias": "mid",
            "followup_style": "optimize for speed and correctness",
        },
    },
    {
        "company": "microsoft",
        "track": "general",
        "style_config": {
            "interviewer_style": "supportive",
            "question_tendencies": [
                "coding with gradual complexity",
                "system design for cloud scale",
                "collaboration-focused behavioral questions",
            ],
            "difficulty_bias": "junior-to-mid",
            "followup_style": "hint-driven guidance",
        },
    },
    {
        "company": "apple",
        "track": "general",
        "style_config": {
            "interviewer_style": "challenging",
            "question_tendencies": [
                "low-level and performance questions",
                "coding with memory constraints",
                "design for quality bar",
            ],
            "difficulty_bias": "mid-to-senior",
            "followup_style": "probe implementation details",
        },
    },
    {
        "company": "netflix",
        "track": "general",
        "style_config": {
            "interviewer_style": "challenging",
            "question_tendencies": [
                "system design at massive scale",
                "freedom-and-responsibility behavioral questions",
                "coding with performance focus",
            ],
            "difficulty_bias": "senior",
            "followup_style": "push on scale and tradeoffs",
        },
    },
    {
        "company": "stripe",
        "track": "backend",
        "style_config": {
            "interviewer_style": "neutral",
            "question_tendencies": [
                "api design and correctness",
                "distributed systems reasoning",
                "coding with edge-case coverage",
            ],
            "difficulty_bias": "mid-to-senior",
            "followup_style": "explore failure modes",
        },
    },
    {
        "company": "airbnb",
        "track": "general",
        "style_config": {
            "interviewer_style": "supportive",
            "question_tendencies": [
                "product-oriented behavioral questions",
                "coding with real-world constraints",
                "system design for marketplaces",
            ],
            "difficulty_bias": "mid",
            "followup_style": "encourage storytelling",
        },
    },
    {
        "company": "uber",
        "track": "backend",
        "style_config": {
            "interviewer_style": "challenging",
            "question_tendencies": [
                "real-time systems design",
                "coding under concurrency",
                "data-intensive behavioral questions",
            ],
            "difficulty_bias": "mid-to-senior",
            "followup_style": "stress-test under load",
        },
    },
    {
        "company": "linkedin",
        "track": "backend",
        "style_config": {
            "interviewer_style": "neutral",
            "question_tendencies": [
                "distributed systems fundamentals",
                "coding with clarity",
                "growth-oriented behavioral questions",
            ],
            "difficulty_bias": "mid",
            "followup_style": "clarify then deepen",
        },
    },
)
