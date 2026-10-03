"""SQLite-backed flashcard deck and card persistence.

Follows the :class:`~app.services.document_store.DocumentStore`
pattern: a dedicated SQLite connection, a single-thread executor,
and an async ``run_async`` shim so FastAPI handlers never block
the event loop.

Every card row links to an ``fsrs_cards`` row (``fsrs_card_id``)
so deck cards participate in FSRS-4.5 scheduling with
``source_type='flashcard'``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar
from uuid import uuid4

from app.services import fsrs_scheduler

T = TypeVar("T")

DECK_NAME_MAX = 200
DECK_DESCRIPTION_MAX = 2000
CARD_FRONT_MAX = 8000
CARD_BACK_MAX = 16000
CARD_TAGS_MAX = 20
SOURCE_REF_MAX = 400


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _from_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class CardStore:
    """Persist decks, cards, and their FSRS linkage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="card-store"
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
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

    # ── Schema ─────────────────────────────────────────
    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decks (
                    user_id TEXT NOT NULL, deck_id TEXT NOT NULL,
                    name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, deck_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    user_id TEXT NOT NULL, card_id TEXT NOT NULL,
                    deck_id TEXT NOT NULL, front TEXT NOT NULL, back TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    fsrs_card_id TEXT NOT NULL,
                    suspended INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, card_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cards_deck ON cards(user_id, deck_id)"
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cards_fsrs
                ON cards(user_id, fsrs_card_id)
                """
            )

    # ── Row mapping ────────────────────────────────────
    @staticmethod
    def _deck_row_to_dict(row: sqlite3.Row) -> dict:
        return {
            "deck_id": str(row["deck_id"]),
            "name": str(row["name"]),
            "description": str(row["description"]),
            "card_count": int(row["card_count"]) if row["card_count"] is not None else 0,
            "due_count": int(row["due_count"]) if row["due_count"] is not None else 0,
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _card_row_to_dict(row: sqlite3.Row) -> dict:
        tags_raw = row["tags_json"] if "tags_json" in row.keys() else "[]"
        try:
            tags = json.loads(str(tags_raw or "[]"))
            if not isinstance(tags, list):
                tags = []
        except json.JSONDecodeError:
            tags = []
        return {
            "card_id": str(row["card_id"]),
            "deck_id": str(row["deck_id"]),
            "front": str(row["front"]),
            "back": str(row["back"]),
            "source_ref": str(row["source_ref"] or ""),
            "tags": [str(t) for t in tags],
            "fsrs_card_id": str(row["fsrs_card_id"]),
            "suspended": int(row["suspended"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _fsrs_columns() -> str:
        return (
            "f.state, f.stability, f.difficulty, f.due_at, f.last_review_at, "
            "f.reps, f.lapses, f.scheduled_days, f.elapsed_days, f.suspended"
        )

    def _card_with_fsrs_row_to_dict(self, row: sqlite3.Row) -> dict:
        data = self._card_row_to_dict(row)
        data["fsrs"] = {
            "state": str(row["state"]),
            "stability": (
                float(row["stability"]) if row["stability"] is not None else None
            ),
            "difficulty": (
                float(row["difficulty"]) if row["difficulty"] is not None else None
            ),
            "due_at": str(row["due_at"]),
            "last_review_at": (
                str(row["last_review_at"]) if row["last_review_at"] is not None else None
            ),
            "reps": int(row["reps"]),
            "lapses": int(row["lapses"]),
            "scheduled_days": int(row["scheduled_days"]),
            "elapsed_days": int(row["elapsed_days"]),
            "suspended": int(row["suspended"]),
        }
        return data

    # ── Decks ──────────────────────────────────────────
    def create_deck(
        self, *, user_id: str, name: str, description: str = ""
    ) -> dict:
        deck_id = uuid4().hex
        now_iso = _to_iso(_utc_now())
        clean_name = str(name).strip()[:DECK_NAME_MAX]
        clean_description = str(description or "").strip()[:DECK_DESCRIPTION_MAX]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO decks(user_id, deck_id, name, description, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, deck_id, clean_name, clean_description, now_iso, now_iso),
                )
        deck = self.get_deck(user_id=user_id, deck_id=deck_id)
        assert deck is not None
        return deck

    def list_decks(self, *, user_id: str) -> list[dict]:
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            """
            SELECT d.deck_id, d.name, d.description, d.created_at, d.updated_at,
                (SELECT COUNT(*) FROM cards c
                 WHERE c.user_id = d.user_id AND c.deck_id = d.deck_id) AS card_count,
                (SELECT COUNT(*) FROM cards c
                 JOIN fsrs_cards f ON f.user_id = c.user_id AND f.card_id = c.fsrs_card_id
                 WHERE c.user_id = d.user_id AND c.deck_id = d.deck_id
                   AND c.suspended = 0 AND f.suspended = 0 AND f.due_at <= ?) AS due_count
            FROM decks d
            WHERE d.user_id = ?
            ORDER BY d.updated_at DESC, d.deck_id ASC
            """,
            (now_iso, user_id),
        ).fetchall()
        return [self._deck_row_to_dict(row) for row in rows]

    def get_deck(self, *, user_id: str, deck_id: str) -> dict | None:
        now_iso = _to_iso(_utc_now())
        row = self._conn.execute(
            """
            SELECT d.deck_id, d.name, d.description, d.created_at, d.updated_at,
                (SELECT COUNT(*) FROM cards c
                 WHERE c.user_id = d.user_id AND c.deck_id = d.deck_id) AS card_count,
                (SELECT COUNT(*) FROM cards c
                 JOIN fsrs_cards f ON f.user_id = c.user_id AND f.card_id = c.fsrs_card_id
                 WHERE c.user_id = d.user_id AND c.deck_id = d.deck_id
                   AND c.suspended = 0 AND f.suspended = 0 AND f.due_at <= ?) AS due_count
            FROM decks d
            WHERE d.user_id = ? AND d.deck_id = ?
            """,
            (now_iso, user_id, deck_id),
        ).fetchone()
        return self._deck_row_to_dict(row) if row else None

    def update_deck(
        self, *, user_id: str, deck_id: str, name: str, description: str
    ) -> dict | None:
        now_iso = _to_iso(_utc_now())
        clean_name = str(name).strip()[:DECK_NAME_MAX]
        clean_description = str(description or "").strip()[:DECK_DESCRIPTION_MAX]
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE decks SET name = ?, description = ?, updated_at = ?
                    WHERE user_id = ? AND deck_id = ?
                    """,
                    (clean_name, clean_description, now_iso, user_id, deck_id),
                )
        if cur.rowcount == 0:
            return None
        deck = self.get_deck(user_id=user_id, deck_id=deck_id)
        assert deck is not None
        return deck

    def delete_deck(self, *, user_id: str, deck_id: str) -> bool:
        with self._lock:
            with self._conn:
                card_ids = [
                    row["card_id"]
                    for row in self._conn.execute(
                        "SELECT card_id FROM cards WHERE user_id = ? AND deck_id = ?",
                        (user_id, deck_id),
                    ).fetchall()
                ]
                fsrs_ids = [
                    row["fsrs_card_id"]
                    for row in self._conn.execute(
                        "SELECT fsrs_card_id FROM cards WHERE user_id = ? AND deck_id = ?",
                        (user_id, deck_id),
                    ).fetchall()
                ]
                if card_ids:
                    placeholders = ",".join("?" * len(card_ids))
                    self._conn.execute(
                        f"DELETE FROM cards WHERE user_id = ? AND card_id IN ({placeholders})",
                        [user_id, *card_ids],
                    )
                if fsrs_ids:
                    placeholders = ",".join("?" * len(fsrs_ids))
                    self._conn.execute(
                        f"DELETE FROM fsrs_review_log WHERE user_id = ? AND card_id IN ({placeholders})",
                        [user_id, *fsrs_ids],
                    )
                    self._conn.execute(
                        f"DELETE FROM fsrs_cards WHERE user_id = ? AND card_id IN ({placeholders})",
                        [user_id, *fsrs_ids],
                    )
                cur = self._conn.execute(
                    "DELETE FROM decks WHERE user_id = ? AND deck_id = ?",
                    (user_id, deck_id),
                )
        return cur.rowcount > 0

    # ── Cards ──────────────────────────────────────────
    def create_card(
        self,
        *,
        user_id: str,
        deck_id: str,
        front: str,
        back: str,
        tags: list[str] | None = None,
        source_ref: str = "",
        now: datetime | None = None,
    ) -> dict:
        """Create a card and its linked ``fsrs_cards`` row."""
        card_id = uuid4().hex
        fsrs_card_id = uuid4().hex
        moment = now or _utc_now()
        now_iso = _to_iso(moment)
        clean_front = str(front).strip()[:CARD_FRONT_MAX]
        clean_back = str(back).strip()[:CARD_BACK_MAX]
        clean_tags = [str(t).strip()[:60] for t in (tags or []) if str(t).strip()][
            :CARD_TAGS_MAX
        ]
        clean_source_ref = str(source_ref or "").strip()[:SOURCE_REF_MAX]
        tags_json = json.dumps(clean_tags, ensure_ascii=True)
        fresh_fsrs = fsrs_scheduler.create_card(moment)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO cards(
                        user_id, card_id, deck_id, front, back, source_ref,
                        tags_json, fsrs_card_id, suspended, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                    """,
                    (
                        user_id, card_id, deck_id, clean_front, clean_back,
                        clean_source_ref, tags_json, fsrs_card_id, now_iso, now_iso,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO fsrs_cards(
                        user_id, card_id, topic_id, source_type, state,
                        stability, difficulty, due_at, last_review_at,
                        reps, lapses, scheduled_days, elapsed_days,
                        suspended, created_at, updated_at
                    ) VALUES (?, ?, ?, 'flashcard', ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, ?, ?)
                    """,
                    (
                        user_id, fsrs_card_id, deck_id,
                        fresh_fsrs["state"], fresh_fsrs["stability"],
                        fresh_fsrs["difficulty"], fresh_fsrs["due_at"],
                        fresh_fsrs["last_review_at"], now_iso, now_iso,
                    ),
                )
        card = self.get_card(user_id=user_id, card_id=card_id)
        assert card is not None
        return card

    def list_cards(
        self, *, user_id: str, deck_id: str, due_only: bool = False, limit: int = 500
    ) -> list[dict]:
        now_iso = _to_iso(_utc_now())
        due_clause = ""
        params: list[Any] = [user_id, deck_id]
        if due_only:
            due_clause = (
                " AND f.due_at <= ? AND f.suspended = 0 AND c.suspended = 0"
            )
            params.append(now_iso)
        params.append(int(limit))
        rows = self._conn.execute(
            f"""
            SELECT c.user_id, c.card_id, c.deck_id, c.front, c.back, c.source_ref,
                c.tags_json, c.fsrs_card_id, c.suspended, c.created_at, c.updated_at,
                {self._fsrs_columns()}
            FROM cards c
            JOIN fsrs_cards f ON f.user_id = c.user_id AND f.card_id = c.fsrs_card_id
            WHERE c.user_id = ? AND c.deck_id = ?{due_clause}
            ORDER BY f.due_at ASC, c.created_at ASC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._card_with_fsrs_row_to_dict(row) for row in rows]

    def get_card(self, *, user_id: str, card_id: str) -> dict | None:
        row = self._conn.execute(
            f"""
            SELECT c.user_id, c.card_id, c.deck_id, c.front, c.back, c.source_ref,
                c.tags_json, c.fsrs_card_id, c.suspended, c.created_at, c.updated_at,
                {self._fsrs_columns()}
            FROM cards c
            JOIN fsrs_cards f ON f.user_id = c.user_id AND f.card_id = c.fsrs_card_id
            WHERE c.user_id = ? AND c.card_id = ?
            """,
            (user_id, card_id),
        ).fetchone()
        return self._card_with_fsrs_row_to_dict(row) if row else None

    def update_card(
        self,
        *,
        user_id: str,
        card_id: str,
        front: str,
        back: str,
        tags: list[str] | None = None,
    ) -> dict | None:
        now_iso = _to_iso(_utc_now())
        clean_front = str(front).strip()[:CARD_FRONT_MAX]
        clean_back = str(back).strip()[:CARD_BACK_MAX]
        clean_tags = [str(t).strip()[:60] for t in (tags or []) if str(t).strip()][
            :CARD_TAGS_MAX
        ]
        tags_json = json.dumps(clean_tags, ensure_ascii=True)
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE cards SET front = ?, back = ?, tags_json = ?, updated_at = ?
                    WHERE user_id = ? AND card_id = ?
                    """,
                    (clean_front, clean_back, tags_json, now_iso, user_id, card_id),
                )
        if cur.rowcount == 0:
            return None
        card = self.get_card(user_id=user_id, card_id=card_id)
        assert card is not None
        return card

    def delete_card(self, *, user_id: str, card_id: str) -> bool:
        with self._lock:
            with self._conn:
                row = self._conn.execute(
                    "SELECT fsrs_card_id FROM cards WHERE user_id = ? AND card_id = ?",
                    (user_id, card_id),
                ).fetchone()
                if row is None:
                    return False
                fsrs_card_id = str(row["fsrs_card_id"])
                self._conn.execute(
                    "DELETE FROM cards WHERE user_id = ? AND card_id = ?",
                    (user_id, card_id),
                )
                self._conn.execute(
                    "DELETE FROM fsrs_review_log WHERE user_id = ? AND card_id = ?",
                    (user_id, fsrs_card_id),
                )
                self._conn.execute(
                    "DELETE FROM fsrs_cards WHERE user_id = ? AND card_id = ?",
                    (user_id, fsrs_card_id),
                )
        return True

    def set_card_suspended(
        self, *, user_id: str, card_id: str, suspended: bool
    ) -> dict | None:
        """Suspend/unsuspend both the card and its FSRS row."""
        now_iso = _to_iso(_utc_now())
        value = 1 if suspended else 0
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "UPDATE cards SET suspended = ?, updated_at = ? WHERE user_id = ? AND card_id = ?",
                    (value, now_iso, user_id, card_id),
                )
                if cur.rowcount == 0:
                    return None
                row = self._conn.execute(
                    "SELECT fsrs_card_id FROM cards WHERE user_id = ? AND card_id = ?",
                    (user_id, card_id),
                ).fetchone()
                if row is not None:
                    self._conn.execute(
                        "UPDATE fsrs_cards SET suspended = ?, updated_at = ? WHERE user_id = ? AND card_id = ?",
                        (value, now_iso, user_id, str(row["fsrs_card_id"])),
                    )
        card = self.get_card(user_id=user_id, card_id=card_id)
        assert card is not None
        return card

    # ── Study session ──────────────────────────────────
    def due_cards_for_deck(
        self, *, user_id: str, deck_id: str, limit: int = 50
    ) -> list[dict]:
        """Due cards for a deck study session (join ``fsrs_cards``)."""
        now_iso = _to_iso(_utc_now())
        rows = self._conn.execute(
            f"""
            SELECT c.user_id, c.card_id, c.deck_id, c.front, c.back, c.source_ref,
                c.tags_json, c.fsrs_card_id, c.suspended, c.created_at, c.updated_at,
                {self._fsrs_columns()}
            FROM cards c
            JOIN fsrs_cards f ON f.user_id = c.user_id AND f.card_id = c.fsrs_card_id
            WHERE c.user_id = ? AND c.deck_id = ?
              AND c.suspended = 0 AND f.suspended = 0 AND f.due_at <= ?
            ORDER BY f.due_at ASC, c.created_at ASC
            LIMIT ?
            """,
            (user_id, deck_id, now_iso, int(limit)),
        ).fetchall()
        return [self._card_with_fsrs_row_to_dict(row) for row in rows]

    # ── Bulk import ────────────────────────────────────
    def find_card_by_text(
        self, *, user_id: str, deck_id: str, front: str, back: str
    ) -> dict | None:
        row = self._conn.execute(
            "SELECT card_id FROM cards WHERE user_id = ? AND deck_id = ? AND front = ? AND back = ? LIMIT 1",
            (user_id, deck_id, str(front).strip(), str(back).strip()),
        ).fetchone()
        if row is None:
            return None
        return self.get_card(user_id=user_id, card_id=str(row["card_id"]))

    def import_cards(
        self,
        *,
        user_id: str,
        deck_id: str,
        cards: list[dict[str, Any]],
        review_logs: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict:
        """Bulk-import cards, each with a fresh ``fsrs_cards`` row.

        ``review_logs`` maps the caller-side card key (the index in
        ``cards``) to Anki revlog-derived history rows, which are
        stored as ``fsrs_review_log`` history only — they never drive
        scheduling.
        """
        now_iso = _to_iso(_utc_now())
        imported: list[dict] = []
        skipped = 0
        with self._lock:
            with self._conn:
                for index, raw in enumerate(cards):
                    front = str(raw.get("front", "")).strip()[:CARD_FRONT_MAX]
                    back = str(raw.get("back", "")).strip()[:CARD_BACK_MAX]
                    if not front or not back:
                        skipped += 1
                        continue
                    existing = self._conn.execute(
                        "SELECT card_id FROM cards WHERE user_id = ? AND deck_id = ? AND front = ? AND back = ? LIMIT 1",
                        (user_id, deck_id, front, back),
                    ).fetchone()
                    if existing is not None:
                        skipped += 1
                        continue
                    card_id = uuid4().hex
                    fsrs_card_id = uuid4().hex
                    tags = [
                        str(t).strip()[:60]
                        for t in (raw.get("tags") or [])
                        if str(t).strip()
                    ][:CARD_TAGS_MAX]
                    source_ref = str(raw.get("source_ref", ""))[:SOURCE_REF_MAX]
                    fresh_fsrs = fsrs_scheduler.create_card()
                    self._conn.execute(
                        """
                        INSERT INTO cards(
                            user_id, card_id, deck_id, front, back, source_ref,
                            tags_json, fsrs_card_id, suspended, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            user_id, card_id, deck_id, front, back, source_ref,
                            json.dumps(tags, ensure_ascii=True), fsrs_card_id,
                            now_iso, now_iso,
                        ),
                    )
                    self._conn.execute(
                        """
                        INSERT INTO fsrs_cards(
                            user_id, card_id, topic_id, source_type, state,
                            stability, difficulty, due_at, last_review_at,
                            reps, lapses, scheduled_days, elapsed_days,
                            suspended, created_at, updated_at
                        ) VALUES (?, ?, ?, 'flashcard', ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, ?, ?)
                        """,
                        (
                            user_id, fsrs_card_id, deck_id,
                            fresh_fsrs["state"], fresh_fsrs["stability"],
                            fresh_fsrs["difficulty"], fresh_fsrs["due_at"],
                            fresh_fsrs["last_review_at"], now_iso, now_iso,
                        ),
                    )
                    for entry in (review_logs or {}).get(str(index), []):
                        self._conn.execute(
                            """
                            INSERT INTO fsrs_review_log(
                                user_id, card_id, rating, state, review_duration_ms,
                                scheduled_days, elapsed_days, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                user_id, fsrs_card_id,
                                str(entry.get("rating", "again")),
                                str(entry.get("state", "learning")),
                                int(entry.get("review_duration_ms", 0) or 0),
                                entry.get("scheduled_days"),
                                entry.get("elapsed_days"),
                                str(entry.get("created_at", now_iso)),
                            ),
                        )
                    imported.append(
                        {
                            "card_id": card_id,
                            "fsrs_card_id": fsrs_card_id,
                            "front": front,
                            "back": back,
                        }
                    )
        return {"imported": imported, "skipped_duplicates": skipped}

    # ── Duplicate detection support ────────────────────
    def review_logs_for_cards(
        self, *, user_id: str, fsrs_card_ids: list[str]
    ) -> list[dict[str, Any]]:
        """FSRS review history for the given FSRS card ids."""
        if not fsrs_card_ids:
            return []
        rows: list[sqlite3.Row] = []
        for start in range(0, len(fsrs_card_ids), 400):
            batch = fsrs_card_ids[start : start + 400]
            placeholders = ",".join("?" * len(batch))
            rows.extend(
                self._conn.execute(
                    f"""
                    SELECT card_id, rating, state, review_duration_ms,
                        scheduled_days, elapsed_days, created_at
                    FROM fsrs_review_log
                    WHERE user_id = ? AND card_id IN ({placeholders})
                    ORDER BY created_at ASC, id ASC
                    """,
                    [user_id, *batch],
                ).fetchall()
            )
        return [
            {
                "card_id": str(row["card_id"]),
                "rating": str(row["rating"]),
                "state": str(row["state"]),
                "review_duration_ms": int(row["review_duration_ms"]),
                "scheduled_days": (
                    int(row["scheduled_days"])
                    if row["scheduled_days"] is not None
                    else None
                ),
                "elapsed_days": (
                    int(row["elapsed_days"])
                    if row["elapsed_days"] is not None
                    else None
                ),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def list_card_texts(
        self, *, user_id: str, deck_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Card front/back text for duplicate detection."""
        if deck_id:
            rows = self._conn.execute(
                "SELECT card_id, deck_id, front, back, tags_json FROM cards WHERE user_id = ? AND deck_id = ?",
                (user_id, deck_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT card_id, deck_id, front, back, tags_json FROM cards WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                tags = json.loads(str(row["tags_json"] or "[]"))
                if not isinstance(tags, list):
                    tags = []
            except json.JSONDecodeError:
                tags = []
            out.append(
                {
                    "card_id": str(row["card_id"]),
                    "deck_id": str(row["deck_id"]),
                    "front": str(row["front"]),
                    "back": str(row["back"]),
                    "tags": [str(t) for t in tags],
                }
            )
        return out
