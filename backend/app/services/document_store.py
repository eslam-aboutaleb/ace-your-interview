"""SQLite-backed document metadata, chunk storage and vector search.

Vector search uses the sqlite-vec ``vec0`` virtual table (cosine
distance). When the loadable extension is unavailable, a pure-Python
cosine scan over the stored embeddings runs instead — correct but
O(n). The active mode is probed at startup and surfaced through
``vector_index_status()`` so the health endpoint can report it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")

DOCUMENT_STATUSES = ("uploaded", "processing", "ready", "failed")

# Cap on the size of a vec0 ``chunk_id IN (...)`` filter; beyond this the
# exact pure-Python scan is used instead.
VEC_IN_LIMIT = 2000

_vector_index_status: str = "fallback"


def vector_index_status() -> str:
    """Current vector index mode: ``available`` or ``fallback``."""
    return _vector_index_status


def _set_vector_index_status(status: str) -> None:
    global _vector_index_status
    _vector_index_status = status


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def _token_count(content: str) -> int:
    return max(1, len(content.split()))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class DocumentStore:
    """Document metadata, chunks, embeddings and vector search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="document-store"
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._vec_available = self._probe_vec()
        _set_vector_index_status(
            "available" if self._vec_available else "fallback"
        )
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

    # ── sqlite-vec probe ──────────────────────────────────
    def _probe_vec(self) -> bool:
        try:
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _vec_probe "
                "USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[2] "
                "distance_metric=cosine)"
            )
            self._conn.execute("DROP TABLE IF EXISTS _vec_probe")
            logger.info("sqlite-vec extension loaded — vector index available")
            return True
        except Exception as e:
            logger.warning(
                "sqlite-vec unavailable (%s) — using pure-Python cosine "
                "fallback",
                e,
            )
            return False

    # ── Schema ────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    user_id TEXT NOT NULL, document_id TEXT NOT NULL,
                    filename TEXT NOT NULL, mime_type TEXT NOT NULL,
                    title TEXT NOT NULL, status TEXT NOT NULL
                        CHECK(status IN ('uploaded','processing','ready','failed')),
                    error TEXT NOT NULL DEFAULT '', chunk_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, document_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL, document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_doc "
                "ON document_chunks(user_id, document_id, chunk_index)"
            )
            # Stored embeddings (JSON float lists). Always populated; the
            # vec0 table mirrors them when sqlite-vec is available so the
            # pure-Python fallback and vec0 return identical rankings.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id INTEGER PRIMARY KEY,
                    embedding TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS vec_meta "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    feature_key TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_events_feature_time "
                "ON feature_events(feature_key, event_name, created_at DESC)"
            )

    # ── vec index management ──────────────────────────────
    def _vec_meta_get(self, key: str) -> str:
        row = self._conn.execute(
            "SELECT value FROM vec_meta WHERE key = ?", (key,)
        ).fetchone()
        return str(row["value"]) if row else ""

    def _vec_meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO vec_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def ensure_vec_index(self, dimension: int) -> bool:
        """Create the vec0 index; rebuild when the dimension changed.

        Returns ``True`` when the index was (re)built, meaning stored
        vectors were wiped and chunks must be re-embedded.
        """
        if not self._vec_available:
            return False
        stored_dimension = self._vec_meta_get("dimension")
        index_ready = self._vec_meta_get("index_ready")
        if index_ready == "1" and stored_dimension == str(dimension):
            return False
        with self._lock:
            with self._conn:
                self._conn.execute("DROP TABLE IF EXISTS vec_chunks")
                self._create_vec_table(int(dimension))
        logger.info(
            "Vector index (re)built with dimension=%s", int(dimension)
        )
        return True

    def _vec_table_exists(self) -> bool:
        row = self._conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name = 'vec_chunks'"
        ).fetchone()
        return row is not None

    def _create_vec_table(self, dimension: int) -> None:
        self._conn.execute(
            "CREATE VIRTUAL TABLE vec_chunks "
            "USING vec0(chunk_id INTEGER PRIMARY KEY, "
            f"embedding FLOAT[{int(dimension)}] distance_metric=cosine)"
        )
        self._vec_meta_set("dimension", str(int(dimension)))
        self._vec_meta_set("index_ready", "1")

    def _ensure_vec_table_locked(self, dimension: int) -> None:
        """Create vec_chunks on first use. Caller holds ``self._lock``."""
        if not self._vec_available or self._vec_table_exists():
            return
        self._create_vec_table(int(dimension))

    def chunks_missing_vectors(self, dimension: int) -> list[dict]:
        """Chunks whose stored embedding is absent or stale-dimensional."""
        if self._vec_available:
            rows = self._conn.execute(
                """
                SELECT c.chunk_id, c.content
                FROM document_chunks c
                LEFT JOIN vec_chunks v ON v.chunk_id = c.chunk_id
                WHERE v.chunk_id IS NULL
                ORDER BY c.chunk_id
                """
            ).fetchall()
            return [
                {"chunk_id": row["chunk_id"], "content": row["content"]}
                for row in rows
            ]
        rows = self._conn.execute(
            """
            SELECT c.chunk_id, c.content
            FROM document_chunks c
            LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id
            WHERE e.chunk_id IS NULL
               OR json_array_length(e.embedding) != ?
            ORDER BY c.chunk_id
            """,
            (int(dimension),),
        ).fetchall()
        return [
            {"chunk_id": row["chunk_id"], "content": row["content"]}
            for row in rows
        ]

    def upsert_chunk_embedding(
        self, chunk_id: int, embedding: list[float]
    ) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO chunk_embeddings(chunk_id, embedding) "
                    "VALUES (?, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET embedding = excluded.embedding",
                    (chunk_id, json.dumps(embedding)),
                )
                if self._vec_available:
                    self._ensure_vec_table_locked(len(embedding))
                    # vec0 virtual tables do not support
                    # UPSERT — replace via delete + insert.
                    self._conn.execute(
                        "DELETE FROM vec_chunks WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                    self._conn.execute(
                        "INSERT INTO vec_chunks(chunk_id, embedding) "
                        "VALUES (?, ?)",
                        (chunk_id, json.dumps(embedding)),
                    )

    # ── Document status machine ───────────────────────────
    def create_document(
        self,
        user_id: str,
        filename: str,
        mime_type: str,
        title: str,
    ) -> dict:
        document_id = uuid4().hex
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO documents("
                    "user_id, document_id, filename, mime_type, title, "
                    "status, error, chunk_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, 'uploaded', '', 0, ?, ?)",
                    (user_id, document_id, filename, mime_type, title,
                     now_iso, now_iso),
                )
        detail = self.get_document(user_id, document_id)
        assert detail is not None
        return detail

    def update_document_status(
        self,
        user_id: str,
        document_id: str,
        status: str,
        error: str = "",
        chunk_count: int | None = None,
    ) -> dict | None:
        if status not in DOCUMENT_STATUSES:
            raise ValueError(f"invalid document status: {status}")
        sets = ["status = ?", "updated_at = ?", "error = ?"]
        params: list[Any] = [status, _to_iso(_utc_now()), error[:2000]]
        if chunk_count is not None:
            sets.append("chunk_count = ?")
            params.append(int(chunk_count))
        params.extend([user_id, document_id])
        with self._lock:
            with self._conn:
                self._conn.execute(
                    f"UPDATE documents SET {', '.join(sets)} "
                    "WHERE user_id = ? AND document_id = ?",
                    params,
                )
        return self.get_document(user_id, document_id)

    def get_document(self, user_id: str, document_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? AND document_id = ?",
            (user_id, document_id),
        ).fetchone()
        return self._document_row_to_dict(row) if row else None

    def list_documents(self, user_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE user_id = ? "
            "ORDER BY created_at DESC, document_id DESC",
            (user_id,),
        ).fetchall()
        return [self._document_row_to_dict(row) for row in rows]

    def delete_document(self, user_id: str, document_id: str) -> bool:
        chunk_ids = [
            row["chunk_id"]
            for row in self._conn.execute(
                "SELECT chunk_id FROM document_chunks "
                "WHERE user_id = ? AND document_id = ?",
                (user_id, document_id),
            ).fetchall()
        ]
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "DELETE FROM document_chunks "
                    "WHERE user_id = ? AND document_id = ?",
                    (user_id, document_id),
                )
                if chunk_ids:
                    placeholders = ",".join("?" * len(chunk_ids))
                    self._conn.execute(
                        f"DELETE FROM chunk_embeddings "
                        f"WHERE chunk_id IN ({placeholders})",
                        chunk_ids,
                    )
                    if self._vec_available:
                        self._conn.execute(
                            f"DELETE FROM vec_chunks "
                            f"WHERE chunk_id IN ({placeholders})",
                            chunk_ids,
                        )
                cur = self._conn.execute(
                    "DELETE FROM documents "
                    "WHERE user_id = ? AND document_id = ?",
                    (user_id, document_id),
                )
        return cur.rowcount > 0

    def _document_row_to_dict(self, row: sqlite3.Row) -> dict:
        return {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "mime_type": row["mime_type"],
            "title": row["title"],
            "status": row["status"],
            "error": row["error"],
            "chunk_count": int(row["chunk_count"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # ── Chunks ────────────────────────────────────────────
    def persist_chunks(
        self,
        user_id: str,
        document_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        """Replace a document's chunks and store their embeddings."""
        now_iso = _to_iso(_utc_now())
        with self._lock:
            with self._conn:
                existing = [
                    row["chunk_id"]
                    for row in self._conn.execute(
                        "SELECT chunk_id FROM document_chunks "
                        "WHERE user_id = ? AND document_id = ?",
                        (user_id, document_id),
                    ).fetchall()
                ]
                if existing:
                    placeholders = ",".join("?" * len(existing))
                    self._conn.execute(
                        "DELETE FROM document_chunks "
                        "WHERE user_id = ? AND document_id = ?",
                        (user_id, document_id),
                    )
                    self._conn.execute(
                        f"DELETE FROM chunk_embeddings "
                        f"WHERE chunk_id IN ({placeholders})",
                        existing,
                    )
                    if self._vec_available:
                        self._conn.execute(
                            f"DELETE FROM vec_chunks "
                            f"WHERE chunk_id IN ({placeholders})",
                            existing,
                        )
                for index, (content, embedding) in enumerate(
                    zip(chunks, embeddings)
                ):
                    cursor = self._conn.execute(
                        "INSERT INTO document_chunks("
                        "user_id, document_id, chunk_index, content, "
                        "token_count, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (user_id, document_id, index, content,
                         _token_count(content), now_iso),
                    )
                    chunk_id = cursor.lastrowid
                    self._conn.execute(
                        "INSERT INTO chunk_embeddings(chunk_id, embedding) "
                        "VALUES (?, ?)",
                        (chunk_id, json.dumps(embedding)),
                    )
                    if self._vec_available:
                        self._ensure_vec_table_locked(len(embedding))
                        self._conn.execute(
                            "INSERT INTO vec_chunks(chunk_id, embedding) "
                            "VALUES (?, ?)",
                            (chunk_id, json.dumps(embedding)),
                        )
        return len(chunks)

    def get_chunk(
        self, user_id: str, document_id: str, chunk_index: int
    ) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM document_chunks "
            "WHERE user_id = ? AND document_id = ? AND chunk_index = ?",
            (user_id, document_id, int(chunk_index)),
        ).fetchone()
        if not row:
            return None
        return {
            "chunk_id": row["chunk_id"],
            "document_id": row["document_id"],
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "token_count": int(row["token_count"]),
        }

    def get_document_chunks(self, user_id: str, document_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM document_chunks "
            "WHERE user_id = ? AND document_id = ? ORDER BY chunk_index",
            (user_id, document_id),
        ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "token_count": int(row["token_count"]),
            }
            for row in rows
        ]

    # ── Daily upload cap (feature_events) ─────────────────
    def record_upload_event(self, user_id: str, document_id: str) -> None:
        payload = json.dumps({"document_id": document_id})
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO feature_events("
                    "user_id, feature_key, event_name, metadata_json, "
                    "created_at) VALUES (?, 'documents', "
                    "'document_uploaded', ?, ?)",
                    (user_id, payload, _to_iso(_utc_now())),
                )

    def count_uploads_today(self, user_id: str) -> int:
        midnight = _to_iso(
            _utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        )
        row = self._conn.execute(
            "SELECT COUNT(*) FROM feature_events "
            "WHERE user_id = ? AND feature_key = 'documents' "
            "AND event_name = 'document_uploaded' AND created_at >= ?",
            (user_id, midnight),
        ).fetchone()
        return int(row[0])

    # ── Vector search ─────────────────────────────────────
    def search(
        self,
        user_id: str,
        query_embedding: list[float],
        k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[dict]:
        """Return the top-k chunks by cosine similarity (descending)."""
        k = max(1, min(int(k), 100))
        eligible = self._eligible_chunk_ids(user_id, document_ids)
        if not eligible:
            return []
        if (
            self._vec_available
            and self._vec_table_exists()
            and len(eligible) <= VEC_IN_LIMIT
        ):
            return self._search_vec(query_embedding, k, eligible)
        return self._search_fallback(query_embedding, k, eligible)

    def _eligible_chunk_ids(
        self, user_id: str, document_ids: list[str] | None
    ) -> list[int]:
        if document_ids:
            placeholders = ",".join("?" * len(document_ids))
            rows = self._conn.execute(
                f"SELECT chunk_id FROM document_chunks "
                f"WHERE user_id = ? AND document_id IN ({placeholders}) "
                f"ORDER BY chunk_id",
                [user_id] + list(document_ids),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT chunk_id FROM document_chunks "
                "WHERE user_id = ? ORDER BY chunk_id",
                (user_id,),
            ).fetchall()
        return [int(row["chunk_id"]) for row in rows]

    def _search_vec(
        self,
        query_embedding: list[float],
        k: int,
        eligible: list[int],
    ) -> list[dict]:
        placeholders = ",".join("?" * len(eligible))
        rows = self._conn.execute(
            f"SELECT chunk_id, distance FROM vec_chunks "
            f"WHERE embedding MATCH ? AND k = ? "
            f"AND chunk_id IN ({placeholders})",
            [json.dumps(query_embedding), k] + eligible,
        ).fetchall()
        if not rows:
            return []
        scores = {
            int(row["chunk_id"]): 1.0 - float(row["distance"])
            for row in rows
        }
        return self._load_results(scores)

    def _search_fallback(
        self,
        query_embedding: list[float],
        k: int,
        eligible: list[int],
    ) -> list[dict]:
        placeholders = ",".join("?" * len(eligible))
        rows = self._conn.execute(
            f"SELECT c.chunk_id, c.document_id, c.chunk_index, c.content, "
            f"e.embedding FROM document_chunks c "
            f"JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id "
            f"WHERE c.chunk_id IN ({placeholders})",
            eligible,
        ).fetchall()
        scored: list[dict] = []
        for row in rows:
            stored = json.loads(row["embedding"])
            if len(stored) != len(query_embedding):
                continue
            scored.append(
                {
                    "chunk_id": int(row["chunk_id"]),
                    "document_id": row["document_id"],
                    "chunk_index": int(row["chunk_index"]),
                    "content": row["content"],
                    "score": cosine_similarity(query_embedding, stored),
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:k]

    def _load_results(self, scores: dict[int, float]) -> list[dict]:
        if not scores:
            return []
        placeholders = ",".join("?" * len(scores))
        rows = self._conn.execute(
            f"SELECT chunk_id, document_id, chunk_index, content "
            f"FROM document_chunks WHERE chunk_id IN ({placeholders})",
            list(scores.keys()),
        ).fetchall()
        results = [
            {
                "chunk_id": int(row["chunk_id"]),
                "document_id": row["document_id"],
                "chunk_index": int(row["chunk_index"]),
                "content": row["content"],
                "score": scores[int(row["chunk_id"])],
            }
            for row in rows
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results
