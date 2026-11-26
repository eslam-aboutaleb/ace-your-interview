"""Generation job orchestration: reliability layer for batch generation (issue #16).

This is the orchestration slice of the generation refactor: durable jobs
with explicit states, idempotent submission keys, cooperative
cancellation, per-item persistence (accepted items are stored before the
next item is attempted, so a crash or retry never re-charges them),
structured provider capability detection, and a hard cost limit per job.
The oversized ``question_generator`` itself is untouched; the orchestrator
drives it through an injected ``generator_fn`` so the reliability contract
is testable without a model.

Job states: queued -> running -> completed | partial | failed | cancelled.
    partial: at least one item accepted and at least one failed.
    failed: nothing accepted (or an unrecoverable orchestration error).
    cancelled: caller cancelled while queued/running; accepted items persist.

Retry is idempotent: ``retry_job`` re-attempts only items that never
produced an accepted result. ``create_job`` with an existing
``(user_id, idempotency_key)`` returns the original job unchanged, so a
retried submission is never charged twice.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_PARTIAL = "partial"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"
TERMINAL_STATES = (JOB_COMPLETED, JOB_PARTIAL, JOB_FAILED, JOB_CANCELLED)

#: A job enriches at most one 50-question outline.
MAX_JOB_ITEMS = 50
#: Per-item generation timeout; a slow provider fails one item, not the job.
DEFAULT_ITEM_TIMEOUT_SECONDS = 60.0
#: Hard per-job cost ceiling in abstract units (one unit per attempted item).
DEFAULT_COST_LIMIT = MAX_JOB_ITEMS

GeneratorFn = Callable[[str], Awaitable[dict[str, Any]]]
CapabilityDetector = Callable[[], dict[str, Any]]


class GenerationJobError(Exception):
    """Base error for generation job operations."""


class GenerationJobNotFoundError(GenerationJobError):
    """The job does not exist or belongs to another user."""


class GenerationJobStateError(GenerationJobError):
    """The operation is not valid for the job's current state."""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def default_capability_detector() -> dict[str, Any]:
    """Conservative static provider capabilities.

    The orchestrator only needs a structured record of what the provider
    can do; real detection (probing the configured provider) plugs in
    through the injectable detector without changing the job contract.
    """
    return {"streaming": False, "json_mode": True, "max_output_tokens": 4096}


class GenerationJobStore:
    """SQLite-backed durable store for generation jobs."""

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
                CREATE TABLE IF NOT EXISTS generation_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outline_json TEXT NOT NULL DEFAULT '[]',
                    accepted_json TEXT NOT NULL DEFAULT '[]',
                    failed_json TEXT NOT NULL DEFAULT '[]',
                    capabilities_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    cost_units INTEGER NOT NULL DEFAULT 0,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (user_id, idempotency_key)
                )
                """
            )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "outline": json.loads(row["outline_json"]),
            "accepted": json.loads(row["accepted_json"]),
            "failed": json.loads(row["failed_json"]),
            "capabilities": json.loads(row["capabilities_json"]),
            "error": row["error"],
            "cost_units": row["cost_units"],
            "attempt_count": row["attempt_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def create_job(
        self,
        *,
        user_id: str,
        idempotency_key: str,
        outline: list[str],
    ) -> tuple[dict[str, Any], bool]:
        """Create a queued job, or return the existing one for the key.

        The (user_id, idempotency_key) uniqueness is what makes a retried
        submission free: the second call returns the original job with
        ``created=False`` and no new work is enqueued.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM generation_jobs WHERE user_id = ? AND idempotency_key = ?",
                (user_id, idempotency_key),
            ).fetchone()
            if row is not None:
                return self._row_to_dict(row), False
            job_id = f"job_{uuid.uuid4().hex[:24]}"
            now = _now_iso()
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO generation_jobs
                        (id, user_id, idempotency_key, status, outline_json,
                         created_at, updated_at)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        job_id,
                        user_id,
                        idempotency_key,
                        JOB_QUEUED,
                        json.dumps(list(outline)),
                        now,
                        now,
                    ),
                )
            return self.get_job(user_id=user_id, job_id=job_id), True

    def get_job(self, *, user_id: str, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM generation_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
            if row is None:
                raise GenerationJobNotFoundError(f"generation job {job_id} not found")
            return self._row_to_dict(row)

    def list_jobs(self, *, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM generation_jobs WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def _update(self, job_id: str, **fields: Any) -> None:
        assignments = []
        values: list[Any] = []
        for key, value in fields.items():
            column = f"{key}_json" if key in ("outline", "accepted", "failed", "capabilities") else key
            assignments.append(f"{column} = ?")
            values.append(json.dumps(value) if column.endswith("_json") else value)
        assignments.append("updated_at = ?")
        values.append(_now_iso())
        values.append(job_id)
        with self._lock, self._conn:
            self._conn.execute(
                f"UPDATE generation_jobs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )

    def mark_running(self, job_id: str, capabilities: dict[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT attempt_count FROM generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            attempts = int(row["attempt_count"]) + 1 if row else 1
        self._update(
            job_id, status=JOB_RUNNING, capabilities=capabilities, attempt_count=attempts
        )

    def append_accepted(self, job_id: str, item: dict[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT accepted_json, cost_units FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            accepted = json.loads(row["accepted_json"])
            accepted.append(item)
            cost = int(row["cost_units"]) + 1
        self._update(job_id, accepted=accepted, cost_units=cost)

    def append_failed(self, job_id: str, item: str, error: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT failed_json, cost_units FROM generation_jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            failed = json.loads(row["failed_json"])
            failed.append({"item": item, "error": error[:500]})
            cost = int(row["cost_units"]) + 1
        self._update(job_id, failed=failed, cost_units=cost)

    def finalize(self, job_id: str, status: str, error: str = "") -> None:
        self._update(job_id, status=status, error=error)

    def cancel(self, *, user_id: str, job_id: str) -> dict[str, Any]:
        job = self.get_job(user_id=user_id, job_id=job_id)
        if job["status"] in TERMINAL_STATES:
            raise GenerationJobStateError(
                f"job {job_id} is already {job['status']} and cannot be cancelled"
            )
        self._update(job_id, status=JOB_CANCELLED)
        return self.get_job(user_id=user_id, job_id=job_id)

    def requeue_for_retry(self, job_id: str) -> None:
        """Move failed items back into the pending set for an idempotent retry."""
        self._update(job_id, failed=[], status=JOB_QUEUED, error="")


class GenerationOrchestrator:
    """Drives one job's outline through an injected generator function."""

    def __init__(
        self,
        store: GenerationJobStore,
        generator_fn: GeneratorFn,
        *,
        capability_detector: CapabilityDetector | None = None,
        max_items: int = MAX_JOB_ITEMS,
        cost_limit: int = DEFAULT_COST_LIMIT,
        item_timeout: float = DEFAULT_ITEM_TIMEOUT_SECONDS,
    ):
        self.store = store
        self.generator_fn = generator_fn
        self.capability_detector = capability_detector or default_capability_detector
        self.max_items = max_items
        self.cost_limit = cost_limit
        self.item_timeout = item_timeout

    async def run_job(self, *, user_id: str, job_id: str) -> dict[str, Any]:
        job = self.store.get_job(user_id=user_id, job_id=job_id)
        if job["status"] == JOB_CANCELLED:
            return job
        if job["status"] in (JOB_COMPLETED, JOB_PARTIAL, JOB_FAILED):
            return job  # idempotent: a finished run is not re-charged

        outline = list(job["outline"])[: self.max_items]
        accepted = list(job["accepted"])
        failed = list(job["failed"])
        done_items = {str(a.get("item", "")) for a in accepted}
        failed_items = {str(f.get("item", "")) for f in failed}
        pending = [
            item for item in outline if item not in done_items and item not in failed_items
        ]

        self.store.mark_running(job_id, self.capability_detector())
        error = ""
        for item in pending:
            # Cooperative cancellation between items.
            current = self.store.get_job(user_id=user_id, job_id=job_id)
            if current["status"] == JOB_CANCELLED:
                return current
            if int(current["cost_units"]) >= self.cost_limit:
                error = "cost limit reached"
                break
            try:
                result = await asyncio.wait_for(
                    self.generator_fn(item), timeout=self.item_timeout
                )
                self.store.append_accepted(job_id, {"item": item, "result": result})
            except (TimeoutError, asyncio.TimeoutError):
                self.store.append_failed(job_id, item, "provider timeout")
            except Exception as exc:  # malformed JSON, provider errors, crashes
                logger_safe_error(exc)
                self.store.append_failed(job_id, item, f"{type(exc).__name__}: {exc}"[:500])

        final = self.store.get_job(user_id=user_id, job_id=job_id)
        n_accepted = len(final["accepted"])
        n_failed = len(final["failed"])
        if n_accepted and (n_failed or error):
            status = JOB_PARTIAL
        elif n_accepted:
            status = JOB_COMPLETED
        else:
            status = JOB_FAILED
            error = error or "no outline items produced an accepted result"
        self.store.finalize(job_id, status, error)
        return self.store.get_job(user_id=user_id, job_id=job_id)

    async def retry_job(self, *, user_id: str, job_id: str) -> dict[str, Any]:
        """Re-attempt only the items that never produced an accepted result."""
        job = self.store.get_job(user_id=user_id, job_id=job_id)
        if job["status"] not in (JOB_PARTIAL, JOB_FAILED):
            raise GenerationJobStateError(
                f"only partial or failed jobs can be retried (job is {job['status']})"
            )
        self.store.requeue_for_retry(job_id)
        return await self.run_job(user_id=user_id, job_id=job_id)


def logger_safe_error(exc: Exception) -> None:
    import logging

    logging.getLogger(__name__).warning("generation item failed: %s", type(exc).__name__)
