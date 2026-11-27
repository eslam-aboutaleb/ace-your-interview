"""Durable storage for learning journeys (issue #4).

A journey is a persistent, versioned learning plan for one user:
``journey -> phases -> modules -> objectives`` plus immutable, numbered
question batches and an optimistic-concurrency reader checkpoint.

Design rules that later stages rely on:

* IDs are server-assigned. Phase/module/objective IDs are stable content
  hashes over ``journey_id`` + position-in-source + title, so a replan keeps
  unchanged objectives addressable and completed progress attached.
* Batches are immutable once created: ``batch_number`` is assigned
  transactionally and never reused, slots keep their insertion order, and
  ``(journey_id, idempotency_key)`` is unique so a retried POST returns the
  original batch instead of duplicating reservations.
* Checkpoints use an integer revision for optimistic concurrency and a
  mutation log for idempotent writes; a conflict never overwrites newer
  server state.
* Objectives are never hard-deleted. A replan retires missing objectives so
  historical progress survives curriculum upgrades.

The store is synchronous SQLite behind the same single-writer executor
pattern as ``LearningStore``; routers call it through ``run_async``.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Callable, TypeVar

T = TypeVar("T")

MAX_BATCH_SLOTS = 50


class JourneyError(Exception):
    """Base class for journey domain errors."""


class JourneyNotFoundError(JourneyError):
    """The journey does not exist or does not belong to the user."""


class NoPendingObjectivesError(JourneyError):
    """A batch was requested but every objective is already reserved."""


class JourneyValidationError(JourneyError):
    """A write referenced entities that do not exist in this journey."""


class RevisionConflictError(JourneyError):
    """The caller's expected revision is stale."""

    def __init__(self, message: str, *, current: dict[str, Any]):
        super().__init__(message)
        self.current = current


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


class JourneyStore:
    """Persist journeys, plans, batches and reader checkpoints."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="journey-store"
        )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    async def run_async(self, fn: Callable[..., T], /, *args, **kwargs) -> T:
        """Run synchronous SQLite work on the dedicated writer thread."""
        loop = asyncio.get_running_loop()
        if kwargs:
            return await loop.run_in_executor(
                self._executor, lambda: fn(*args, **kwargs)
            )
        return await loop.run_in_executor(self._executor, fn, *args)

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journeys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    goal TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL,
                    source_topic_id TEXT NOT NULL DEFAULT '',
                    track TEXT NOT NULL DEFAULT '',
                    level TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'active',
                    plan_version INTEGER NOT NULL DEFAULT 1,
                    provenance_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_journeys_user_updated
                ON journeys(user_id, updated_at DESC)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_phases (
                    id TEXT PRIMARY KEY,
                    journey_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    UNIQUE(journey_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_modules (
                    id TEXT PRIMARY KEY,
                    journey_id TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    UNIQUE(phase_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_objectives (
                    id TEXT PRIMARY KEY,
                    journey_id TEXT NOT NULL,
                    module_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'concept',
                    source_section TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(module_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_journey_objectives_journey
                ON journey_objectives(journey_id, status)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_batches (
                    id TEXT PRIMARY KEY,
                    journey_id TEXT NOT NULL,
                    batch_number INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    prompt_version TEXT NOT NULL DEFAULT '',
                    model_version TEXT NOT NULL DEFAULT '',
                    schema_version TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'planned',
                    accepted_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(journey_id, batch_number),
                    UNIQUE(journey_id, idempotency_key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_questions (
                    id TEXT PRIMARY KEY,
                    journey_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    objective_id TEXT NOT NULL DEFAULT '',
                    question_ref TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'reserved',
                    created_at TEXT NOT NULL,
                    UNIQUE(batch_id, position)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_journey_questions_journey
                ON journey_questions(journey_id, objective_id)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_checkpoints (
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    last_batch_id TEXT NOT NULL DEFAULT '',
                    last_question_id TEXT NOT NULL DEFAULT '',
                    last_section_id TEXT NOT NULL DEFAULT '',
                    scroll_offset REAL NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, journey_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_mutations (
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    mutation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    result_revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, journey_id, mutation_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_question_fingerprints (
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    objective_id TEXT NOT NULL DEFAULT '',
                    fingerprint TEXT NOT NULL,
                    signature_json TEXT NOT NULL DEFAULT '[]',
                    embedding_json TEXT NOT NULL DEFAULT '',
                    perspective TEXT NOT NULL DEFAULT '',
                    question_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (journey_id, fingerprint),
                    UNIQUE (slot_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_generation_jobs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    prompt_version TEXT NOT NULL DEFAULT '',
                    model_version TEXT NOT NULL DEFAULT '',
                    schema_version TEXT NOT NULL DEFAULT '',
                    rounds INTEGER NOT NULL DEFAULT 0,
                    accepted_count INTEGER NOT NULL DEFAULT 0,
                    rejected_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_journey_generation_jobs_batch
                ON journey_generation_jobs(journey_id, batch_id)
                """
            )
            # Column migrations for databases created before issue #2:
            # ADD COLUMN is idempotent-guarded via PRAGMA table_info.
            existing = {
                row["name"]
                for row in self._conn.execute("PRAGMA table_info(journey_questions)").fetchall()
            }
            for column, ddl in (
                ("perspective", "ALTER TABLE journey_questions ADD COLUMN perspective TEXT NOT NULL DEFAULT ''"),
                ("question_json", "ALTER TABLE journey_questions ADD COLUMN question_json TEXT NOT NULL DEFAULT ''"),
                ("rejection_reason", "ALTER TABLE journey_questions ADD COLUMN rejection_reason TEXT NOT NULL DEFAULT ''"),
            ):
                if column not in existing:
                    self._conn.execute(ddl)

            # Issue #5: per-question learner state and attempt history.
            # Completion derives from distinct server-confirmed slots, never
            # from incrementing counters; attempts are idempotent per
            # mutation_id so duplicate submits cannot double-count.
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_question_states (
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    viewed_at TEXT NOT NULL DEFAULT '',
                    studied_at TEXT NOT NULL DEFAULT '',
                    practiced_at TEXT NOT NULL DEFAULT '',
                    mastered_at TEXT NOT NULL DEFAULT '',
                    completed INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, journey_id, slot_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journey_question_attempts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    journey_id TEXT NOT NULL,
                    slot_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    confidence INTEGER NOT NULL DEFAULT 0,
                    response_time_ms INTEGER NOT NULL DEFAULT 0,
                    mutation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (user_id, journey_id, mutation_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_journey_attempts_slot
                ON journey_question_attempts(user_id, journey_id, slot_id, created_at)
                """
            )

    # ------------------------------------------------------------------
    # row mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _journey_row_to_summary(
        row: sqlite3.Row, *, objective_count: int = 0, batch_count: int = 0
    ) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "goal": row["goal"],
            "source_type": row["source_type"],
            "source_topic_id": row["source_topic_id"],
            "track": row["track"],
            "level": row["level"],
            "status": row["status"],
            "plan_version": row["plan_version"],
            "objective_count": objective_count,
            "batch_count": batch_count,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _batch_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "journey_id": row["journey_id"],
            "batch_number": row["batch_number"],
            "idempotency_key": row["idempotency_key"],
            "plan_version": row["plan_version"],
            "prompt_version": row["prompt_version"],
            "model_version": row["model_version"],
            "schema_version": row["schema_version"],
            "status": row["status"],
            "accepted_count": row["accepted_count"],
            "rejected_count": row["rejected_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _slot_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "batch_id": row["batch_id"],
            "position": row["position"],
            "objective_id": row["objective_id"],
            "objective_title": row["objective_title"] or "",
            "question_ref": row["question_ref"],
            "status": row["status"],
            "perspective": row["perspective"] if "perspective" in row.keys() else "",
            "rejection_reason": row["rejection_reason"] if "rejection_reason" in row.keys() else "",
        }

    @staticmethod
    def _checkpoint_row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            return {
                "last_batch_id": "",
                "last_question_id": "",
                "last_section_id": "",
                "scroll_offset": 0.0,
                "revision": 0,
                "updated_at": "",
            }
        return {
            "last_batch_id": row["last_batch_id"],
            "last_question_id": row["last_question_id"],
            "last_section_id": row["last_section_id"],
            "scroll_offset": row["scroll_offset"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }

    def _get_journey_row(self, *, user_id: str, journey_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM journeys WHERE id = ? AND user_id = ?",
            (journey_id, user_id),
        ).fetchone()

    def _require_journey_row(self, *, user_id: str, journey_id: str) -> sqlite3.Row:
        row = self._get_journey_row(user_id=user_id, journey_id=journey_id)
        if row is None:
            raise JourneyNotFoundError(f"journey {journey_id!r} not found")
        return row

    # ------------------------------------------------------------------
    # plan helpers
    # ------------------------------------------------------------------
    @staticmethod
    def plan_ids(journey_id: str, plan: dict[str, Any]) -> dict[str, Any]:
        """Assign stable content-hash IDs to a plan tree.

        The same source content always maps to the same IDs within a journey,
        which is what makes replans non-destructive.
        """
        stamped = {"phases": []}
        for phase_pos, phase in enumerate(plan.get("phases", [])):
            phase_id = _stable_id("ph", journey_id, phase.get("title", ""), phase_pos)
            stamped_phase = {
                "id": phase_id,
                "position": phase_pos,
                "title": str(phase.get("title", ""))[:300],
                "description": str(phase.get("description", ""))[:2000],
                "modules": [],
            }
            for mod_pos, module in enumerate(phase.get("modules", [])):
                module_id = _stable_id(
                    "mod", phase_id, module.get("title", ""), mod_pos
                )
                stamped_module = {
                    "id": module_id,
                    "phase_id": phase_id,
                    "position": mod_pos,
                    "title": str(module.get("title", ""))[:300],
                    "description": str(module.get("description", ""))[:2000],
                    "objectives": [],
                }
                for obj_pos, objective in enumerate(module.get("objectives", [])):
                    stamped_module["objectives"].append(
                        {
                            "id": _stable_id(
                                "obj", module_id, objective.get("title", ""), obj_pos
                            ),
                            "module_id": module_id,
                            "position": obj_pos,
                            "title": str(objective.get("title", ""))[:300],
                            "kind": str(objective.get("kind", "concept"))[:50],
                            "source_section": str(objective.get("source_section", ""))[
                                :300
                            ],
                        }
                    )
                stamped_phase["modules"].append(stamped_module)
            stamped["phases"].append(stamped_phase)
        return stamped

    def _insert_plan(
        self, *, journey_id: str, plan: dict[str, Any], now_iso: str
    ) -> int:
        count = 0
        for phase in plan["phases"]:
            self._conn.execute(
                "INSERT INTO journey_phases (id, journey_id, position, title, description) VALUES (?,?,?,?,?)",
                (
                    phase["id"],
                    journey_id,
                    phase["position"],
                    phase["title"],
                    phase["description"],
                ),
            )
            for module in phase["modules"]:
                self._conn.execute(
                    "INSERT INTO journey_modules (id, journey_id, phase_id, position, title, description) VALUES (?,?,?,?,?,?)",
                    (
                        module["id"],
                        journey_id,
                        module["phase_id"],
                        module["position"],
                        module["title"],
                        module["description"],
                    ),
                )
                for objective in module["objectives"]:
                    self._conn.execute(
                        """
                        INSERT INTO journey_objectives
                            (id, journey_id, module_id, position, title, kind, source_section, status, created_at, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            objective["id"],
                            journey_id,
                            objective["module_id"],
                            objective["position"],
                            objective["title"],
                            objective["kind"],
                            objective["source_section"],
                            "pending",
                            now_iso,
                            now_iso,
                        ),
                    )
                    count += 1
        return count

    # ------------------------------------------------------------------
    # journeys
    # ------------------------------------------------------------------
    def create_journey(
        self,
        *,
        user_id: str,
        title: str,
        goal: str,
        source_type: str,
        source_topic_id: str,
        track: str,
        level: str,
        plan: dict[str, Any],
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        import json

        journey_id = f"jrn_{uuid.uuid4().hex[:24]}"
        stamped = self.plan_ids(journey_id, plan)
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO journeys
                    (id, user_id, title, goal, source_type, source_topic_id, track, level,
                     status, plan_version, provenance_json, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?, 'active', 1, ?, ?, ?)
                """,
                (
                    journey_id,
                    user_id,
                    title.strip(),
                    goal,
                    source_type,
                    source_topic_id,
                    track,
                    level,
                    json.dumps(provenance or {}, sort_keys=True),
                    now_iso,
                    now_iso,
                ),
            )
            self._insert_plan(journey_id=journey_id, plan=stamped, now_iso=now_iso)
        return self.get_journey(user_id=user_id, journey_id=journey_id)

    def list_journeys(
        self, *, user_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM journeys WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_archived:
            query += " AND status != 'archived'"
        query += " ORDER BY updated_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        summaries = []
        for row in rows:
            counts = self._counts_for_journey(row["id"])
            summaries.append(self._journey_row_to_summary(row, **counts))
        return summaries

    def _counts_for_journey(self, journey_id: str) -> dict[str, int]:
        objective_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM journey_objectives WHERE journey_id = ? AND status != 'retired'",
            (journey_id,),
        ).fetchone()["c"]
        batch_count = self._conn.execute(
            "SELECT COUNT(*) AS c FROM journey_batches WHERE journey_id = ?",
            (journey_id,),
        ).fetchone()["c"]
        return {"objective_count": objective_count, "batch_count": batch_count}

    def get_journey(self, *, user_id: str, journey_id: str) -> dict[str, Any]:
        row = self._require_journey_row(user_id=user_id, journey_id=journey_id)
        counts = self._counts_for_journey(journey_id)
        detail = self._journey_row_to_summary(row, **counts)

        phases = self._conn.execute(
            "SELECT * FROM journey_phases WHERE journey_id = ? ORDER BY position",
            (journey_id,),
        ).fetchall()
        modules = self._conn.execute(
            "SELECT * FROM journey_modules WHERE journey_id = ? ORDER BY position",
            (journey_id,),
        ).fetchall()
        objectives = self._conn.execute(
            "SELECT * FROM journey_objectives WHERE journey_id = ? ORDER BY position",
            (journey_id,),
        ).fetchall()

        objectives_by_module: dict[str, list[dict[str, Any]]] = {}
        for obj in objectives:
            objectives_by_module.setdefault(obj["module_id"], []).append(
                {
                    "id": obj["id"],
                    "module_id": obj["module_id"],
                    "position": obj["position"],
                    "title": obj["title"],
                    "kind": obj["kind"],
                    "source_section": obj["source_section"],
                    "status": obj["status"],
                }
            )
        modules_by_phase: dict[str, list[dict[str, Any]]] = {}
        for mod in modules:
            modules_by_phase.setdefault(mod["phase_id"], []).append(
                {
                    "id": mod["id"],
                    "phase_id": mod["phase_id"],
                    "position": mod["position"],
                    "title": mod["title"],
                    "description": mod["description"],
                    "objectives": objectives_by_module.get(mod["id"], []),
                }
            )
        detail["phases"] = [
            {
                "id": ph["id"],
                "position": ph["position"],
                "title": ph["title"],
                "description": ph["description"],
                "modules": modules_by_phase.get(ph["id"], []),
            }
            for ph in phases
        ]
        return detail

    def refresh_plan(
        self, *, user_id: str, journey_id: str, plan: dict[str, Any]
    ) -> dict[str, Any]:
        """Reconcile the stored plan with freshly derived source content.

        Unchanged objectives keep their IDs (and any recorded progress). New
        objectives are inserted, missing ones are retired - never deleted.
        ``plan_version`` only increments when something actually changed.
        """
        row = self._require_journey_row(user_id=user_id, journey_id=journey_id)
        stamped = self.plan_ids(journey_id, plan)
        now_iso = _to_iso(_utc_now())

        new_phase_ids = {ph["id"] for ph in stamped["phases"]}
        new_modules = {
            mod["id"]: mod for ph in stamped["phases"] for mod in ph["modules"]
        }
        new_objectives = {
            obj["id"]: obj
            for ph in stamped["phases"]
            for mod in ph["modules"]
            for obj in mod["objectives"]
        }

        added = 0
        retired = 0
        changed = False
        with self._lock, self._conn:
            existing_phases = {
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM journey_phases WHERE journey_id = ?", (journey_id,)
                ).fetchall()
            }
            existing_objectives = {
                r["id"]: r
                for r in self._conn.execute(
                    "SELECT * FROM journey_objectives WHERE journey_id = ?",
                    (journey_id,),
                ).fetchall()
            }

            # IDs are content hashes over title + position, so an existing
            # row with the same ID is identical by construction: only
            # insert-if-missing is ever needed. A retitle or reorder yields
            # new IDs (new rows), and the superseded objectives retire below.
            if set(existing_phases) != new_phase_ids:
                changed = True
            for phase in stamped["phases"]:
                if phase["id"] not in existing_phases:
                    self._conn.execute(
                        "INSERT INTO journey_phases (id, journey_id, position, title, description) VALUES (?,?,?,?,?)",
                        (
                            phase["id"],
                            journey_id,
                            phase["position"],
                            phase["title"],
                            phase["description"],
                        ),
                    )
                    changed = True

            for module in new_modules.values():
                found = self._conn.execute(
                    "SELECT id FROM journey_modules WHERE id = ?", (module["id"],)
                ).fetchone()
                if found is None:
                    self._conn.execute(
                        "INSERT INTO journey_modules (id, journey_id, phase_id, position, title, description) VALUES (?,?,?,?,?,?)",
                        (
                            module["id"],
                            journey_id,
                            module["phase_id"],
                            module["position"],
                            module["title"],
                            module["description"],
                        ),
                    )
                    changed = True

            for objective in new_objectives.values():
                existing = existing_objectives.get(objective["id"])
                if existing is None:
                    self._conn.execute(
                        """
                        INSERT INTO journey_objectives
                            (id, journey_id, module_id, position, title, kind, source_section, status, created_at, updated_at)
                        VALUES (?,?,?,?,?,?,?, 'pending', ?, ?)
                        """,
                        (
                            objective["id"],
                            journey_id,
                            objective["module_id"],
                            objective["position"],
                            objective["title"],
                            objective["kind"],
                            objective["source_section"],
                            now_iso,
                            now_iso,
                        ),
                    )
                    added += 1
                    changed = True
                elif existing["status"] == "retired":
                    # Content came back in a replan: revive it.
                    self._conn.execute(
                        "UPDATE journey_objectives SET status = 'pending', updated_at = ? WHERE id = ?",
                        (now_iso, objective["id"]),
                    )
                    changed = True

            for old_id, old in existing_objectives.items():
                if old_id not in new_objectives and old["status"] != "retired":
                    self._conn.execute(
                        "UPDATE journey_objectives SET status = 'retired', updated_at = ? WHERE id = ?",
                        (now_iso, old_id),
                    )
                    retired += 1
                    changed = True

            if changed:
                self._conn.execute(
                    "UPDATE journeys SET plan_version = plan_version + 1, updated_at = ? WHERE id = ?",
                    (now_iso, journey_id),
                )

        detail = self.get_journey(user_id=user_id, journey_id=journey_id)
        return {
            "journey": detail,
            "plan_changed": changed,
            "added_objectives": added,
            "retired_objectives": retired,
            "previous_plan_version": row["plan_version"],
        }

    # ------------------------------------------------------------------
    # batches
    # ------------------------------------------------------------------
    def create_batch(
        self,
        *,
        user_id: str,
        journey_id: str,
        idempotency_key: str,
        slot_count: int = 10,
        prompt_version: str = "",
        model_version: str = "",
        schema_version: str = "",
        perspectives: list[str] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Reserve the next immutable batch of question slots.

        Returns ``(batch, created)``. Replaying an idempotency key returns the
        original batch with ``created=False`` and reserves nothing twice.
        """
        if slot_count < 1 or slot_count > MAX_BATCH_SLOTS:
            raise JourneyValidationError(
                f"slot_count must be between 1 and {MAX_BATCH_SLOTS}"
            )
        row = self._require_journey_row(user_id=user_id, journey_id=journey_id)
        if row["status"] == "archived":
            raise JourneyValidationError(
                "cannot create batches for an archived journey"
            )
        now_iso = _to_iso(_utc_now())

        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM journey_batches WHERE journey_id = ? AND idempotency_key = ?",
                (journey_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                return self._batch_row_to_dict(existing), False

            next_number_row = self._conn.execute(
                "SELECT COALESCE(MAX(batch_number), 0) + 1 AS n FROM journey_batches WHERE journey_id = ?",
                (journey_id,),
            ).fetchone()
            batch_number = int(next_number_row["n"])

            objectives = self._conn.execute(
                """
                SELECT o.id, o.title FROM journey_objectives o
                JOIN journey_modules m ON m.id = o.module_id
                JOIN journey_phases p ON p.id = m.phase_id
                WHERE o.journey_id = ? AND o.status = 'pending'
                  AND NOT EXISTS (
                      SELECT 1 FROM journey_questions q
                      WHERE q.journey_id = o.journey_id AND q.objective_id = o.id
                  )
                ORDER BY p.position, m.position, o.position
                LIMIT ?
                """,
                (journey_id, slot_count),
            ).fetchall()
            if not objectives:
                raise NoPendingObjectivesError(
                    "every objective in this journey already has a reserved question slot"
                )

            batch_id = f"bat_{uuid.uuid4().hex[:24]}"
            self._conn.execute(
                """
                INSERT INTO journey_batches
                    (id, journey_id, batch_number, idempotency_key, plan_version,
                     prompt_version, model_version, schema_version, status,
                     accepted_count, rejected_count, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?, 'planned', 0, 0, ?, ?)
                """,
                (
                    batch_id,
                    journey_id,
                    batch_number,
                    idempotency_key,
                    row["plan_version"],
                    prompt_version,
                    model_version,
                    schema_version,
                    now_iso,
                    now_iso,
                ),
            )
            clean_perspectives = [
                (p or "").strip().lower() for p in (perspectives or []) if (p or "").strip()
            ]
            for position, objective in enumerate(objectives):
                slot_perspective = (
                    clean_perspectives[position % len(clean_perspectives)]
                    if clean_perspectives
                    else ""
                )
                self._conn.execute(
                    """
                    INSERT INTO journey_questions
                        (id, journey_id, batch_id, position, objective_id, question_ref, status, perspective, created_at)
                    VALUES (?,?,?,?,?,?, 'reserved', ?, ?)
                    """,
                    (
                        f"slot_{uuid.uuid4().hex[:24]}",
                        journey_id,
                        batch_id,
                        position,
                        objective["id"],
                        "",
                        slot_perspective,
                        now_iso,
                    ),
                )
            self._conn.execute(
                "UPDATE journeys SET updated_at = ? WHERE id = ?",
                (now_iso, journey_id),
            )
            created = self._conn.execute(
                "SELECT * FROM journey_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            return self._batch_row_to_dict(created), True

    def list_batches(self, *, user_id: str, journey_id: str) -> list[dict[str, Any]]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        rows = self._conn.execute(
            "SELECT * FROM journey_batches WHERE journey_id = ? ORDER BY batch_number",
            (journey_id,),
        ).fetchall()
        return [self._batch_row_to_dict(row) for row in rows]

    def get_batch_slots(
        self, *, user_id: str, journey_id: str, batch_id: str
    ) -> list[dict[str, Any]]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        rows = self._conn.execute(
            """
            SELECT q.*, o.title AS objective_title FROM journey_questions q
            LEFT JOIN journey_objectives o ON o.id = q.objective_id
            WHERE q.journey_id = ? AND q.batch_id = ?
            ORDER BY q.position
            """,
            (journey_id, batch_id),
        ).fetchall()
        return [self._slot_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # reader state and checkpoints
    # ------------------------------------------------------------------
    def get_reader_state(self, *, user_id: str, journey_id: str) -> dict[str, Any]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        checkpoint = self._checkpoint_row_to_dict(
            self._conn.execute(
                "SELECT * FROM journey_checkpoints WHERE user_id = ? AND journey_id = ?",
                (user_id, journey_id),
            ).fetchone()
        )
        latest = self._conn.execute(
            "SELECT * FROM journey_batches WHERE journey_id = ? ORDER BY batch_number DESC LIMIT 1",
            (journey_id,),
        ).fetchone()
        questions: list[dict[str, Any]] = []
        if latest is not None:
            questions = self.get_batch_slots(
                user_id=user_id, journey_id=journey_id, batch_id=latest["id"]
            )
        return {
            "journey_id": journey_id,
            "checkpoint": checkpoint,
            "latest_batch": self._batch_row_to_dict(latest)
            if latest is not None
            else None,
            "questions": questions,
        }

    def update_checkpoint(
        self,
        *,
        user_id: str,
        journey_id: str,
        expected_revision: int,
        mutation_id: str,
        last_batch_id: str | None = None,
        last_question_id: str | None = None,
        last_section_id: str | None = None,
        scroll_offset: float | None = None,
    ) -> dict[str, Any]:
        """Apply an optimistic, idempotent checkpoint write.

        * A replayed ``mutation_id`` returns the current state unchanged.
        * A stale ``expected_revision`` raises :class:`RevisionConflictError`
          carrying the current server state for client reconciliation.
        * Referenced batch/question IDs must belong to the journey.
        """
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        now_iso = _to_iso(_utc_now())

        with self._lock, self._conn:
            replay = self._conn.execute(
                "SELECT result_revision FROM journey_mutations WHERE user_id = ? AND journey_id = ? AND mutation_id = ?",
                (user_id, journey_id, mutation_id),
            ).fetchone()
            current_row = self._conn.execute(
                "SELECT * FROM journey_checkpoints WHERE user_id = ? AND journey_id = ?",
                (user_id, journey_id),
            ).fetchone()
            current = self._checkpoint_row_to_dict(current_row)

            if replay is not None:
                return current

            if expected_revision != current["revision"]:
                raise RevisionConflictError(
                    f"checkpoint revision mismatch: expected {expected_revision}, "
                    f"server has {current['revision']}",
                    current=current,
                )

            if last_batch_id:
                found = self._conn.execute(
                    "SELECT id FROM journey_batches WHERE id = ? AND journey_id = ?",
                    (last_batch_id, journey_id),
                ).fetchone()
                if found is None:
                    raise JourneyValidationError(
                        f"batch {last_batch_id!r} does not belong to journey {journey_id!r}"
                    )
            if last_question_id:
                found = self._conn.execute(
                    "SELECT id FROM journey_questions WHERE id = ? AND journey_id = ?",
                    (last_question_id, journey_id),
                ).fetchone()
                if found is None:
                    raise JourneyValidationError(
                        f"question slot {last_question_id!r} does not belong to journey {journey_id!r}"
                    )

            new_revision = current["revision"] + 1
            merged = {
                "last_batch_id": current["last_batch_id"]
                if last_batch_id is None
                else last_batch_id,
                "last_question_id": current["last_question_id"]
                if last_question_id is None
                else last_question_id,
                "last_section_id": current["last_section_id"]
                if last_section_id is None
                else last_section_id,
                "scroll_offset": current["scroll_offset"]
                if scroll_offset is None
                else scroll_offset,
            }
            self._conn.execute(
                """
                INSERT INTO journey_checkpoints
                    (user_id, journey_id, last_batch_id, last_question_id, last_section_id,
                     scroll_offset, revision, updated_at)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id, journey_id) DO UPDATE SET
                    last_batch_id = excluded.last_batch_id,
                    last_question_id = excluded.last_question_id,
                    last_section_id = excluded.last_section_id,
                    scroll_offset = excluded.scroll_offset,
                    revision = excluded.revision,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    journey_id,
                    merged["last_batch_id"],
                    merged["last_question_id"],
                    merged["last_section_id"],
                    merged["scroll_offset"],
                    new_revision,
                    now_iso,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO journey_mutations
                    (user_id, journey_id, mutation_id, kind, result_revision, created_at)
                VALUES (?,?,?, 'checkpoint', ?, ?)
                """,
                (user_id, journey_id, mutation_id, new_revision, now_iso),
            )
            return {
                **merged,
                "revision": new_revision,
                "updated_at": now_iso,
            }

    # ------------------------------------------------------------------
    # novelty fingerprints and generation jobs (issue #2)
    # ------------------------------------------------------------------
    def fingerprints_for_journey(self, *, user_id: str, journey_id: str) -> list[dict[str, Any]]:
        """Every accepted question's novelty record for this journey.

        Powers the generation dedup funnel: exact hash, concept signature,
        optional embedding, perspective and the source text for the
        character-level stage.
        """
        import json

        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        rows = self._conn.execute(
            """
            SELECT * FROM journey_question_fingerprints
            WHERE journey_id = ? AND user_id = ?
            ORDER BY created_at, rowid
            """,
            (journey_id, user_id),
        ).fetchall()
        records = []
        for row in rows:
            embedding_raw = row["embedding_json"]
            records.append(
                {
                    "slot_id": row["slot_id"],
                    "objective_id": row["objective_id"],
                    "fingerprint": row["fingerprint"],
                    "signature": frozenset(json.loads(row["signature_json"] or "[]")),
                    "embedding": tuple(json.loads(embedding_raw)) if embedding_raw else None,
                    "perspective": row["perspective"],
                    "question_text": row["question_text"],
                    "created_at": row["created_at"],
                }
            )
        return records

    def create_generation_job(
        self,
        *,
        user_id: str,
        journey_id: str,
        batch_id: str,
        prompt_version: str = "",
        model_version: str = "",
        schema_version: str = "",
    ) -> dict[str, Any]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        job_id = f"gen_{uuid.uuid4().hex[:24]}"
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO journey_generation_jobs
                    (id, user_id, journey_id, batch_id, status, prompt_version,
                     model_version, schema_version, created_at, updated_at)
                VALUES (?,?,?,?, 'running', ?,?,?, ?, ?)
                """,
                (
                    job_id, user_id, journey_id, batch_id,
                    prompt_version, model_version, schema_version,
                    now_iso, now_iso,
                ),
            )
            row = self._conn.execute(
                "SELECT * FROM journey_generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_row_to_dict(row)

    @staticmethod
    def _job_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "journey_id": row["journey_id"],
            "batch_id": row["batch_id"],
            "status": row["status"],
            "prompt_version": row["prompt_version"],
            "model_version": row["model_version"],
            "schema_version": row["schema_version"],
            "rounds": row["rounds"],
            "accepted_count": row["accepted_count"],
            "rejected_count": row["rejected_count"],
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def get_generation_job(self, *, user_id: str, journey_id: str, job_id: str) -> dict[str, Any]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        row = self._conn.execute(
            "SELECT * FROM journey_generation_jobs WHERE id = ? AND journey_id = ?",
            (job_id, journey_id),
        ).fetchone()
        if row is None:
            raise JourneyNotFoundError(f"generation job {job_id!r} not found")
        return self._job_row_to_dict(row)

    def complete_generation_job(
        self,
        *,
        user_id: str,
        job_id: str,
        status: str,
        rounds: int,
        accepted_count: int,
        rejected_count: int,
        error: str = "",
    ) -> dict[str, Any]:
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM journey_generation_jobs WHERE id = ? AND user_id = ?",
                (job_id, user_id),
            ).fetchone()
            if row is None:
                raise JourneyNotFoundError(f"generation job {job_id!r} not found")
            self._conn.execute(
                """
                UPDATE journey_generation_jobs
                SET status = ?, rounds = ?, accepted_count = ?, rejected_count = ?,
                    error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, rounds, accepted_count, rejected_count, error, now_iso, job_id),
            )
            row = self._conn.execute(
                "SELECT * FROM journey_generation_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._job_row_to_dict(row)

    def accept_slot(
        self,
        *,
        user_id: str,
        journey_id: str,
        slot_id: str,
        question_ref: str,
        question_json: str,
        question_text: str,
        fingerprint: str,
        signature: list[str],
        embedding: list[float] | None,
        perspective: str = "",
    ) -> dict[str, Any]:
        """Mark a reserved slot accepted and persist its novelty record.

        One transaction: the slot flips to ``accepted`` and the fingerprint
        row lands atomically, so a crash can never leave an accepted slot
        whose question is invisible to future dedup rounds.
        """
        import json

        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            slot = self._require_slot_for_update(journey_id=journey_id, slot_id=slot_id)
            if slot["status"] not in ("reserved", "rejected"):
                raise JourneyValidationError(
                    f"slot {slot_id!r} is {slot['status']}, expected reserved"
                )
            self._conn.execute(
                """
                UPDATE journey_questions
                SET status = 'accepted', question_ref = ?, question_json = ?,
                    rejection_reason = ''
                WHERE id = ?
                """,
                (question_ref, question_json, slot_id),
            )
            self._conn.execute(
                """
                INSERT INTO journey_question_fingerprints
                    (user_id, journey_id, slot_id, objective_id, fingerprint,
                     signature_json, embedding_json, perspective, question_text, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(slot_id) DO UPDATE SET
                    fingerprint = excluded.fingerprint,
                    signature_json = excluded.signature_json,
                    embedding_json = excluded.embedding_json,
                    perspective = excluded.perspective,
                    question_text = excluded.question_text
                """,
                (
                    user_id,
                    journey_id,
                    slot_id,
                    slot["objective_id"],
                    fingerprint,
                    json.dumps(sorted(signature)),
                    json.dumps(embedding) if embedding else "",
                    perspective,
                    question_text,
                    now_iso,
                ),
            )
            self._conn.execute(
                "UPDATE journeys SET updated_at = ? WHERE id = ?", (now_iso, journey_id)
            )
        row = self._conn.execute(
            "SELECT q.*, o.title AS objective_title FROM journey_questions q "
            "LEFT JOIN journey_objectives o ON o.id = q.objective_id WHERE q.id = ?",
            (slot_id,),
        ).fetchone()
        return self._slot_row_to_dict(row)

    def reject_slot(
        self,
        *,
        user_id: str,
        journey_id: str,
        slot_id: str,
        reason: str,
    ) -> dict[str, Any]:
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            slot = self._require_slot_for_update(journey_id=journey_id, slot_id=slot_id)
            if slot["status"] != "reserved":
                raise JourneyValidationError(
                    f"slot {slot_id!r} is {slot['status']}, expected reserved"
                )
            self._conn.execute(
                "UPDATE journey_questions SET status = 'rejected', rejection_reason = ? WHERE id = ?",
                (reason, slot_id),
            )
            self._conn.execute(
                "UPDATE journeys SET updated_at = ? WHERE id = ?", (now_iso, journey_id)
            )
        row = self._conn.execute(
            "SELECT q.*, o.title AS objective_title FROM journey_questions q "
            "LEFT JOIN journey_objectives o ON o.id = q.objective_id WHERE q.id = ?",
            (slot_id,),
        ).fetchone()
        return self._slot_row_to_dict(row)

    def _require_slot_for_update(self, *, journey_id: str, slot_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM journey_questions WHERE id = ? AND journey_id = ?",
            (slot_id, journey_id),
        ).fetchone()
        if row is None:
            raise JourneyValidationError(
                f"question slot {slot_id!r} does not belong to journey {journey_id!r}"
            )
        return row

    def set_batch_generation_counts(
        self,
        *,
        user_id: str,
        journey_id: str,
        batch_id: str,
        accepted_count: int,
        rejected_count: int,
        status: str = "completed",
    ) -> dict[str, Any]:
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE journey_batches
                SET accepted_count = ?, rejected_count = ?, status = ?, updated_at = ?
                WHERE id = ? AND journey_id = ?
                """,
                (accepted_count, rejected_count, status, now_iso, batch_id, journey_id),
            )
            row = self._conn.execute(
                "SELECT * FROM journey_batches WHERE id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise JourneyNotFoundError(f"batch {batch_id!r} not found")
        return self._batch_row_to_dict(row)


    # ------------------------------------------------------------------
    # question progress (issue #5)
    #
    # Semantics contract:
    # * viewed/studied/practiced/mastered are distinct ladder states with
    #   their own timestamps; completion is an explicit boolean the learner
    #   controls via the completion endpoint.
    # * Progress percentages derive from DISTINCT completed slots in the
    #   active plan - never from incrementing counters - so duplicate
    #   submits, retries and replans cannot inflate them.
    # * mastered = completed AND at least two correct attempts AND the
    #   latest attempt correct; a fresh incorrect attempt clears it again.
    # * Retention scheduling stays with the existing FSRS store; nothing
    #   here builds a second scheduler.
    # ------------------------------------------------------------------
    VALID_ATTEMPT_OUTCOMES = ("correct", "incorrect", "skipped")

    @staticmethod
    def _state_row_to_dict(row: sqlite3.Row | None, *, slot_id: str) -> dict[str, Any]:
        if row is None:
            return {
                "slot_id": slot_id,
                "viewed": False,
                "studied": False,
                "practiced": False,
                "mastered": False,
                "completed": False,
                "viewed_at": "",
                "studied_at": "",
                "practiced_at": "",
                "mastered_at": "",
                "completed_at": "",
                "revision": 0,
                "updated_at": "",
            }
        return {
            "slot_id": row["slot_id"],
            "viewed": bool(row["viewed_at"]),
            "studied": bool(row["studied_at"]),
            "practiced": bool(row["practiced_at"]),
            "mastered": bool(row["mastered_at"]),
            "completed": bool(row["completed"]),
            "viewed_at": row["viewed_at"],
            "studied_at": row["studied_at"],
            "practiced_at": row["practiced_at"],
            "mastered_at": row["mastered_at"],
            "completed_at": row["completed_at"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }

    def _get_state_row_locked(self, *, user_id: str, journey_id: str, slot_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM journey_question_states WHERE user_id = ? AND journey_id = ? AND slot_id = ?",
            (user_id, journey_id, slot_id),
        ).fetchone()

    def _require_slot_locked(self, *, journey_id: str, slot_id: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM journey_questions WHERE id = ? AND journey_id = ?",
            (slot_id, journey_id),
        ).fetchone()
        if row is None:
            raise JourneyValidationError(
                f"question slot {slot_id!r} does not belong to journey {journey_id!r}"
            )
        return row

    def _write_state_locked(
        self,
        *,
        user_id: str,
        journey_id: str,
        slot_id: str,
        fields: dict[str, Any],
        revision: int,
        now_iso: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO journey_question_states
                (user_id, journey_id, slot_id, viewed_at, studied_at, practiced_at,
                 mastered_at, completed, completed_at, revision, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, journey_id, slot_id) DO UPDATE SET
                viewed_at = excluded.viewed_at,
                studied_at = excluded.studied_at,
                practiced_at = excluded.practiced_at,
                mastered_at = excluded.mastered_at,
                completed = excluded.completed,
                completed_at = excluded.completed_at,
                revision = excluded.revision,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                journey_id,
                slot_id,
                fields["viewed_at"],
                fields["studied_at"],
                fields["practiced_at"],
                fields["mastered_at"],
                fields["completed"],
                fields["completed_at"],
                revision,
                now_iso,
            ),
        )

    def _recompute_mastery_locked(self, *, user_id: str, journey_id: str, slot_id: str, completed: bool) -> bool:
        if not completed:
            return False
        rows = self._conn.execute(
            """
            SELECT outcome FROM journey_question_attempts
            WHERE user_id = ? AND journey_id = ? AND slot_id = ?
            ORDER BY created_at DESC, rowid DESC
            """,
            (user_id, journey_id, slot_id),
        ).fetchall()
        if not rows:
            return False
        correct = sum(1 for row in rows if row["outcome"] == "correct")
        return correct >= 2 and rows[0]["outcome"] == "correct"

    def get_journey_progress(self, *, user_id: str, journey_id: str) -> dict[str, Any]:
        """Progress summary + per-slot states for the active plan.

        Percentages count DISTINCT completed slots among active (non-retired)
        objectives only; retired slots keep their states but leave the
        denominator, so a curriculum upgrade never fabricates or destroys
        progress.
        """
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        active_slots = self._conn.execute(
            """
            SELECT q.id FROM journey_questions q
            JOIN journey_objectives o ON o.id = q.objective_id
            WHERE q.journey_id = ? AND o.status != 'retired'
            """,
            (journey_id,),
        ).fetchall()
        active_ids = [row["id"] for row in active_slots]
        states: list[dict[str, Any]] = []
        completed = 0
        studied = 0
        practiced = 0
        mastered = 0
        for slot_id in active_ids:
            row = self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id)
            state = self._state_row_to_dict(row, slot_id=slot_id)
            states.append(state)
            completed += 1 if state["completed"] else 0
            studied += 1 if state["studied"] else 0
            practiced += 1 if state["practiced"] else 0
            mastered += 1 if state["mastered"] else 0
        total = len(active_ids)
        percent = round(100.0 * completed / total, 1) if total else 0.0
        return {
            "journey_id": journey_id,
            "total_questions": total,
            "completed": completed,
            "studied": studied,
            "practiced": practiced,
            "mastered": mastered,
            "percent_complete": percent,
            "states": states,
        }

    def set_question_completion(
        self,
        *,
        user_id: str,
        journey_id: str,
        slot_id: str,
        completed: bool,
        mutation_id: str,
    ) -> dict[str, Any]:
        """Explicit complete/uncomplete. Idempotent per mutation_id."""
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            replay = self._conn.execute(
                "SELECT result_revision FROM journey_mutations WHERE user_id = ? AND journey_id = ? AND mutation_id = ?",
                (user_id, journey_id, mutation_id),
            ).fetchone()
            if replay is not None:
                state = self._state_row_to_dict(
                    self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                    slot_id=slot_id,
                )
                return {"state": state, "summary": self.get_journey_progress(user_id=user_id, journey_id=journey_id)}

            self._require_slot_locked(journey_id=journey_id, slot_id=slot_id)
            current = self._state_row_to_dict(
                self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                slot_id=slot_id,
            )
            new_revision = current["revision"] + 1
            mastered_at = current["mastered_at"]
            if not completed:
                mastered_at = ""
            fields = {
                "viewed_at": current["viewed_at"] or now_iso,
                "studied_at": current["studied_at"] or (now_iso if completed else ""),
                "practiced_at": current["practiced_at"],
                "mastered_at": mastered_at,
                "completed": 1 if completed else 0,
                "completed_at": now_iso if completed else "",
            }
            if completed and self._recompute_mastery_locked(
                user_id=user_id, journey_id=journey_id, slot_id=slot_id, completed=True
            ):
                fields["mastered_at"] = fields["mastered_at"] or now_iso
            self._write_state_locked(
                user_id=user_id, journey_id=journey_id, slot_id=slot_id,
                fields=fields, revision=new_revision, now_iso=now_iso,
            )
            self._conn.execute(
                """
                INSERT INTO journey_mutations
                    (user_id, journey_id, mutation_id, kind, result_revision, created_at)
                VALUES (?,?,?, 'completion', ?, ?)
                """,
                (user_id, journey_id, mutation_id, new_revision, now_iso),
            )
            state = self._state_row_to_dict(
                self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                slot_id=slot_id,
            )
        return {"state": state, "summary": self.get_journey_progress(user_id=user_id, journey_id=journey_id)}

    def record_question_attempt(
        self,
        *,
        user_id: str,
        journey_id: str,
        slot_id: str,
        outcome: str,
        confidence: int = 0,
        response_time_ms: int = 0,
        mutation_id: str,
    ) -> dict[str, Any]:
        """Record one practice attempt. Idempotent per mutation_id."""
        if outcome not in self.VALID_ATTEMPT_OUTCOMES:
            raise JourneyValidationError(
                f"outcome must be one of {', '.join(self.VALID_ATTEMPT_OUTCOMES)}"
            )
        self._require_journey_row(user_id=user_id, journey_id=journey_id)
        now_iso = _to_iso(_utc_now())
        with self._lock, self._conn:
            replay = self._conn.execute(
                "SELECT result_revision FROM journey_mutations WHERE user_id = ? AND journey_id = ? AND mutation_id = ?",
                (user_id, journey_id, mutation_id),
            ).fetchone()
            if replay is not None:
                state = self._state_row_to_dict(
                    self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                    slot_id=slot_id,
                )
                return {"state": state, "summary": self.get_journey_progress(user_id=user_id, journey_id=journey_id)}

            self._require_slot_locked(journey_id=journey_id, slot_id=slot_id)
            self._conn.execute(
                """
                INSERT INTO journey_question_attempts
                    (id, user_id, journey_id, slot_id, outcome, confidence, response_time_ms, mutation_id, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    f"att_{uuid.uuid4().hex[:24]}",
                    user_id,
                    journey_id,
                    slot_id,
                    outcome,
                    int(confidence),
                    int(response_time_ms),
                    mutation_id,
                    now_iso,
                ),
            )
            current = self._state_row_to_dict(
                self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                slot_id=slot_id,
            )
            mastered_at = current["mastered_at"]
            if outcome == "incorrect":
                mastered_at = ""
            fields = {
                "viewed_at": current["viewed_at"] or now_iso,
                "studied_at": current["studied_at"],
                "practiced_at": current["practiced_at"] or now_iso,
                "mastered_at": mastered_at,
                "completed": 1 if current["completed"] else 0,
                "completed_at": current["completed_at"],
            }
            if self._recompute_mastery_locked(
                user_id=user_id, journey_id=journey_id, slot_id=slot_id, completed=bool(fields["completed"])
            ):
                fields["mastered_at"] = fields["mastered_at"] or now_iso
            new_revision = current["revision"] + 1
            self._write_state_locked(
                user_id=user_id, journey_id=journey_id, slot_id=slot_id,
                fields=fields, revision=new_revision, now_iso=now_iso,
            )
            self._conn.execute(
                """
                INSERT INTO journey_mutations
                    (user_id, journey_id, mutation_id, kind, result_revision, created_at)
                VALUES (?,?,?, 'attempt', ?, ?)
                """,
                (user_id, journey_id, mutation_id, new_revision, now_iso),
            )
            state = self._state_row_to_dict(
                self._get_state_row_locked(user_id=user_id, journey_id=journey_id, slot_id=slot_id),
                slot_id=slot_id,
            )
        return {"state": state, "summary": self.get_journey_progress(user_id=user_id, journey_id=journey_id)}
