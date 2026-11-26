"""Tests for generation job orchestration (issue #16).

Covers the reliability contract: job states, idempotent submission keys,
per-item persistence before the next attempt, provider timeouts, malformed
results, worker crashes, cooperative cancellation, idempotent retry of
failed items only, cost limits, capability detection, and per-user
isolation.
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import unittest

from app.services.generation_jobs import (
    JOB_CANCELLED,
    JOB_COMPLETED,
    JOB_FAILED,
    JOB_PARTIAL,
    GenerationJobNotFoundError,
    GenerationJobStateError,
    GenerationJobStore,
    GenerationOrchestrator,
)

USER = "learner"


def ok_generator(item):
    async def _run():
        return {"text": f"generated: {item}"}

    return _run()


class GenerationJobsTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmpdir.name, "learning.db")
        self.store = GenerationJobStore(self.db)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _orchestrator(self, generator_fn, **kwargs):
        return GenerationOrchestrator(self.store, generator_fn, **kwargs)

    def _create(self, outline=("a", "b", "c"), key="key-1", user_id=USER):
        job, created = self.store.create_job(
            user_id=user_id, idempotency_key=key, outline=list(outline)
        )
        self.assertTrue(created)
        return job


class StoreTests(GenerationJobsTestBase):
    def test_create_is_idempotent_per_key(self):
        first = self._create()
        second, created = self.store.create_job(
            user_id=USER, idempotency_key="key-1", outline=["x", "y"]
        )
        self.assertFalse(created)
        self.assertEqual(second["id"], first["id"])
        # The duplicate submission keeps the original outline: no re-charge.
        self.assertEqual(second["outline"], ["a", "b", "c"])

    def test_same_key_for_another_user_is_a_separate_job(self):
        first = self._create()
        other, created = self.store.create_job(
            user_id="someone-else", idempotency_key="key-1", outline=["z"]
        )
        self.assertTrue(created)
        self.assertNotEqual(other["id"], first["id"])

    def test_get_job_enforces_ownership(self):
        job = self._create()
        with self.assertRaises(GenerationJobNotFoundError):
            self.store.get_job(user_id="intruder", job_id=job["id"])
        with self.assertRaises(GenerationJobNotFoundError):
            self.store.get_job(user_id=USER, job_id="job_missing")

    def test_cancel_terminal_job_rejected(self):
        job = self._create()
        self.store.cancel(user_id=USER, job_id=job["id"])
        with self.assertRaises(GenerationJobStateError):
            self.store.cancel(user_id=USER, job_id=job["id"])


class OrchestratorTests(GenerationJobsTestBase):
    def _run(self, orchestrator, job, user_id=USER):
        return asyncio.run(orchestrator.run_job(user_id=user_id, job_id=job["id"]))

    def test_completed_job_accepts_every_item(self):
        job = self._create()
        final = self._run(self._orchestrator(ok_generator), job)
        self.assertEqual(final["status"], JOB_COMPLETED)
        self.assertEqual(len(final["accepted"]), 3)
        self.assertEqual(final["cost_units"], 3)
        self.assertTrue(final["capabilities"].get("json_mode"))

    def test_partial_when_some_items_fail(self):
        def generator(item):
            if item == "b":
                raise ValueError("malformed JSON from provider")
            return {"text": item}

        async def _gen(item):
            return generator(item)

        job = self._create()
        final = self._run(self._orchestrator(_gen), job)
        self.assertEqual(final["status"], JOB_PARTIAL)
        self.assertEqual(len(final["accepted"]), 2)
        self.assertEqual(len(final["failed"]), 1)
        self.assertIn("malformed JSON", final["failed"][0]["error"])

    def test_failed_when_nothing_accepted(self):
        async def _gen(item):
            raise RuntimeError("provider down")

        job = self._create()
        final = self._run(self._orchestrator(_gen), job)
        self.assertEqual(final["status"], JOB_FAILED)
        self.assertEqual(len(final["accepted"]), 0)

    def test_provider_timeout_fails_one_item_not_the_job(self):
        async def _gen(item):
            if item == "b":
                await asyncio.sleep(5)
            return {"text": item}

        job = self._create()
        final = self._run(self._orchestrator(_gen, item_timeout=0.05), job)
        self.assertEqual(final["status"], JOB_PARTIAL)
        self.assertEqual(final["failed"][0]["error"], "provider timeout")

    def test_cost_limit_stops_the_job(self):
        calls = []

        async def _gen(item):
            calls.append(item)
            return {"text": item}

        job = self._create(outline=("a", "b", "c", "d"))
        final = self._run(self._orchestrator(_gen, cost_limit=2), job)
        self.assertEqual(final["status"], JOB_PARTIAL)
        self.assertEqual(len(calls), 2)
        self.assertIn("cost limit", final["error"])

    def test_cancel_between_items_keeps_accepted(self):
        store = self.store

        async def _gen(item):
            # Cancel the job after the first item is accepted.
            store.cancel(user_id=USER, job_id=job["id"])
            return {"text": item}

        job = self._create()
        final = self._run(self._orchestrator(_gen), job)
        self.assertEqual(final["status"], JOB_CANCELLED)
        self.assertEqual(len(final["accepted"]), 1)

    def test_retry_only_reruns_failed_items(self):
        attempted = []

        async def _gen(item):
            attempted.append(item)
            if item == "b" and attempted.count("b") == 1:
                raise ValueError("transient")
            return {"text": item}

        job = self._create()
        orchestrator = self._orchestrator(_gen)
        first = self._run(orchestrator, job)
        self.assertEqual(first["status"], JOB_PARTIAL)
        self.assertEqual(attempted, ["a", "b", "c"])

        retried = asyncio.run(orchestrator.retry_job(user_id=USER, job_id=job["id"]))
        self.assertEqual(retried["status"], JOB_COMPLETED)
        # Only "b" was regenerated; accepted items were not re-charged.
        self.assertEqual(attempted.count("a"), 1)
        self.assertEqual(attempted.count("b"), 2)
        self.assertEqual(len(retried["accepted"]), 3)

    def test_retry_rejects_completed_job(self):
        job = self._create()
        orchestrator = self._orchestrator(ok_generator)
        self._run(orchestrator, job)
        with self.assertRaises(GenerationJobStateError):
            asyncio.run(orchestrator.retry_job(user_id=USER, job_id=job["id"]))

    def test_run_is_idempotent_after_completion(self):
        calls = []

        async def _gen(item):
            calls.append(item)
            return {"text": item}

        job = self._create()
        orchestrator = self._orchestrator(_gen)
        self._run(orchestrator, job)
        again = self._run(orchestrator, job)
        self.assertEqual(again["status"], JOB_COMPLETED)
        self.assertEqual(len(calls), 3)  # no re-charge

    def test_worker_crash_preserves_accepted_items(self):
        async def _gen(item):
            if item == "c":
                raise MemoryError("worker crash")
            return {"text": item}

        job = self._create()
        final = self._run(self._orchestrator(_gen), job)
        self.assertEqual(final["status"], JOB_PARTIAL)
        # Accepted items are durable in the store itself.
        raw = sqlite3.connect(self.db).execute(
            "SELECT accepted_json FROM generation_jobs WHERE id = ?", (job["id"],)
        ).fetchone()
        self.assertEqual(len(json.loads(raw[0])), 2)

    def test_concurrent_users_have_isolated_jobs(self):
        job_a = self._create(key="k", user_id="user-a")
        job_b = self._create(key="k", user_id="user-b")

        async def _run_both():
            orchestrator = self._orchestrator(ok_generator)
            return await asyncio.gather(
                orchestrator.run_job(user_id="user-a", job_id=job_a["id"]),
                orchestrator.run_job(user_id="user-b", job_id=job_b["id"]),
            )

        final_a, final_b = asyncio.run(_run_both())
        self.assertEqual(final_a["status"], JOB_COMPLETED)
        self.assertEqual(final_b["status"], JOB_COMPLETED)
        with self.assertRaises(GenerationJobNotFoundError):
            self.store.get_job(user_id="user-a", job_id=job_b["id"])


if __name__ == "__main__":
    unittest.main()
