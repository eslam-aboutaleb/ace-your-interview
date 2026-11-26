# Learning journeys API (v1)

Durable learning journeys (issue #4): versioned plans, immutable question
batches, and optimistic reader checkpoints. This is the persistence and API
foundation the later learning-journey stages (novelty generation #2, typed
answers #3, progress semantics #5, reader UI #7-#13) build on.

## Rollout

The whole surface is behind a flag and returns 404 while disabled:

```
STUDY_ENABLE_JOURNEYS_V1=true
```

Default is off. Enable it in an environment only together with the reader UI
stage that consumes the API; there is no frontend for journeys yet.

Rollback: set the flag to `false`. All journey tables are additive and
separate from existing tables, so disabling the flag fully hides the feature
without touching any pre-existing data.

## Data model

All tables live in the existing learning SQLite database
(`settings.learning_db_path`) and are created idempotently at startup by
`JourneyStore` (`app/services/journey_store.py`). No Alembic-style migration
runner exists in this codebase; table creation follows the same
`CREATE TABLE IF NOT EXISTS` pattern as the other stores.

* `journeys` - one row per journey (uuid `jrn_...`), `plan_version`,
  `provenance_json`, status `active|completed|archived`.
* `journey_phases` / `journey_modules` / `journey_objectives` - the plan
  tree. IDs are stable content hashes over `journey_id` + title + position,
  so a replan keeps unchanged objectives addressable.
* `journey_batches` - immutable, numbered per journey
  (`UNIQUE(journey_id, batch_number)`), with
  `UNIQUE(journey_id, idempotency_key)` and prompt/model/schema version
  columns that generation stages fill in.
* `journey_questions` - reserved question slots, ordered per batch
  (`UNIQUE(batch_id, position)`). Slots reference objectives; question
  payloads land here in the generation stages.
* `journey_checkpoints` - one row per (user, journey): last batch/question/
  section, scroll offset, and an integer `revision` for optimistic
  concurrency.
* `journey_mutations` - idempotency log: applied `mutation_id`s with their
  resulting revision.

## API contract

Base path `/api/journeys`, auth required (session cookie), JSON.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/journeys` | List the caller's journeys (newest first). |
| `POST /api/journeys` | Create from `static_topic` or `custom_topic`; the plan is derived server-side from the source's sections (one module per section, one objective per bullet, capped at 8 per section). Accepts an optional client-edited `plan` (from a preview) plus `plan_provenance` echo (#6); the store re-stamps stable IDs on commit. 201 with the full detail. |
| `POST /api/journeys/plan-preview` | Goal-driven syllabus preview (#6): goal + target level + weekly hours + depth (`survey`/`standard`/`deep`) + source. Shapes the baseline plan by depth caps, marks review obligations from the topic's active FSRS cards, estimates pacing. Persists nothing; echoes inputs as `provenance.inputs` for the commit call. |
| `GET /api/journeys/{id}` | Full detail with phases/modules/objectives. |
| `POST /api/journeys/{id}/refresh` | Replan from current source content. Bumps `plan_version` only on change; retires removed objectives, revives returning ones, never deletes. |
| `POST /api/journeys/{id}/batches` | Reserve the next batch (`idempotency_key`, `slot_count` 1-50, default 10). Transactional numbering; replaying a key returns the original batch with `created=false`. 409 when no pending objectives remain. |
| `GET /api/journeys/{id}/batches` | Batches in order. |
| `GET /api/journeys/{id}/reader-state` | Checkpoint + latest batch with ordered slots. |
| `PATCH /api/journeys/{id}/checkpoints` | Optimistic write: `expected_revision` + `mutation_id`; omitted fields keep their values. 409 with `{message, current}` on a stale revision; replaying a `mutation_id` returns current state unchanged. Referenced batch/question IDs must belong to the journey (422). |

Error shape: FastAPI `{"detail": ...}`; conflicts use 409, unknown journeys
404 (existence is never leaked across users), invalid references 422.

## Ownership and security

Every store method filters by `user_id`; cross-user access is
indistinguishable from "not found". All writes run on a single-writer
executor thread with SQLite transactions; uniqueness constraints are the
backstop for concurrent batch reservation and mutation replay.

## Observability

Journey writes reuse the standard request logging. Batch creation returns
`accepted_count`/`rejected_count` columns (0 until generation lands) so the
generation stages can report reservation outcomes per batch without schema
changes.

## Plan preview and onboarding (#6)

`POST /api/journeys/plan-preview` is the commitment-free first half of
onboarding: it returns the shaped syllabus, pacing estimate and provenance
echo; the learner edits the tree client-side and commits through
`POST /api/journeys` with `plan` + `plan_provenance`. Edited plans are
validated server-side (structure, bounds, kinds) and re-stamped with
stable content-hash IDs, so replans still preserve progress. Review
obligations are topic-granular: any active FSRS card on the source topic
marks every objective `review_suggested` (FSRS has no per-section linkage
yet). Natural-language goal decomposition into a full syllabus is a
deliberate follow-up; the preview/commit contract already carries the
goal through provenance.

## Legacy progress

Existing `question_progress` rows are untouched. Journey objectives link
back to curriculum sections via `source_section`; the progress-semantics
stage (#5) reconciles legacy per-topic progress with journey objectives.
Nothing is migrated destructively: disabling the flag leaves all prior data
exactly as it was.

## Verification

```
cd backend
python -m pytest tests/test_journey_store.py tests/test_journeys_router.py -q
python -m pytest tests/test_plan_preview.py tests/test_plan_preview_router.py -q
python -m pytest tests/ --cov   # full suite, coverage gate 90%
python -m ruff check .
```
