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
| `POST /api/journeys` | Create from `static_topic` or `custom_topic`; the plan is derived server-side from the source's sections (one module per section, one objective per bullet, capped at 8 per section). 201 with the full detail. |
| `GET /api/journeys/{id}` | Full detail with phases/modules/objectives. |
| `POST /api/journeys/{id}/refresh` | Replan from current source content. Bumps `plan_version` only on change; retires removed objectives, revives returning ones, never deletes. |
| `POST /api/journeys/{id}/batches` | Reserve the next batch (`idempotency_key`, `slot_count` 1-50, default 10). Transactional numbering; replaying a key returns the original batch with `created=false`. 409 when no pending objectives remain. |
| `GET /api/journeys/{id}/batches` | Batches in order. |
| `GET /api/journeys/{id}/reader-state` | Checkpoint + latest batch with ordered slots. |
| `POST /api/journeys/{id}/batches/{batch_id}/generate` | Fill the batch's reserved slots with novelty-checked questions (#2). Runs the dedup funnel per candidate, regenerates only rejected slots (bounded rounds), persists fingerprints + a generation job with prompt/model/schema versions and accepted/rejected counts. 409 when the batch has no reserved slots left. |
| `PATCH /api/journeys/{id}/checkpoints` | Optimistic write: `expected_revision` + `mutation_id`; omitted fields keep their values. 409 with `{message, current}` on a stale revision; replaying a `mutation_id` returns current state unchanged. Referenced batch/question IDs must belong to the journey (422). |
| `PUT /api/journeys/{id}/questions/{slot_id}/completion` | Explicit complete/uncomplete (#5). `mutation_id` makes retries and double-taps safe: a replay returns the recorded state unchanged. |
| `POST /api/journeys/{id}/questions/{slot_id}/attempts` | Record one practice attempt (`outcome` correct/incorrect/skipped, optional `confidence` 0-5 and `response_time_ms`, `mutation_id`). |
| `GET /api/journeys/{id}/progress` | Summary over DISTINCT active question slots (never event counts) plus per-slot states. |

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

## Novelty-aware generation (#2)

`POST .../batches/{id}/generate` fills reserved slots through a four-stage
dedup funnel (`app/services/novelty.py`), strictest first:

1. exact normalized hash (sha1 over the shared question normaliser);
2. character near-duplicate (containment + sequence ratio 0.87, mirroring
   the in-session dedup the generator already ships);
3. concept equivalence (content-token Jaccard 0.65) - skipped when the
   candidate and the seen question carry intentionally distinct
   `perspective` values, so legitimate deepening survives;
4. calibrated embedding similarity (cosine, default reject >= 0.92) with a
   review band [0.85, 0.92) for optional LLM adjudication
   (`STUDY_ENABLE_JOURNEY_GENERATION_ADJUDICATION`, default off; without an
   adjudicator the band rejects conservatively). Embeddings are
   best-effort: a provider failure degrades to no-embedding, never fails
   the batch.

Every accepted question persists a fingerprint row (hash, concept
signature, optional embedding, perspective, text) in
`journey_question_fingerprints`, so dedup survives restarts and spans
batches. Only rejected slots are regenerated (bounded at 3 rounds per
slot); accepted slots are never re-generated, and re-calling generate on
a finished batch returns 409. Each run is a `journey_generation_jobs` row
with prompt/model/schema versions and accepted/rejected counts that roll
up onto the batch (`completed` when nothing was rejected, else `partial`).

Batches accept an optional `perspectives` list; slots cycle through it and
the perspective is recorded on the slot, passed to the generator (as a
section-title suffix) and honored by the funnel.


## Progress semantics (#5)

Per-slot states live in `journey_question_states`; every practice event is
appended to `journey_question_attempts`.

* `viewed` is set on any interaction; `studied` on completion; `practiced`
  on any attempt. Each is a distinct ladder state with its own timestamp.
* Completion is an explicit learner action (complete/uncomplete), never an
  inferred counter.
* `mastered` requires completion AND at least two correct attempts AND the
  latest attempt correct. A fresh incorrect attempt clears it; `skipped`
  attempts never clear it.
* Summary percentages count DISTINCT completed slots among active
  (non-retired) objectives, so retries, replays and duplicate submits
  cannot inflate progress. A replan that retires an objective removes its
  slots from the denominator but keeps the recorded states per-slot.
* Retention scheduling stays with the existing FSRS store; this surface
  does not build a second scheduler.
* Both writes are idempotent per `mutation_id` (replayed through
  `journey_mutations`), matching the checkpoint contract.

## Legacy progress

Existing `question_progress` rows are untouched. Journey objectives link
back to curriculum sections via `source_section`. The legacy wiring is
deliberately deferred: joining `TopicStudy` progress to journey slots now
would be throwaway work before the reader slice (#7+) defines how slots
surface content. Disabling the flag leaves all prior data exactly as it
was.

## Verification

```
cd backend
python -m pytest tests/test_journey_store.py tests/test_journeys_router.py -q
python -m pytest tests/test_novelty.py tests/test_journey_generation.py tests/test_journey_generation_router.py -q


python -m pytest tests/test_journey_progress.py tests/test_journey_progress_router.py -q
python -m pytest tests/ --cov   # full suite, coverage gate 90%
python -m ruff check .
```
