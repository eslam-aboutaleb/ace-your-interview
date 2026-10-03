# Plan: FSRS-4.5 Spaced Repetition Engine

**Plan series index** (execution order):
1. `1790987112911-platform-fixes-ci-testing.md` — quick fixes, CI, frontend tests (independent, ship first)
2. `1790987112911-fsrs-spaced-repetition-engine.md` — **this file** (foundation for plans 2 and 5)
3. `1790987112911-document-ingestion-rag-chat.md` — sqlite-vec + RAG (enables duplicates in plan 2)
4. `1790987112911-flashcard-decks-anki-interop.md` — depends on plan 2 (FSRS cards)
5. `1790987112911-interview-personalization.md` — resume/JD, hints, STAR, voice interviews
6. `1790987112911-analytics-gamification-exam-mode.md` — depends on plan 2 (FSRS review log)

## Goal
Replace the fixed interval ladder in `backend/app/services/learning_store.py` with FSRS-4.5 via `py-fsrs`, and add leech detection, retention estimation, a 7-day due forecast, and question-difficulty calibration.

## Context
- Current scheduling: `_INTERVAL_DAYS = [0, 1, 3, 7, 14, 30]` with `review_bucket` 0–5 and `_next_bucket()` bumping by confidence (`learning_store.py:43,267-271`). No difficulty modulation, no lapse-aware intervals, no retention prediction.
- Competitors (Cortex, Pupil, KnowledgeLoom, flashcard-mcp) all ship FSRS — this is the core learning-science gap.
- `learning_attempts` and `question_progress` tables live in SQLite at `settings.learning_db_path`; all store work runs on a single-thread executor via `run_async()`.
- Interview answers already flow into `record_attempt` (`routers/interview_sessions.py:380-390`) — those events become FSRS reviews.

## Decisions (resolved)
- Algorithm: **FSRS-4.5 via `py-fsrs`** (pin `fsrs>=0.10,<0.11` in `backend/pyproject.toml`).
- `learning_attempts` stays the immutable event log (source of truth for analytics/heatmaps).
- New `fsrs_cards` + `fsrs_review_log` tables; `question_progress` becomes read-only during migration, removed in a follow-up cleanup PR.
- Feature flag `STUDY_ENABLE_FSRS_V1` (default `true`) so the legacy ladder can be disabled instantly.
- Backfill: existing `question_progress` rows → FSRS cards, state derived from `review_bucket` + `mastery_score`; idempotent.

## Data model (DDL, added to `LearningStore._init_db`)
```sql
CREATE TABLE IF NOT EXISTS fsrs_cards (
    user_id TEXT NOT NULL,
    card_id TEXT NOT NULL,          -- question_id or flashcard_id
    topic_id TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK(source_type IN ('question','flashcard','interview','exam')),
    state TEXT NOT NULL DEFAULT 'new' CHECK(state IN ('new','learning','review','relearning')),
    stability REAL, difficulty REAL,
    due_at TEXT NOT NULL,
    last_review_at TEXT,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    scheduled_days INTEGER NOT NULL DEFAULT 0,
    elapsed_days INTEGER NOT NULL DEFAULT 0,
    suspended INTEGER NOT NULL DEFAULT 0,   -- leech suspension
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, card_id)
);
CREATE INDEX IF NOT EXISTS idx_fsrs_due ON fsrs_cards(user_id, due_at) WHERE suspended = 0;

CREATE TABLE IF NOT EXISTS fsrs_review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, card_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK(rating IN ('again','hard','good','easy')),
    state TEXT NOT NULL, review_duration_ms INTEGER NOT NULL,
    scheduled_days INTEGER, elapsed_days INTEGER,
    created_at TEXT NOT NULL
);
```

## API changes
- `POST /api/learning/review` — body `{card_id, topic_id, rating: again|hard|good|easy, response_time_ms, source_type}` → 201 `{card, log, leech: bool}`. Replaces confidence-based grading for cards; `POST /api/learning/attempts` remains for legacy question events.
- `GET /api/learning/review-queue` — extend items with `state`, `lapses`, `due_at`, `suspended`, `leech` (lapses ≥ 5).
- `GET /api/learning/forecast?days=7` → `{due_counts: [{date, count}], total_due}`.
- `GET /api/learning/calibration` → true retention (reviews with `elapsed_days >= 1` and rating ≥ good) vs the 80–90% band; `low_signal` when < 50 eligible reviews or > 80% Good ratings.
- `GET /api/learning/mastery` — response shape unchanged; mastery now derived from FSRS stability/difficulty.

## Implementation tasks (ordered)
1. Add `fsrs` dependency to `backend/pyproject.toml`; regenerate `uv.lock`.
2. New `backend/app/services/fsrs_scheduler.py`: facade over `fsrs.Scheduler` — `create_card()`, `review(card, rating, now, duration_ms) -> (card, log)`, `due_cards(user_id, limit)`, `forecast(user_id, days)`, `calibration(user_id)`, `is_leech(card)` (lapses ≥ 5). All FSRS calls go through this facade so library API drift is contained.
3. Extend `LearningStore._init_db` with the DDL above; add `record_review()`, `get_due_cards()`, `get_forecast()`, `get_calibration()`, `backfill_fsrs_from_progress()` (batched 500 rows/txn, `INSERT OR IGNORE` on `(user_id, card_id)`).
4. New schemas in `backend/app/schemas/models.py`: `LearningReviewRequest`, `LearningReviewResponse`, `ForecastResponse`, `CalibrationResponse`; extend `ReviewQueueItem`.
5. Wire routes in `backend/app/routers/learning.py` (all behind `_ensure_enabled()` + flag check).
6. Startup migration: run `backfill_fsrs_from_progress()` when flag on and `fsrs_cards` empty; log before/after counts.
7. Frontend: `frontend/src/services/api.ts` add `submitCardReview`, `fetchForecast`, `fetchCalibration`; `ReviewQueue.tsx` renders Again/Hard/Good/Easy buttons replacing correct/incorrect; `Dashboard.tsx` adds a due-forecast widget.
8. Tests: `backend/tests/test_fsrs_scheduler.py` (known FSRS vectors from the library's own expected values), `test_learning_store_fsrs.py` (backfill idempotency, leech flag at 5 lapses, forecast counts), `test_learning_router_fsrs.py` (contract).

## Failure modes
- py-fsrs API drift → exact version pin + facade module.
- Clock skew on `due_at` → always `datetime.now(UTC)`.
- Backfill double-run → idempotent `INSERT OR IGNORE`.
- Flag off → legacy ladder path must remain functional (keep `_next_bucket` until cleanup PR).

## Rollout / migration
1. Ship with flag on but legacy path intact; backfill runs at startup.
2. Dev: verify backfill counts vs `question_progress` row counts per user.
3. Prod: enable, monitor `/api/learning/review-queue` latency (new partial index on `due_at`).
4. Follow-up cleanup PR removes the legacy ladder and `question_progress` writes.

## Validation
- `pytest tests/test_fsrs_scheduler.py tests/test_learning_store_fsrs.py tests/test_learning_router_fsrs.py`
- Property test: for random rating sequences, `due_at` after Again < Hard < Good < Easy.
- Manual: review 10 cards in the UI, confirm the forecast widget updates and leech flag appears at 5 lapses.
