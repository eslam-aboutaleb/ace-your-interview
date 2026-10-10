# Plan D — Consolidate Spaced Repetition onto FSRS (P1)

**Wave 2 · Depends on Plan A (both edit `interview_sessions.py`)**

## Goal

Two schedulers run in parallel: real FSRS-4.5 (`fsrs_cards`, wired to flashcards and the manual review queue) and a legacy bucket engine (`question_progress`, `_INTERVAL_DAYS = [0,1,3,7,14,30]`, written by every quiz attempt and interview turn). Route quiz mistakes and interview results into FSRS and retire the legacy write path — the exact "second independent scheduler" the review warns against.

## Current state (verified)

- `learning_store.py:347-469` — `record_attempt` writes only `learning_attempts` + `question_progress`; never touches `fsrs_cards`.
- `learning_store.py:537-657` — `record_review` is the only FSRS write path (flashcards, `cards.py:561-569`; manual `POST /api/learning/review`).
- `learning_planner.py:440-458` — `_question_progress_rows` reads only `question_progress`; the planner has no FSRS visibility.
- `learning_store.py:808-908` — `backfill_fsrs_from_progress` skips globally once *any* `fsrs_cards` row exists (`:827-832`) — a pre-existing coverage gap for mixed-state databases.
- `routers/learning.py:104-111, 180-187` — flag-gated legacy fallbacks for review-queue and mastery.
- `interview_sessions.py:648-668` and `:801-821` — interview turns call `record_attempt` with `question_id=f"interview:{session_id}:{turn_index}"`, `is_correct=overall >= 70`, `confidence=confidence_signal`.

## Changes (ordered — follow the cutover sequence; each step is independently shippable)

1. **Dual-write in `record_attempt`** (`learning_store.py:347-469`): after the `learning_attempts` INSERT, derive an FSRS rating — incorrect → `again`; correct + confidence ≥4 → `easy`; correct → `good` — and upsert an `fsrs_cards` row via `fsrs_scheduler.review` (reuse the `record_review` upsert pattern at `:554-627`). `card_id` = `question_id`; `source_type` = `mode` (`study`/`quiz`/`interview` — all valid per the CHECK constraint at `:266`). Keep `question_progress` writes during the transition.
2. **Fix the backfill guard** (`:827-832`): drop the global `card_count > 0` skip; rely on per-row `INSERT OR IGNORE` keyed on `(user_id, card_id)` so every user's legacy rows migrate.
3. **Repoint reads**: extend `get_fsrs_topic_mastery` (`:765-806`) with `accuracy` and `due_count` aggregates from `fsrs_cards`/`fsrs_review_log`; repoint `learning_planner._question_progress_rows` (`:440-458`) and `LearningStore.get_weak_areas` (`:910-939`) at the FSRS tables.
4. **Retire legacy writes**: remove the `question_progress` INSERT/UPDATE from `record_attempt`; delete `_INTERVAL_DAYS` (`:44`), `_next_bucket` (`:341-345`), `get_review_queue` (`:471-495`); remove the legacy fallbacks in `routers/learning.py` (`:104-111`, `:180-187`) and the `enable_fsrs_v1` gate on the attempts path.
5. **Response shape**: `LearningAttemptResponse` (`models.py:484-490`) returns `mastery_score`/`due_at`/`review_bucket` — replace with the FSRS card state (`state`, `due_at`, `lapses`, `leech`). Frontend consumers: `QuizMode.tsx:450-453` (`res.mastery_score`), `progressStore.ts:115-146` (client-side EWMA mirror — keep, it is display-only).
6. **Tests** — update: `test_coverage_learning_store.py:95-137` (direct `question_progress` SELECTs), `test_learning_router_fsrs.py:227-241` (legacy `review_bucket` in response), `test_coverage_learning_router.py:191-216` (legacy fallbacks) and `:239-287` (backfill), `test_learning_store_fsrs.py:301-412` (backfill guard), `test_learning_planner.py:35-68` (diagnostic from attempt data — now via FSRS), `test_learning_store_study_plan.py`. Add: rating-derivation table test (again/hard/good/easy mapping), interview-turn → FSRS card test, backfill per-user idempotency test.

## Failure modes

- The rating derivation (`is_correct`/`confidence` → FSRS rating) is a heuristic — document it in the module docstring; it is the standard approximation when a learner self-reports confidence.
- Dual-write transition: if step 4 ships before step 3, the planner goes cold-start — ship steps 1–3 first, verify diagnostics, then step 4.
- `fsrs_cards.source_type` CHECK allows `question`/`flashcard`/`interview`/`exam` — `mode` values from `LearningMode` are `study`/`quiz`; map `study`→`question`.
- Existing user data: run the fixed backfill once (step 2) before retiring legacy writes.

## Validation

- Backend suite + ruff; specifically `test_learning_*`, `test_fsrs_scheduler.py` (known FSRS-4.5 vectors must stay green).
- Manual: fresh DB → attempt → card appears in `GET /api/learning/review-queue`; legacy DB → restart → backfill migrates per-user.

## Out of scope

Dropping the `question_progress` table (keep it read-only for a soak period), per-user FSRS parameter tuning, `exam` source-type wiring.
