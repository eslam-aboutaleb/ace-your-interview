# Plan: Analytics, Gamification, and Timed Exam Mode

## Goal
Add a study-analytics dashboard (activity heatmap, retention curve, time-to-mastery, due forecast), gamification (XP, streaks, daily goals, badges), and a timed exam mode with auto-grading and a deterministic "remark" re-grade.

## Context
- Analytics today: `interview-sessions/stats`, `/trends`, `/api/learning/mastery`, `weak-areas`. No heatmap, no retention-over-time, no gamification.
- Depends on `1790987112911-fsrs-spaced-repetition-engine.md` (FSRS review log drives retention curves and the due forecast) and `1790987112911-flashcard-decks-anki-interop.md` (exam questions can come from decks).
- `learning_attempts` is the immutable event source for activity/XP.

## Decisions (resolved)
- Timed exams: server-side timestamps (client timer is never trusted); MCQ auto-graded locally, written answers LLM-graded with the same rubric as interviews; remark re-grades with an identical rubric + fixed seed and shows a diff.
- Gamification: XP from `learning_attempts` + exam completion + card reviews; streaks computed in the user's timezone (tz from `LearnerProfile`); daily goals default 3 reviews/day, configurable.
- Heatmap: GitHub-style 52-week activity from `learning_attempts` + `fsrs_review_log` + exam events.

## Data model
```sql
CREATE TABLE IF NOT EXISTS exam_sessions (
    user_id TEXT NOT NULL, exam_id TEXT NOT NULL,
    title TEXT NOT NULL, question_refs_json TEXT NOT NULL,
    duration_minutes INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('created','in_progress','completed')),
    started_at TEXT, completed_at TEXT,
    score REAL, passed INTEGER,
    created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, exam_id)
);
CREATE TABLE IF NOT EXISTS exam_answers (
    user_id TEXT NOT NULL, exam_id TEXT NOT NULL,
    question_ref TEXT NOT NULL, answer_text TEXT NOT NULL,
    is_correct INTEGER, score REAL, graded_by TEXT,   -- 'auto' | 'llm'
    rubric_json TEXT, created_at TEXT NOT NULL,
    PRIMARY KEY (user_id, exam_id, question_ref)
);
CREATE TABLE IF NOT EXISTS xp_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL, event_key TEXT NOT NULL UNIQUE,  -- idempotency
    event_type TEXT NOT NULL, xp INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gamification_state (
    user_id TEXT PRIMARY KEY, total_xp INTEGER NOT NULL DEFAULT 0,
    current_streak INTEGER NOT NULL DEFAULT 0,
    longest_streak INTEGER NOT NULL DEFAULT 0,
    last_activity_date TEXT,            -- user-tz date
    daily_goal INTEGER NOT NULL DEFAULT 3,
    badges_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
```

## API changes
- `POST /api/exams` `{title, source: {topic_ids?[], deck_ids?[], weak_areas?bool}, question_count, duration_minutes, question_types: ['mcq','written']}` → 201 exam (questions pre-selected from weak areas + due cards).
- `POST /api/exams/{id}/start` → `{started_at, duration_minutes, questions[]}` (server stamps the start).
- `POST /api/exams/{id}/submit` `{answers: [{question_ref, answer_text, time_ms}]}` → grades MCQ locally, written via LLM rubric; returns `{score, passed, per_question[]}`.
- `POST /api/exams/{id}/remark` → re-grade with identical rubric + fixed seed; returns `{original_score, new_score, diff[], changed[]}`.
- `GET /api/exams` → history; `GET /api/exams/{id}` → detail + report.
- `GET /api/gamification/summary` → `{total_xp, level, current_streak, longest_streak, daily_goal, today_progress, badges[]}`; `PUT /api/gamification/goals` `{daily_goal}`.
- `GET /api/analytics/heatmap?weeks=52` → `{days: [{date, count, types}]}`.
- `GET /api/analytics/retention` → recall rate at 1d/7d/30d buckets from `fsrs_review_log`.
- `GET /api/analytics/mastery-curve` → topic mastery over time (from `learning_attempts` EMA history).
- `GET /api/analytics/due-forecast` → delegates to the plan-2 forecast endpoint.

## Implementation tasks (ordered)
1. DDL + `ExamStore`, `GamificationStore` services (run_async pattern); register in `main.py`.
2. `backend/app/services/exam_service.py`: question selection (weak areas via `get_weak_areas`, due cards via FSRS), MCQ grading, written grading via an interview-style rubric, remark with fixed seed.
3. `backend/app/services/gamification_service.py`: XP awarding on learning events (hook into `LearningStore.record_attempt` and exam completion), streak computation in user tz, badge rules (first review, 7-day streak, 100 cards, first exam, exam passed).
4. `backend/app/services/analytics_service.py`: heatmap aggregation, retention buckets, mastery curve.
5. Routers: `backend/app/routers/exams.py`, `routers/gamification.py`, `routers/analytics.py`; register in `main.py`.
6. Frontend: new `frontend/src/pages/AnalyticsPage.tsx` (`/analytics`) with heatmap grid, retention chart, mastery curve (lightweight SVG or recharts; framer-motion animations); new `ExamPage.tsx` (`/exams`, `/exams/:id`) with a server-synced countdown; gamification widget on `Dashboard.tsx` (streak flame, XP progress bar, badge shelf).
7. Tests: `test_exam_service.py` (selection from weak areas, MCQ grading, remark determinism), `test_gamification_service.py` (streak UTC-midnight vs user-tz boundaries, XP idempotency via `event_key`), `test_analytics_service.py` (heatmap counts, retention buckets), router contract tests.

## Failure modes
- Client timer manipulation → server `started_at`/`completed_at` authoritative; late submits flagged `time_exceeded` but still graded.
- XP double-counting → `event_key` UNIQUE constraint; award is idempotent.
- Streak timezone → store `last_activity_date` as a user-tz date string; recompute on read.
- Remark non-determinism → fixed seed + identical rubric template; the diff is surfaced to the user and both scores are stored.
- LLM grading cost → written-answer grading reuses the budget machinery; exam size capped at 50 questions.

## Rollout
1. Ship behind `STUDY_ENABLE_EXAMS_V1` and `STUDY_ENABLE_GAMIFICATION_V1` (both default `true`).
2. Analytics endpoints unflagged (read-only, derived from existing tables).
3. Monitor exam grading latency; adjust question caps.

## Validation
- Remark determinism: same exam + answers → identical re-grade (fixed seed).
- Streak: activity at 23:59 and 00:01 user-tz counts as same/adjacent days correctly.
- Heatmap: counts match raw `learning_attempts` for a fixture user.
- `pytest tests/test_exam_service.py tests/test_gamification_service.py tests/test_analytics_service.py`
