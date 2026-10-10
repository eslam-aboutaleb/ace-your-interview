# Plan G — Adaptive Learning Engine & Quiz Controls (P1-product)

**Wave 4 · Depends on Plan D (FSRS consolidation is the data foundation)**

## Goal

Mastery exists per question (EWMA) and per topic (diagnostic), but there is no uncertainty estimate, no per-objective tracking, and no next-item selection by objective or performance — generation is topic + LLM driven. Add uncertainty, persist learning objectives, select next items adaptively, and add quiz controls (exam vs practice mode, timing).

## Current state (verified)

- `learning_planner.py:548-686` — `run_diagnostic` computes topic/track/interview-dimension competencies; `confidence_gap` is declared-vs-observed calibration, **not** measurement uncertainty; `evidence_count` is a raw attempt count.
- `learning_store.py:71-83` — `learning_attempts` has no `learning_objective` column; `learning_objective` is generated per question (`QuestionAnswerV2.learning_objective`, `models.py:138`) and discarded after the response.
- `learning_store.py:471-495` — the only mastery-aware ordering is the review queue (`ORDER BY mastery_score ASC, due_at ASC`).
- `models.py:469-471` — `LearningMode` has only `study|quiz`; `QuizMode.tsx:448` hardcodes `mode: "quiz"`; there is no timer, no exam mode, and answers are revealed immediately per question.
- `learning_planner.py:842-915` — `build_study_plan` takes only `user_id/days/daily_items`.

## Changes (ordered)

1. **Persist objectives** — add `learning_objective TEXT NOT NULL DEFAULT ''` to `learning_attempts` DDL (`learning_store.py:71-83`); add optional `learning_objective` to `LearningAttemptRequest` (`models.py:474-481`); `QuizMode.tsx`/`TopicStudy.tsx` send it with each attempt.
2. **Uncertainty** — extend `learner_competency_scores` (`learning_planner.py:124-140`) with `uncertainty REAL`; compute in `run_diagnostic` as the standard error of the accuracy estimate (`sqrt(p(1-p)/n)` with a Beta prior smoothing for small `n`), exposed on `CompetencyScore` (`models.py:607-618`).
3. **Objective-based next-item selection** — new `LearningStore.select_next_objectives(user_id, topic_ids, limit)`: rank objectives by (low mastery, high uncertainty, due pressure, low accuracy) from `learning_attempts` aggregates; expose as `GET /api/learning/next-objectives`. Wire into quiz generation: `GenerateQuizRequest` gains optional `target_objectives: list[str]`, and `routers/questions.py` injects them into the generation prompt as a targeting block (reuse the `prior_progress` seam at `questions.py:211-291`).
4. **Quiz controls** — `GenerateQuizRequest` (`models.py:429-437`) gains `mode: "practice"|"exam"` (default `practice`) and optional `time_limit_seconds`; `LearningMode` gains `"exam"`. Exam mode: `QuizMode.tsx` withholds correctness/explanation until final submission (reuse the existing `showReview` machinery at `:1083-1089`), renders a timer with an `aria-live` region, and submits all answers at once.
5. **Frontend** — `QuizMode.tsx` setup controls (mode toggle, question count already exists at `:618-625`, timer); `StudyPlan.tsx` diagnostic panel (`:674-723`) shows an uncertainty band per competency; `Dashboard.tsx` keeps bootstrapping server mastery (`:113-115`).
6. **Tests** — uncertainty computation (small-n smoothing, zero-evidence case), `select_next_objectives` ordering, exam-mode request validation, objective persistence round-trip, prompt targeting block rendering.

## Failure modes

- Objective strings are free-text LLM output — normalize (lowercase, trim, dedupe) before grouping; do not promise exact matching.
- Exam mode is a client-side practice feature: the API response still contains `correct_answer`, so it must not be used for high-stakes proctored assessment — document this limitation in the plan/PR. (A server-side exam model that withholds answers until submission is a larger change; out of scope.)
- `select_next_objectives` must degrade gracefully to topic-level selection when no attempts exist (cold start).

## Validation

- Backend suite + ruff; frontend build + tests.
- Manual: complete a quiz with wrong answers → diagnostic shows the topic with high uncertainty → next quiz targets that objective; exam mode hides answers until submit.

## Out of scope

Bayesian mastery models (BKT/DKT), hidden-test-case execution for coding questions, server-side exam answer withholding, per-section granularity.
