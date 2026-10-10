# Master Plan — AI Learning Platform: Correctness & Architecture Roadmap

**Orchestration plan for 8 sub-plans across dependency-gated waves. Read this first, then the wave's sub-plan(s).**

## Objective

Implement the review-driven roadmap: (1) make generation correctness explicit instead of silent, (2) consolidate the duplicated scheduler and make the LLM layer reusable, (3) centralize the frontend, (4) make learning adaptive. No rewrites — the current stack (FastAPI + LiteLLM + SQLite + React/Zustand) stays.

## Resolved decisions (2026-10-09)

1. **Scope**: all four phases (P0 correctness → P1 refactor → product features).
2. **Quiz fallback**: return partial results with an explicit `degraded` status; never silently substitute generic questions. No deterministic-template fallback in this round.
3. **Frontend state**: hand-rolled `useApi` hook + Zustand session stores; no new runtime dependencies.
4. **Interview storage**: extend existing `interview_sessions`/`interview_turns` tables (JSON columns + new text columns), not a new table.

## Wave structure

| Wave | Plans (parallel within a wave) | Gated on |
|---|---|---|
| 1 | **A** interview assessment contract · **B** evidence grounding & quiz validation | — |
| 2 | **C** quiz fallback removal & degraded contract · **D** FSRS consolidation | C←B, D←A |
| 3 | **E** LLM structured generation · **F** frontend centralization | E←C, F←A+C |
| 4 | **G** adaptive learning engine & quiz controls · **H** voice & JD preparation | G←D, H←A+D |

Dependency rationale: B defines the quiz item contract that C's fallback removal relies on; A defines the interview turn contract that D's attempt-recording and F's session store consume; C and A add the response fields F surfaces; D builds the FSRS data foundation G selects from.

## Worktree & session conventions (binding for every wave)

- One Agent Manager worktree session per sub-plan; branch named `plan/<plan-name>` (e.g. `plan/interview-assessment-contract`).
- **Before launching any wave**: commit or stash all pending work on `main`, then verify each worktree's merge-base equals the intended `main` HEAD. A stale leftover worktree can silently become the base — any misbased branch must be rebased onto `main` before dependent waves.
- **Shared files** (`backend/app/schemas/models.py`, `frontend/src/types/index.ts`, `frontend/src/services/api.ts`): append new content at the end of the file where possible; when editing existing classes, keep edits in **non-overlapping regions** (Wave 1: A edits interview classes `models.py:703-794`, B edits quiz classes `models.py:414-466`).
- Before committing on a branch: run the backend test suite, `ruff check backend/app`, and `cd frontend && npm run build`.
- Do **not** blindly commit the untracked `test_coverage_*.py` files — `grpcio` is not a declared dependency and committing them reintroduces the suite-wide collection break on a fresh CI install (commit c3e2e60 removed `test_coverage_grpc_client.py` for this reason).
- Do not rename `study_app*` identifiers — they are the LLM feature/API contract (error codes, `llm_source` enum, `grpc_client.py` user_id).
- Dependency pin to respect: `fsrs>=4.1,<4.2` (the real py-fsrs FSRS-4.5 package).

## Shared-file conflict map

| File | Plans | Wave ordering |
|---|---|---|
| `backend/app/services/question_generator.py` | B, C, E | sequential: B → C → E |
| `backend/app/services/interview_generator.py` | A, E | A → E |
| `backend/app/routers/interview_sessions.py` | A, D | A → D |
| `backend/app/services/learning_store.py` | D, G | D → G |
| `backend/app/services/learning_planner.py` | D, G, H | D → G/H |
| `backend/app/services/voice_session.py` | A, H | A → H |
| `backend/app/schemas/models.py` | A, B, C, D, E, G | per-wave, non-overlapping regions |
| `frontend/src/types/index.ts` | C, F | C → F |
| `frontend/src/services/api.ts` | C, F | C → F |
| `frontend/src/pages/QuizMode.tsx` | C, F, G | C → F → G |

## Validation gates (per wave, before merging to main)

1. `cd backend && python -m pytest tests/ -q` — known pre-existing flaky: `test_forecast_buckets_by_day` fails around 23:26 UTC when `now+1day+2h` crosses midnight; not a regression.
2. `ruff check backend/app`
3. `cd frontend && npm run build && npm test`
4. `cd backend && python scripts/eval_agent_quality.py` — all deterministic gates green (Plans B, C, E touch prompt/validator surfaces; rebaseline `prompt_snapshots.json` and goldens deliberately with `--blame-prompts`, never suppress the derived gates).
5. CI is consolidated in a single `backend.yml` — confirm it stays green before starting the next wave.

## Rollout & rollback

- Every plan is additive-first: new columns default, new response fields default, new flags default off where behavior is risky (none planned — all changes are contract-strictening).
- Rollback = revert the plan's branch; no data migrations are destructive (Plan D keeps `question_progress` read-only during soak; Plan A/B migrations are idempotent `ALTER TABLE ADD COLUMN`).
- Feature-flag escape hatches that already exist and must keep working: `enable_fsrs_v1`, `enable_v2_generation`, `enable_adaptive_learning`, `enable_mock_interview_v1`, `enable_voice_interview_v1`, `enable_voice_agent`.

## Definition of done (per plan)

- Sub-plan's ordered task list complete; its validation section green.
- New/changed API fields reflected in `frontend/src/types/index.ts` in the same branch (no wire drift).
- Tests added for each fixed defect (regression tests are mandatory for Plans A–D).
- Eval fixtures/goldens regenerated in the same branch when prompts or validators change (Plans B, C, E).

## Out of scope (entire roadmap)

Repository/DAO abstraction layer and Postgres migration (SQLite stays until deployment/concurrency requires it); API version prefixes (`/api/v1`); universal agent framework or plugin registries; LLM-as-judge as sole correctness authority; real-time collaborative interviews; enterprise assessment packs; multi-agent orchestration. Security review (auth, tenant isolation, user-supplied content) is a separate audit — the plans do not cover it.

## Open questions (none blocking)

- Whether to build deterministic per-topic verified templates as a future fallback replacement (deferred; measure the partial-rate first via Plan C's `degraded` telemetry).
- Whether `frontend/src/types/index.ts` should become OpenAPI-generated (deferred; hand-mirror is acceptable with the contract test noted in Plan F).
