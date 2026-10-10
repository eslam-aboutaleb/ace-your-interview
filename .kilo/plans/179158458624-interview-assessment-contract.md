# Plan A — Persist the Interview Assessment Contract (P0)

**Wave 1 · Parallel with Plan B · No dependencies**

## Goal

The interview question generator returns `competency_focus` and `expected_signals`, but the answer-submission route retrieves only the question text and passes it to the evaluator. The assessment contract is discarded before evaluation. Persist the full question contract per turn and evaluate against it.

## Current state (verified)

- `interview_generator.py:998-1002` — `generate_question` returns `{question, competency_focus, expected_signals}` only.
- `interview_store.py:380-405` — `set_current_question(user_id, session_id, question: str)` stores text only; `interview_sessions.current_question` is `TEXT NOT NULL` (DDL `:52-77`).
- `interview_store.py:78-95` — `interview_turns` has `question` text and `rubric_json` (the *evaluation result*), no columns for the question's contract.
- `interview_sessions.py:595` and `:714` — answer routes read `session["current_question"]` as text.
- `interview_generator.py:516-618` — `_evaluate_prompt` receives `question: str` only; the contract never reaches the evaluator.
- `voice_session.py:268-272` and `:317` — voice path has the same discard.
- `models.py:732` and `:735` — `degraded: bool = False` is declared twice on `InterviewTurn` (copy-paste bug; harmless at runtime, remove the duplicate).

## Changes (ordered)

1. **`interview_store.py`**
   - `interview_sessions`: add column `current_question_contract_json TEXT NOT NULL DEFAULT ''` via the existing `required_columns` + `PRAGMA table_info` pattern (`:119-130`).
   - `interview_turns`: add `competency_focus TEXT NOT NULL DEFAULT ''`, `expected_signals_json TEXT NOT NULL DEFAULT '[]'`, `question_contract_json TEXT NOT NULL DEFAULT ''` inside `_migrate_interview_turns` (`:134-178`, same idempotent pattern).
   - `set_current_question`: accept `contract: dict | None = None`; store `json.dumps(contract)`; keep appending the question text to `asked_questions_json`.
   - `record_turn`: accept `contract: dict | None = None`; persist the three new columns in the INSERT (`:461-481`); return them in the turn dict (`:518-530`).
   - `get_turns` (`:534-546`) and `_session_row_to_dict`: return the new fields.
2. **`interview_generator.py`**
   - `generate_question`: add `question_id` (deterministic hash of the question text — the frontend already derives one at `InterviewSessionPage.tsx:36-41`; make it server-owned) and `difficulty` to the return dict. Keep `competency_focus`/`expected_signals`.
   - `evaluate_answer`: accept `question_contract: dict | None = None`.
   - `_evaluate_prompt`: when a contract is present, render an "Assessment contract" section (`competency_focus`, `expected_signals`, `question_id`) and add the rule: "Score the candidate's evidence against the expected signals; do not reward content the question did not ask for."
   - Tolerate a missing/empty contract (legacy sessions): fall back to question-text-only evaluation.
3. **`interview_sessions.py`** — all four write sites (`:314-318`, `:387-391`, `:927-931`, `:989-993`) pass the full contract to `set_current_question`. Both answer routes (`:595`, `:714`) read the contract from the session and pass it to `evaluate_answer` (`:604-612`, `:741-751`) and `record_turn` (`:618-630`, `:764-776`).
4. **`voice_session.py`** — `start()` (`:268-272`) persists the contract; `_evaluate_answer()` (`:317`, `:319-326`, `:341`) reads it from the session and threads it through. (Known gap, out of scope: the voice path never passes `hint_level` — leave as-is.)
5. **`schemas/models.py`** — `InterviewTurn` gains `competency_focus: str = ""` and `expected_signals: list[str] = Field(default_factory=list)`; remove the duplicated `degraded` declaration (`:732`/`:735`). `InterviewQuestionResponse` (`:789-794`) already carries the fields — unchanged.
6. **Tests** — update `test_interview_store.py`, `test_interview_router_contract.py`, `test_interview_stream_router.py`, `test_voice_interview.py`, `test_coverage_interview_sessions_router.py` for the new columns/fields. Add: (a) contract round-trip through `set_current_question` → `record_turn` → `get_turns`; (b) evaluator receives the contract (assert the rendered `_evaluate_prompt` contains `competency_focus`); (c) legacy session with empty contract still evaluates.
7. **Evals** — extend `evals/interview.jsonl` input shape to carry an optional contract; the harness (`eval_agent_quality.py`) feeds `question` text only today.

## Failure modes

- Migration must be idempotent (existing DBs have no columns) — follow the `PRAGMA table_info` pattern exactly.
- Historical turns have no contract — evaluation must not regress for old sessions.
- `record_turn` runs inside `with self._lock: with self._conn:` (`:457-458`) — keep the new columns in the same transaction.

## Validation

- `cd backend && python -m pytest tests/ -x -q` (known pre-existing flaky: `test_forecast_buckets_by_day` at ~23:26 UTC — not a regression).
- `ruff check backend/app`.
- Frontend build unaffected (additive fields only).

## Out of scope

Per-type rubrics (coding/behavioral/system-design rubric schemas), rubric versioning, `source_refs` on interview questions.
