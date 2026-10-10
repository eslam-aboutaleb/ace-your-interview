# Plan C — Remove the Generic Quiz Fallback & Standardize Degraded Responses (P0)

**Wave 2 · Depends on Plan B (quiz contract fields must exist first; both edit the quiz region of `schemas/models.py`)**

## Goal

`_build_fallback_quiz_items` silently substitutes generic "engineering judgment" questions — identical choices, correct answer always "A" — for any topic when generation fails. Remove it and return an explicit partial/degraded contract instead. Decision (resolved): **partial results + explicit degraded status; no substitution.**

## Current state (verified)

- `question_generator.py:3083-3190` — `_build_fallback_quiz_items`: 4 fixed MCQ stems (`:3127-3132`), 4 fixed choices (`:3134-3139`), `correct = "A"` (`:3140`), 3 fixed true/false stems (`:3143-3148`), fixed explanation/reasoning (`:3168-3175`).
- Injected silently at `:4509-4533` (stream) and `:4649-4666` (batch, with a comment justifying the silence).
- The **question** fallback `_build_fallback_question_items` (`:2747-2924`) is genuinely topic-aware (content fragments, difficulty-tiered templates, grounding-validated) — **keep it**, but mark when it was used.
- No `status`/`degraded`/`warnings`/`requested_count`/`returned_count` field exists on any question or quiz response model (`models.py:124-151`, `:440-466`).
- Latent frontend bug: the backend emits a `warning` event for partial topic resolution (`routers/questions.py:829-837`) but `GenerateQuizStreamEvent` (`types/index.ts:372-377`) has no `warning` member, so `dispatchEvent` (`api.ts:837-846`) throws a non-fallback-eligible `QuizStreamError` and aborts the stream.

## Changes (ordered)

1. **`question_generator.py`**
   - Delete `_build_fallback_quiz_items` and both injection sites (`:4509-4533`, `:4649-4666`).
   - `generate_quiz_v2_stream`: `done` event gains `degraded: bool` (true when `generated_count < target_count`), `requested_count`, `returned_count` (`:4535-4544`).
   - `generate_quiz_v2`: `GenerateQuizV2Response` gains the same three fields (`:4690-4697`).
   - `generate_v2_stream` / `generate_v2`: keep the topic-aware question fallback, but set `degraded: true` on `GenerateQuestionsV2Response` and the stream `done` event when fallback items were injected (`:3840-3883`, `:4183-4224`, `:3885-3894`).
2. **`schemas/models.py`** — `GenerateQuizV2Response` (`:460-466`) and `GenerateQuestionsV2Response` (`:144-151`) gain `degraded: bool = False`, `requested_count: int = 0`, `returned_count: int = 0`.
3. **`routers/questions.py`** — pass the new fields through all six endpoints (`:466-872`); keep the `X-Topic-Resolution-Failures` header and the stream `warning` event as-is.
4. **Frontend**
   - `types/index.ts`: add `warning` to `GenerateQuizStreamEvent` (`:372-377`) with `{topics: string[], topics_used: string[]}`; add `degraded`/`requested_count`/`returned_count` to the `done` payload types and to `GenerateQuizV2Response`/`GenerateQuestionsV2Response`.
   - `api.ts`: handle `warning` in both `dispatchEvent` implementations (`:753-777`, `:819-847`) as a non-throwing callback (`onWarning`).
   - `QuizMode.tsx`: render a degraded banner when `done.degraded` (precedent: `InterviewSessionPage.tsx:633-640`); rework `topUpMissingQuestions` (`:317-355`) so it no longer silently backfills through v2-then-v1 — surface the partial state and offer an explicit retry instead.
5. **Tests** — update the tests that assert fallback-filled counts: `test_coverage_question_generator_deep.py:1612-1645` (`test_a_terminal_quiz_v2_failure_still_serves_a_batch`, `test_the_legacy_quiz_drops_junk_and_duplicates`, `test_the_quiz_fallback_never_exceeds_the_requested_count`), `:1358-1374`, `:1335-1354`; the streaming tests at `:136-159`, `:184-225`; `test_quiz_stream_router.py:151-165`. New expectations: partial counts + `degraded: true`, never generic questions. Add: degraded flag round-trip on all four endpoints; warning-event handling in `api.ts` (frontend test).
6. **Evals** — the deterministic tier does not import either fallback builder (verified: `eval_agent_quality.py:63-72`), so gates are unaffected. The live tier replays real generation and diffs against `evals/golden/` — expect drift on any provider-failure replay; regenerate goldens deliberately.

## Failure modes

- Users receive shorter quizzes on generation failure — accepted trade-off per the resolved decision; the degraded banner + retry must make it obvious.
- Empty result path already exists (`QuizMode.tsx:399-402`) — verify it still triggers.
- v1 endpoints (`generate`, `generate_quiz`) delegate to v2 implementations, so all four paths change together — do not remove only the stream injection.

## Validation

- Backend suite + ruff; frontend build + `npm test`.
- Manual: force a terminal LLM error (budget-exceeded test double) and confirm the response carries `degraded: true` and no generic questions.

## Out of scope

Deterministic per-topic verified templates as a replacement fallback (revisit only if partial-rate data justifies it).
