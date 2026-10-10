# Plan F — Frontend Centralization: useApi, Session Stores, ErrorBoundary, a11y (P1)

**Wave 3 · Depends on Plan A (interview turn fields) and Plan C (quiz degraded fields) — both add response fields this plan surfaces**

## Goal

Server state is fetched ad hoc in ~15 page effects; interview and quiz session state lives in 15–25 local `useState` hooks; there is no `ErrorBoundary`, no shared `useApi` hook, no retry control (one in the whole app), no `aria-live` region, and no linter. Centralize without adding runtime dependencies. Decision (resolved): **hand-rolled `useApi` hook + Zustand session stores; no TanStack Query.**

## Current state (verified)

- `services/api.ts:103-108` — one axios instance, ~110 endpoint functions; `_streamNdjson` (`:292-481`) and typed stream errors with `fallbackEligible` (`:175-268`) are genuinely well-factored.
- `store/` — 4 Zustand stores (`authStore` 51, `settingsStore` 82, `progressStore` 192, `voiceStore` 177 lines); voice is the only centralized session domain.
- `InterviewSessionPage.tsx:49-64` — 15 `useState` hooks; `QuizMode.tsx:72-119` — ~25.
- The `fallbackEligible` recovery block is copy-pasted into 5 pages (`InterviewSessionPage.tsx:183-204, 263-286, 334-356`; `TopicStudy.tsx:979-996`; `QuizMode.tsx:388-393`).
- 51 `animate-spin` occurrences across 14 files; 96 `catch` blocks; one shared error extractor (`InterviewSetup.tsx:54`); one real retry button (`TopicStudy.tsx:1627-1632`).
- Zero `aria-live`/`role="status"`/`role="alert"` in `src/`; no eslint config or dependency; `types/index.ts` (1510 lines) is a hand-mirrored copy of `backend/app/schemas/models.py` with confirmed drift (frontend `SubmitInterviewAnswerRequest` omits `question_id`/`hint_level` that the page actually sends).

## Changes (ordered)

1. **`hooks/useApi.ts`** (new) — shared async hook: `{data, error, loading, retry}` with abort-on-unmount, shared error-message extraction (move `InterviewSetup.tsx:54` into `utils/apiError.ts`), and optional `deps` refetch. Migrate the simplest pages first (`ReviewQueue`, `StarStories`, `InterviewTrends`), then the rest.
2. **`store/interviewSessionStore.ts`** and **`store/quizStore.ts`** (new, Zustand, mirroring `voiceStore.ts`) — session object, turns, current question + contract, degraded flags, submission state. Migrate `InterviewSessionPage.tsx` and `QuizMode.tsx` state incrementally; keep purely local UI state (drafts, modals) local.
3. **`hooks/useStreamGeneration.ts`** (new) — extract the duplicated `fallbackEligible` recovery block into one hook wrapping `generateQuizV2Stream`/`generateQuestionsV2Stream`/`createInterviewSessionStream`/`submitInterviewAnswerStream`/`generateNextInterviewQuestionStream`.
4. **`components/ErrorBoundary.tsx`** (new) — class component with a fallback UI; wrap route elements in `App.tsx`.
5. **Response-field surfacing** — `types/index.ts`: add `competency_focus`/`expected_signals` to `InterviewTurn` (Plan A), `degraded`/`requested_count`/`returned_count` to quiz/question responses (Plan C), `warning` to `GenerateQuizStreamEvent`; fix the `SubmitInterviewAnswerRequest` drift by adding `question_id`/`hint_level`. Render the interview degraded banner from the top-level `degraded` flag too (`InterviewSessionPage.tsx` currently reads only `turn.degraded` at `:633`).
6. **Accessibility** — `aria-live="polite"` regions for generation progress, degraded banners, and any timer; `aria-label` on icon-only buttons (`VoicePanel.tsx:313, 347`); move focus to newly generated content; add `eslint`, `eslint-plugin-jsx-a11y`, `@axe-core/react` as devDependencies (requires `npm install` — implementation agent runs it) and a `lint` script.
7. **Retry controls** — real retry buttons where prose-only today (`QuizMode.tsx:400`, `DecksPage.tsx:89,197,794`, `DocumentsPage.tsx:91,175,197`, `ReviewQueue.tsx:80,140`), wired to `useApi.retry`.

## Failure modes

- Large-surface refactor: migrate page-by-page, keep `npm run build` green after each page; no visual/behavioral changes beyond the new banners/controls.
- Zustand `persist` keys: do not reuse existing localStorage keys (`ace-your-interview-*`); new stores start unpersisted except explicit prefs.
- `types/index.ts` is hand-mirrored — add a contract test (or OpenAPI codegen) only if the team wants it; at minimum, document the mirror in the file header (already noted at `:1`).

## Validation

- `cd frontend && npm run build && npm test`.
- Manual smoke: interview session (question → answer → degraded turn), quiz (partial generation → degraded banner → retry), voice panel, keyboard tab order through the interview flow.

## Out of scope

TanStack Query adoption, OpenAPI codegen, redesign of any page layout, per-page visual changes.
