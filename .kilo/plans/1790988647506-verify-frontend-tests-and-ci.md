# Plan: Verify and Land Platform Fixes + CI (Remaining Tasks)

## Context
All 7 tasks of `.kilo/plans/1790987112911-platform-fixes-ci-testing.md` are implemented and committed in `d7a7a85` on branch `plan/platform-fixes-ci` (worktree `plan-platform-fixes-ci-f0c451fe6d58b74f`, clean tree):

- PWA manifest rename to "Ace Your Interview" (`frontend/vite.config.ts`)
- Duplicate `HARD_INPUT_GUARD_CHARS` removed (`backend/app/routers/chat.py`, kept `200_000`)
- Per-user voice tier (`voice_tier` column, `PUT /api/user-settings/preferences`, `GET /api/voice/config`)
- `.github/workflows/ci.yml` (backend: `uv sync` + `ruff check .` + `pytest -q`; frontend: `npm ci` + `npm run build` + `npm test`)
- Vitest foundation: 7 test files (`authStore`, `settingsStore`, `ProtectedRoute`, `MarkdownRenderer`, `ReviewQueue`, `api`, `textNormalization`)
- Pluggable rate limiter (`RateLimiterBackend` protocol, `InMemoryBackend` default, `RedisBackend` opt-in)
- `store_protocol.py` async-store interface (Postgres migration notes only)

Backend suite was green (479 passed) at commit time. Frontend had 3 failing test files; fixes were committed but **never verified by a test run**:

1. `frontend/src/test/setup.ts` — shims `globalThis.localStorage` with jsdom's when Node ≥22's experimental `localStorage` global is undefined (zustand persist's default storage reads the bare `localStorage` global; `settingsStore.test.ts` failed with `Cannot read properties of undefined (reading 'setItem')`).
2. `frontend/src/services/api.test.ts` — `vi.hoisted` for `responseUseMock` (vi.mock factories are hoisted above `const` initializers → TDZ `ReferenceError`).
3. `frontend/src/components/ProtectedRoute.test.tsx` — imports `ProtectedRoute` from `@/App` (was `./App`, unresolvable from `src/components/`).

## Remaining tasks (ordered)
1. **Run frontend tests**: `npm test` in `frontend/` (vitest run). Expect all 7 files green.
   - If `ProtectedRoute.test.tsx` fails at collection: `@/App` transitively imports `mermaid` (via `pages/InterviewSessionPage|InterviewReportPage|TopicStudy|QuizMode → MarkdownRenderer → MermaidDiagram`) and `react-syntax-highlighter` (via `CodePanel`). Fix by adding `vi.mock("mermaid", ...)` and `vi.mock("react-syntax-highlighter", ...)` at the top of `ProtectedRoute.test.tsx` — same test-level mock pattern already used in `MarkdownRenderer.test.tsx`.
   - If `settingsStore.test.ts` still throws on `setItem`: the `Object.defineProperty` in `setup.ts` fails when Node's global is non-configurable — replace with `vi.stubGlobal("localStorage", window.localStorage)` in `setup.ts`.
   - `api.test.ts` mock shape already verified against `api.ts:77` (`axios.create`) and `api.ts:458` (`interceptors.response.use(onFulfilled, onRejected)`; test reads `mock.calls[0][1]` as the rejection handler). No change expected.
2. **Run frontend build**: `npm run build` in `frontend/` (tsc + vite build). Verify `dist/manifest.webmanifest` contains `"Ace Your Interview"` (original plan's validation criterion).
3. **Run backend gates**: `uv run ruff check .` (rules pinned to E4/E7/E9/F in `backend/pyproject.toml`) and `uv run pytest -q` in `backend/`. Confirm still green.
4. **Commit**: if residual fixes were needed, commit them on `plan/platform-fixes-ci` as a follow-up commit (do not amend `d7a7a85`).

## Out of scope
- Docker build smoke test (deferred in the original plan).
- Additional frontend test coverage beyond the existing 7 files.
- Pushing the branch / opening the PR (rollout step). Note: CI runs `npm test` on Node 20, where the `setup.ts` localStorage shim is a no-op (jsdom provides `localStorage` natively) — the fix is environment-safe.

## Validation
- `npm test` green: 7 files, 0 failures.
- `npm run build` green; `dist/manifest.webmanifest` shows the new name.
- `uv run ruff check .` clean; `uv run pytest -q` green.
- CI workflow mirrors these exact commands, so local green ⇒ CI green.
