# Plan: Platform Fixes, CI, and Frontend Test Foundation

## Goal
Ship the quick correctness fixes from the review, establish CI, and lay the frontend test foundation — plus the pluggable rate-limiter and store-abstraction groundwork for future scaling.

## Context (verified in code)
- `frontend/vite.config.ts:17-18`: PWA manifest still says `"Too lazy for this interview"` / `"Too Lazy"` after the rebrand to Ace Your Interview.
- `backend/app/routers/chat.py`: `HARD_INPUT_GUARD_CHARS` is defined twice (once as `200_000` with a comment saying "well above every per-field cap", then redefined as `20000`); the second silently wins, contradicting the comment.
- `backend/app/routers/voice.py:60`: `user_tier=settings.voice_default_tier` with `# TODO: per-user override from UserSettingsStore`.
- No `.github/` directory — 56 backend test files run nowhere automatically.
- Frontend has zero tests.
- `backend/app/main.py:59`: `InMemoryRateLimiter()` — no shared state across replicas.
- Stores use a single SQLite connection + single-thread executor (`learning_store.py:51-52`).

## Decisions (resolved)
- Rate limiter: pluggable backend interface, in-memory default, optional Redis (`STUDY_RATE_LIMIT_BACKEND=redis`, `REDIS_URL`); the 429 contract is unchanged.
- Store abstraction: define the async-store protocol and document the migration path; do NOT port stores to Postgres in this plan (deferred follow-up).
- Frontend tests: Vitest + React Testing Library + jsdom; highest-risk paths first.

## Tasks (ordered)
1. **PWA manifest** (`frontend/vite.config.ts:17-18`): name/short_name → `"Ace Your Interview"` / `"Ace Your Interview"`; verify the icons referenced in the manifest exist in `frontend/public/` (pwa-64x64, pwa-192x192, pwa-512x512, maskable-icon-512x512, apple-touch-icon-180x180).
2. **Duplicate constant** (`backend/app/routers/chat.py`): remove the second `HARD_INPUT_GUARD_CHARS = 20000`; keep `200_000` with the existing comment; add a unit test asserting the effective value and that `_fence` clips at per-field caps.
3. **Voice per-user tier**: add a `voice_tier` column to the user settings store (`user_settings_store.py`), expose it via `PUT /api/user-settings/preferences`, and return the real `user_tier` in `GET /api/voice/config`; the admin default remains the fallback.
4. **CI** (`.github/workflows/ci.yml`): on push/PR to main — backend: `uv sync`, `ruff check .`, `pytest -q`; frontend: `npm ci`, `npm run build` (tsc + vite). Docker build smoke test deferred.
5. **Frontend test foundation**: add `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom` dev deps; `vitest.config.ts` with the `@/` alias and mocks for heavy modules (mermaid, onnxruntime-web); tests: `authStore` hydration + redirect logic, `ProtectedRoute` unauthenticated redirect, `MarkdownRenderer` mermaid/code rendering, `ReviewQueue` due-item rendering, `api` error mapping. Target ~15 tests.
6. **Pluggable rate limiter** (`backend/app/services/rate_limit.py`): extract a `RateLimiterBackend` protocol (`check(scope, key, limit, window_seconds) -> Decision`); `InMemoryBackend` (existing logic) and `RedisBackend` (sliding window via sorted sets); wire in `main.py` via settings; parity test: the same request sequence yields the same allow/deny decisions in both backends (mocked redis).
7. **Store abstraction** (`backend/app/services/store_protocol.py`): define the async-store interface (`run_async`, table init, migration hooks) with docstring-level Postgres migration notes; no behavioral change.

## Failure modes
- Redis unavailable at startup → fail fast with a clear log; do not silently fall back to in-memory in production (`STUDY_RATE_LIMIT_STRICT=true`).
- Frontend test env differences (mermaid/onnxruntime imports) → mock heavy modules in `vitest.config.ts`.
- Voice tier migration → additive column, default null → fallback to the admin default.

## Rollout
1. Fixes 1–3 ship immediately (no flags needed; all backward compatible).
2. CI ships with the PR; required check on main.
3. Rate-limiter backend ships defaulting to in-memory; Redis opt-in.

## Validation
- `npm run build` green; `pytest -q` green locally and in CI.
- New unit tests pass; PWA manifest verified via `vite build` output (`dist/manifest.webmanifest` contains the new name).
- Rate-limiter parity test green.
