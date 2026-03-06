# 04 - Frontend Architecture

## Goal
Understand how UI routes, state, and service calls are organized so you can trace and implement features safely.

## App Composition
Main entry points:
- `frontend/src/main.tsx`: React root mount
- `frontend/src/App.tsx`: router tree + protected route logic
- `frontend/src/components/AppLayout.tsx`: shared shell/navigation/logout/tutorial wrapper

### Route Structure
- Public route: `/login`
- Protected routes under `AppLayout`:
  - `/` dashboard
  - `/markets`, `/opportunities`, `/leaderboard`, `/copy-trading`, `/market-making`, `/backtesting`, `/trades`, `/settings`, `/analysis`, `/guide`
  - `/debug` admin-only guard

## Session and Auth State Pattern
State store:
- `frontend/src/store/authStore.ts`

Bootstrap flow in `App.tsx`:
1. App starts with `sessionReady = false`.
2. Calls `authService.getCurrentUser()`.
3. On success: `setSession(...)`.
4. On failure: `clearSession()`.
5. Marks `sessionReady = true`.

Protected routes render only when session bootstrap is complete.

## API Service Layer Design
- Base client: `frontend/src/services/apiClient.ts`
- Domain services: `frontend/src/services/*.ts`

Pattern:
- `apiClient` sets `withCredentials: true` for cookie auth.
- Response interceptor catches `401` and attempts `POST /api/auth/refresh`, then retries original request.
- If refresh fails, auth store is cleared.

This centralizes auth retry logic so feature services stay thin.

## Domain Service Modules
Examples:
- `authService.ts`: login challenge/verify/logout/me
- `portfolioService.ts`: balance/positions/summary/price refresh
- `tradesService.ts`: history, leaderboard, follow/copy, risk controls, execution
- `marketsService.ts`: search/browse and trader-analysis stream
- `analysisService.ts`: market/trader/opportunity AI analysis requests
- `settingsService.ts`: user and LLM preference updates
- `marketMakerService.ts`, `backtestingService.ts`, `inverseBotService.ts`, `stopLossService.ts`

## SSE Consumption Pattern
SSE appears in analysis flows (not pure polling).

Example pattern (`marketsService.streamTraderAnalysis`):
1. `fetch(..., { signal, credentials: 'include' })`
2. Read `ReadableStream` chunks.
3. Parse `data: {...}` messages.
4. Handle message types:
  - `trader_stats`
  - `chunk`
  - `done`
  - `error`
5. Expose `AbortController` for caller cancellation.

## UI-to-Backend Navigation Map
High-level mapping:
- Login page/components -> `/api/auth/*`
- Dashboard -> `/api/portfolio/*`, `/api/trades/following-feed`, stop-loss/inverse services
- Markets/Opportunities/Analysis -> `/api/markets/*`, `/api/analysis/*`
- Copy Trading/Leaderboard/Trades -> `/api/trades/*`
- Settings -> `/api/settings/*`
- Market Making -> `/api/market-maker/*`
- Backtesting -> `/api/backtesting/*`
- Debug (admin) -> `/api/debug/*`

For exact route mapping, see [09-endpoint-and-service-navigation.md](./09-endpoint-and-service-navigation.md).

## Component Organization Pattern
- `pages/`: route-level wrappers (`LoginPage`, `SettingsPage`, etc.)
- `components/`: feature-heavy UI modules (Dashboard, Markets, CopyTrading, etc.)
- `services/`: backend API adapters
- `types/`: shared TS interfaces
- `utils/`: helper logic (errors, grouping, workspace shortcuts)
- `hooks/`: custom hooks

## Practical Editing Rules
- New backend call: add to relevant `services/*.ts` first.
- New route page: wire in `App.tsx` route tree and navigation.
- Auth-sensitive behavior: test with expired session to verify refresh path.
- Streaming UI: ensure cancel handling and partial chunk resilience.

## Common Mistakes To Avoid
- Duplicating HTTP logic outside service layer.
- Storing auth tokens manually when cookie-based auth is already used.
- Triggering repeated polling without visibility/in-flight guards.
- Mixing domain concerns across components instead of service boundaries.
