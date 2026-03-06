# 03 - Backend Architecture

## Goal
Understand how backend code is organized and where to modify behavior safely.

## Layer Map
- API layer: `backend/app/api/routes/*.py`
- Service layer: `backend/app/services/*.py`
- Persistence layer: `backend/app/models/*.py`, `backend/app/utils/database.py`
- Security layer: `backend/app/security/*.py`
- Cross-cutting infra: `backend/app/config.py`, `backend/app/utils/cache.py`, middleware, gRPC clients

Flow:
`Route` -> `service function/class` -> `model + DB` -> optional external/gRPC call -> response schema.

## App Entry and Composition
- Entry: `backend/app/main.py`
- What it wires:
  - settings + validation
  - rate limiter middleware
  - CORS + security headers
  - request logging middleware
  - router registration
  - startup/shutdown lifecycle workers

## Dependency Injection Patterns
Most route handlers use `Depends(...)` for controlled access to shared concerns:
- DB session: `Depends(get_db)` from `backend/app/utils/database.py`
- current user/session: `Depends(get_current_user_from_token)` from auth routes

Typical route pattern:
1. Validate request with Pydantic model.
2. Resolve authenticated user.
3. Resolve DB session.
4. Delegate to service/model operations.
5. Return response model.

## Route Modules and Responsibilities
- `auth.py`: challenge/signature login, token refresh/logout/me, cookies, origin checks
- `analysis.py`: AI analysis endpoints (including stream endpoints)
- `trades.py`: execution, history, leaderboard, follow/copy, stop-loss/take-profit, emergency controls
- `markets.py`: browse/search categories and trader-analysis stream
- `portfolio.py`: balances/positions/summary and price refresh
- `settings.py`: user settings/profile and LLM preference/admin settings
- `market_maker.py`: market-maker config/start/stop/sync/metrics
- `backtesting.py`: backtest run CRUD + strategy metadata
- `inverse_bot.py`: inverse bot tracked positions and manual evaluation
- `binance_signals.py`: smart-money and ranking APIs
- `debug.py`: guarded debug endpoints (admin/debug mode)

## Service Layer Patterns
Service modules contain business logic and external integration details.

Examples:
- `polymarket_service.py`: API + chain interactions and portfolio/trade data fetch
- `copy_trade_service.py`: copy trade sizing/evaluation/execution helpers
- `trade_monitor.py`: watched-wallet polling/WS + side effects
- `market_maker_service.py`: bands/AMM strategies and sync loops
- `arbitrage_service.py`: periodic arbitrage scan and in-memory opportunity cache
- `stop_loss_monitor.py`, `inverse_bot_monitor.py`, `position_lifecycle_service.py`: background state machines
- `llm_gateway.py`: provider and backend routing decisions

## Concurrency Model
- FastAPI async routes handle request concurrency.
- Background loops run as asyncio tasks started in lifespan startup.
- Some blocking operations are offloaded to dedicated thread pools (e.g., Web3/CLOB calls in `PolymarketService`).
- gRPC client channels/stubs are pooled and shared to reduce connection churn.

## Startup and Shutdown Responsibilities
Startup (`lifespan` in `backend/app/main.py`):
1. Validate security-critical config.
2. Validate DB connectivity.
3. Sync admin wallet list to `is_admin` flags with audit logs.
4. Load followed wallets into watch list.
5. Start monitors and periodic services.

Shutdown:
1. Stop monitors/services in guarded sequence.
2. Close shared gRPC channels.
3. Cancel remaining background tasks.

## Database Access Pattern
- Engine/session are created in `backend/app/utils/database.py`.
- Session lifecycle in routes is handled by generator dependency `get_db`.
- Models are mapped in `backend/app/models/` and imported via package side effects for migrations.

## Practical Navigation Rules
- If request schema/validation needs changing: start in route module.
- If behavior/logic needs changing: start in relevant service module.
- If stored fields/state need changing: update model + migration + API schemas.
- If auth/session behavior needs changing: touch `auth.py` route + `security/*` helpers together.

## Common Mistakes To Avoid
- Putting heavy business logic directly in route handlers.
- Introducing blocking network calls in async paths without offloading.
- Modifying auth/token logic without re-validating cookie and refresh flows.
- Editing models without adding a corresponding migration and tests.
