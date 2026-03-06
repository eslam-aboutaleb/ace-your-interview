# 08 - Design Patterns in This Codebase

## Goal
Recognize recurring patterns so you can extend existing architecture instead of fighting it.

## 1) Layered Architecture
Pattern:
- routes handle transport/validation
- services handle business logic
- models/persistence handle storage concerns

Examples:
- Route modules in `backend/app/api/routes/`
- Services in `backend/app/services/`
- Models in `backend/app/models/`

Benefits:
- better testability and separation of concerns
- easier onboarding/navigation

Pitfalls:
- putting heavy business logic in routes
- bypassing service layer for cross-cutting behavior

## 2) Factory + Gateway Pattern (LLM Selection)
Pattern:
- choose provider/backend at runtime based on request/user/defaults

Examples:
- backend gateway: `backend/app/services/llm_gateway.py`
- llm factory: `services/llm-chain/src/llm_factory.py`

Benefits:
- provider flexibility without route rewrites
- consistent override precedence

Pitfalls:
- adding provider enum in one place but not all mappings
- inconsistent model defaults across services

## 3) Adapter/Proxy Pattern (gRPC Client)
Pattern:
- backend HTTP routes call a local adapter instead of raw service transport details

Example:
- `backend/app/grpc_clients/analysis_client.py`

Benefits:
- decouples API layer from gRPC internals
- centralizes timeout/error/channel pooling behavior

Pitfalls:
- bypassing adapter in ad hoc code
- forgetting proto/stub sync on contract changes

## 4) Singleton / Lazy Initialization
Pattern:
- cache expensive or global configuration/instances lazily

Examples:
- `get_settings()` with `@lru_cache`
- prompt manager singleton-like classes in AI services
- global/persistent cache instances in `utils/cache.py`
- shared gRPC channels with lazy lock init

Benefits:
- avoids repeated expensive setup
- centralizes shared process state

Pitfalls:
- hidden state in tests if cache not reset
- startup-order or event-loop coupling bugs

## 5) Fallback / Resilience Pattern
Pattern:
- preferred dependency with controlled fallback path

Examples:
- Redis -> in-memory cache fallback (`utils/cache.py`)
- WebSocket monitor -> HTTP polling fallback (`trade_monitor.py`)
- provider fallback in AI service/client settings
- data provider fallback (premium search -> free search)

Benefits:
- graceful degradation
- higher uptime under partial dependency failures

Pitfalls:
- silent quality degradation if fallback signals are not monitored
- inconsistent behavior across environments

## 6) Background Monitor Loop Pattern
Pattern:
- create long-running async loop tasks at startup and cancel cleanly at shutdown

Examples:
- `trade_monitor.py`
- `stop_loss_monitor.py`
- `inverse_bot_monitor.py`
- `arbitrage_service.py`
- `position_lifecycle_service.py`

Benefits:
- continuous automation outside request/response path

Pitfalls:
- race conditions and duplicated tasks
- missing cancellation/cleanup
- hidden side effects without observability

## 7) Security-by-Validation Pattern
Pattern:
- strong config and request validation to fail fast

Examples:
- settings validators in `backend/app/config.py`
- auth origin/bearer parsing and challenge checks in `auth.py`
- DB check constraints from migration hardening

Benefits:
- misconfiguration caught early
- reduced runtime ambiguity

Pitfalls:
- changing validation without documenting env impact
- relying only on UI checks for security-critical limits

## How To Apply These Patterns in New Code
1. Reuse route -> service -> model layering.
2. Extend provider mapping centrally when adding AI providers.
3. Use existing adapters rather than adding direct transport calls.
4. Keep fallback behavior explicit and logged.
5. For long-running tasks, implement start/stop lifecycle hooks and cancellation.
