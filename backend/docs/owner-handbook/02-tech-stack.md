# 02 - Tech Stack

## Goal
Know what technologies are used, why they are used here, where they live, and what to learn first.

## Frontend Stack

### React 18 + TypeScript
- Why here: component UI, strict typing across services/components/state.
- Where: `frontend/src/`
- Learn first:
  1. `frontend/src/App.tsx`
  2. `frontend/src/components/AppLayout.tsx`
  3. Domain components (`Dashboard`, `Markets`, `CopyTrading`, `MarketMaking`, etc.)

### Vite
- Why here: fast local dev and bundling.
- Where: `frontend/vite.config.ts`, `frontend/package.json`
- Learn first: `npm/pnpm dev`, env handling via `VITE_API_URL`.

### Tailwind CSS
- Why here: utility-first styling, fast UI iteration.
- Where: `frontend/tailwind.config.js`, `frontend/src/index.css`
- Learn first: shared utility classes and app-wide style tokens.

### Zustand (`authStore`)
- Why here: minimal state management for auth/session state.
- Where: `frontend/src/store/authStore.ts`
- Learn first: session lifecycle (`setSession`, `clearSession`, `sessionReady`).

### Axios (`apiClient`)
- Why here: unified API calls + cookie credentials + 401 refresh interceptor.
- Where: `frontend/src/services/apiClient.ts`
- Learn first: refresh flow and retry behavior.

## Backend Stack

### FastAPI
- Why here: typed API framework with DI and async support.
- Where: `backend/app/main.py`, `backend/app/api/routes/*.py`
- Learn first: router registration and dependencies.

### SQLAlchemy 2
- Why here: ORM + database abstractions.
- Where: `backend/app/models/`, `backend/app/utils/database.py`
- Learn first: model definitions and session handling.

### Alembic
- Why here: controlled schema migrations.
- Where: `backend/migrations/`, `backend/alembic.ini`
- Learn first: revision chain in `backend/migrations/versions/`.

### Pydantic Settings
- Why here: centralized env-driven configuration with validation guardrails.
- Where: `backend/app/config.py`
- Learn first: security-sensitive validators (`JWT_SECRET_KEY`, debug salt requirements).

### SlowAPI (rate limiting)
- Why here: request rate-limit protection.
- Where: limiter setup in `backend/app/main.py`
- Learn first: storage behavior with Redis vs memory fallback.

### Redis + In-memory fallback caches
- Why here: challenge cache, credential store backend, rate-limit storage.
- Where: `backend/app/utils/cache.py`
- Learn first: when Redis is required vs optional.

### JWT + Web3/Ethereum signature auth
- Why here: wallet-native authentication model.
- Where: `backend/app/security/auth.py`, `backend/app/api/routes/auth.py`
- Learn first: challenge/verify + cookie token lifecycle.

## AI Service Stack

### gRPC
- Why here: clear contract between backend API and AI workers.
- Where: `services/proto/analysis.proto`, backend gRPC client files.
- Learn first: RPC list + `LLMConfig` provider override schema.

### LangChain (`llm-chain` service)
- Why here: structured prompt/chain logic and provider abstraction.
- Where: `services/llm-chain/src/analysis_chain.py`
- Learn first: analysis methods and research enrichment flow.

### Provider abstraction/factory
- Why here: switch or override LLM provider without route rewrite.
- Where:
  - backend gateway: `backend/app/services/llm_gateway.py`
  - llm-chain factory: `services/llm-chain/src/llm_factory.py`
- Learn first: provider → backend mapping and request-level overrides.

### MCP research integration
- Why here: attach web/news/social/binance context to analysis.
- Where:
  - research server: `services/mcps/research/server.py`
  - clients: `services/llm-chain/src/mcp_client.py`, `services/cli-agent/src/mcp_client.py`
- Learn first: tool dispatch contract and fallback behavior.

## Infrastructure / Runtime

### Docker Compose
- Why here: multi-service local environment.
- Where: `docker-compose.yml`
- Learn first: service graph and env vars required to boot backend safely.

### Postgres 15
- Why here: system of record for users/settings/trade state/audit/history.
- Where: compose service + SQLAlchemy models/migrations.

### Redis 7
- Why here: cache/rate-limiting/credential store backend.
- Where: compose service + `backend/app/utils/cache.py`.

## What To Learn First (Priority Order)
1. FastAPI route-to-service flow.
2. Auth + credential security model.
3. SQLAlchemy models + Alembic migration lifecycle.
4. gRPC analysis contract and backend `AnalysisClient` adapter.
5. Frontend service-to-route mapping and protected routing.
6. Background workers and failure/recovery behavior.
