# 01 - System Map

## Goal
Understand the full runtime shape of the platform before touching implementation details.

## High-Level Components
- `frontend/`: React + TypeScript UI (Vite dev server)
- `backend/`: FastAPI API server, business logic, background workers, DB access
- `services/llm-chain/`: gRPC analysis backend with LangChain and multi-provider LLM support
- `services/cli-agent/`: gRPC analysis backend using GitHub Models/OpenAI-compatible flow
- `services/mcps/research/`: MCP-style research tool server (web/x/news/binance context)
- `postgres` (Docker): primary relational database
- `redis` (Docker): rate-limit/cache/credential backing store (with controlled fallbacks)

## End-to-End Request/Data Path
1. Browser sends HTTP request to FastAPI backend.
2. Backend validates auth/session and request schema.
3. Backend route delegates to service layer.
4. If AI is needed, backend calls `AnalysisClient` (gRPC adapter).
5. `AnalysisClient` routes to `llm-chain` or `cli-agent` service.
6. AI service may call research MCP tools and external providers/APIs.
7. Backend combines AI/business results and returns JSON/SSE response.
8. Frontend updates local state/UI.

## Runtime Architecture Diagram
```mermaid
graph TD
    A[Browser / Frontend React] -->|HTTP JSON + SSE| B[FastAPI Backend]
    B -->|SQLAlchemy| C[(PostgreSQL)]
    B -->|Cache + rate limit + credentials| D[(Redis)]
    B -->|gRPC| E[LLM Chain Service]
    B -->|gRPC| F[CLI Agent Service]
    E -->|stdio JSON protocol| G[Research MCP Server]
    F -->|stdio JSON protocol| G
    E -->|LLM APIs| H[OpenAI/Anthropic/Google/Groq/Ollama]
    F -->|GitHub Models or OpenAI| I[GitHub Models / OpenAI]
    B -->|Trading + Market data| J[Polymarket APIs]
    G -->|Search + data enrichment| K[DDG/X/News/Tavily/Binance Skills]
```

## Backend Lifespan Workers
The backend starts long-running tasks in `backend/app/main.py` lifecycle startup.

Started services include:
- Trade monitor
- Leaderboard background refresh
- Stop-loss monitor
- Inverse-bot monitor
- Market-maker workers
- Position lifecycle manager
- Trade aggregation service
- Arbitrage monitor

## Lifecycle Diagram
```mermaid
flowchart TD
    S[FastAPI Startup] --> V1[Validate security config]
    V1 --> V2[Validate DB connectivity]
    V2 --> V3[Sync admin wallets]
    V3 --> T1[Start trade monitor]
    T1 --> T2[Start leaderboard refresh]
    T2 --> T3[Start stop-loss monitor]
    T3 --> T4[Start inverse-bot monitor]
    T4 --> T5[Start market-maker loops]
    T5 --> T6[Start position lifecycle manager]
    T6 --> T7[Start trade aggregation]
    T7 --> T8[Start arbitrage monitor]
    T8 --> R[Runtime]
    R --> X[Shutdown]
    X --> C1[Stop monitors/tasks]
    C1 --> C2[Close shared gRPC channels]
    C2 --> D[Process exits]
```

## External Integrations (What Can Fail)
- Polymarket CLOB/Gamma/Data APIs
- LLM provider APIs
- GitHub Models API
- Search/news providers
- Redis/Postgres network connectivity

If behavior looks inconsistent, first isolate whether the issue is:
- UI state issue
- API route/service issue
- external API dependency issue
- background worker side-effect issue

## Ownership Notes
- Treat this as a distributed system, not a single app.
- Any feature touching trade execution or credentials spans backend + external APIs + security modules.
- Any feature touching AI analysis spans backend route + gRPC contract + selected AI service implementation.
