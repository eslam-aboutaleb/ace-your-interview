# 05 - AI Services and gRPC

## Goal
Understand the AI boundary contract, provider routing, and where to change analysis behavior safely.

## gRPC Contract as the System Boundary
Canonical contract:
- `services/proto/analysis.proto`

This file defines:
- RPCs shared by both AI backends (`AnalyzeMarket`, `QuickAnalysis`, `ScanMarkets`, etc.)
- request/response message shapes
- `LLMProvider` enum and per-request `LLMConfig`

If backend and service behavior disagree, check proto first.

## Core RPC Families
- Market analysis: full + quick + stream
- Risk/trade planning: `AssessRisk`, `GenerateTradePlan`
- Trader/copy/inverse evaluation: `AnalyzeTrader`, `EvaluateCopyTrade`, `EvaluateInversePosition`
- Discovery/sentiment/RAG index: `AnalyzeSentiment`, `DiscoverBestTrade`, `IndexMarketsRAG`
- Health checks

## Backend `AnalysisClient` Role (Adapter/Proxy)
File:
- `backend/app/grpc_clients/analysis_client.py`

Responsibilities:
- Choose target service address from selected backend (`llm_chain` or `cli_agent`)
- Reuse pooled gRPC channels/stubs
- Convert internal request dicts into proto messages
- Apply timeout and error normalization

Design intent: route handlers never speak raw gRPC details directly.

## Provider Routing Decision Flow
Backend-side selection occurs via:
- `backend/app/services/llm_gateway.py`
- user preferences in `UserSettings`

Priority order:
1. Request-level provider override (if present)
2. User preferred provider/model
3. Legacy ai_backend fallback
4. System default backend/provider

## LLM Chain Service (`services/llm-chain`)
Key files:
- `src/server.py`: gRPC servicer implementation
- `src/analysis_chain.py`: prompt and analysis logic
- `src/llm_factory.py`: provider factory
- `src/market_rag.py`: ChromaDB-backed semantic retrieval
- `src/mcp_client.py`: research MCP client

Capabilities:
- Multi-provider LLM support (OpenAI/Anthropic/Google/Groq/Ollama)
- Optional premium research integrations
- RAG indexing/search for market discovery
- Binance context enrichment path

## CLI Agent Service (`services/cli-agent`)
Key files:
- `src/server.py`: gRPC servicer
- `src/cli_agent.py`: analysis orchestration over GitHub Models/OpenAI-compatible API
- `src/mcp_client.py`: research context fetch

Capabilities:
- Alternate AI backend for analysis requests
- Uses GitHub token path if configured; OpenAI fallback path otherwise

## Research MCP Server and Data Enrichment
Server:
- `services/mcps/research/server.py`

Tools include:
- web search, x post search
- optional Tavily/News search
- Binance smart money/social/inflow/token context tools

Integration pattern:
- AI services call MCP client
- client spawns stdio server process and requests tool by name
- returned context is summarized/injected into prompts

## Fallback/Resilience Behaviors
- Backend can switch between two AI backends.
- Provider routing supports defaults when specific provider keys are absent.
- Research calls degrade gracefully (empty context rather than hard-fail) in many paths.
- Channel pooling reduces gRPC reconnect overhead.

## Change-Safety Rules
- If request/response shape changes: update `analysis.proto` first, then regenerate stubs and update client/service code.
- If adding a provider: update enum mapping/gateway/factory consistently.
- If prompt or research logic changes: update service chain code, then verify representative endpoints.

## Fast Debug Checklist
1. Check backend `/api/analysis/health` and service health RPCs.
2. Check backend gRPC host/port settings.
3. Check provider API keys in environment.
4. Verify `analysis.proto` and generated stubs are in sync.
5. Check MCP server path resolution if research context is unexpectedly empty.
