# 13 - Glossary

## Domain Terms

### CLOB
Central Limit Order Book used by Polymarket trading APIs for order/trade interactions.

### Condition ID (`condition_id`)
Identifier for a prediction market condition/outcome group. Frequently used as the market key across APIs.

### Token ID (`token_id` / `asset_id`)
Identifier for a specific YES/NO outcome token.

### YES/NO Price
Outcome token price in the 0-1 range, often interpreted as implied probability.

### Proxy Wallet (Polymarket)
Execution/funding abstraction used by Polymarket for some account flows; may differ from plain EOA behavior.

### EOA
Externally Owned Account (normal wallet address/private key account).

### Slippage
Difference between expected and actual execution price.

### Notional
Dollar value of a trade (`size * price` in simplified terms).

### PnL
Profit and Loss.

### Drawdown
Peak-to-trough portfolio decline metric.

## Repo-Specific Terms

### `AnalysisClient`
Backend adapter that wraps gRPC calls to AI services and hides transport details from routes.

### `LLMGateway`
Backend provider-routing component that chooses AI backend/provider based on request/user/default settings.

### `llm-chain`
gRPC AI service implementing LangChain-based analysis with multi-provider support.

### `cli-agent`
gRPC AI service using GitHub Models/OpenAI-compatible API path.

### Research MCP Server
Stdio tool server used by AI services for web/news/social/binance context gathering.

### Trade Monitor
Background service watching followed trader activity via WebSocket/polling and triggering downstream actions.

### Inverse Bot
Feature that evaluates whether existing positions should be reversed based on AI signals.

### Stop-Loss Monitor
Background loop that checks active stop-loss rules and triggers exits when thresholds are met.

### Market Maker
Automated strategy that places/refreshes limit orders around market prices (bands/AMM approaches).

### Position Lifecycle Manager
Background logic to manage ongoing position state transitions and maintenance tasks.

### Trade Aggregation Service
Background/process logic that combines rapid related trade events into more coherent execution units.

### Copy Trade Evaluation Row
Record shape describing source trade vs copied trade outcome and sizing metadata.

### Simulation Mode
Execution mode where strategy logic runs but real order placement is bypassed.

### Emergency Stop
Control that pauses trading activity quickly, with optional sell-all behavior.

### `challenge_cache`
Auth cache storing short-lived wallet login challenges.

### `key_store`
Credential cache namespace storing encrypted private-key-related payloads.

### `opportunity_cache`
Cache namespace storing AI opportunity scan results.

### `sessionReady`
Frontend auth-store flag indicating server-validated session bootstrap has completed.

## Operational Terms

### Fail-Closed
On validation/decryption/config failure, deny action rather than proceed insecurely.

### Graceful Fallback
Use secondary implementation when preferred dependency fails (for example WS -> HTTP polling).

### Ownership Path
Structured progression from onboarding to independently safe feature delivery and operations.
