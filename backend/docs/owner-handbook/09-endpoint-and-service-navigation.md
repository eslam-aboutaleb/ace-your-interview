# 09 - Endpoint and Service Navigation

## Goal
Quickly answer: which files do I edit for a given feature or bug?

## Backend Route Catalog (Grouped by Domain)

| Domain | Prefix | Key Endpoints | Primary Route File |
|---|---|---|---|
| Auth | `/api/auth` | `/login`, `/verify`, `/login-with-key`, `/refresh`, `/logout`, `/me` | `backend/app/api/routes/auth.py` |
| Analysis | `/api/analysis` | `/market`, `/market/stream`, `/quick`, `/scan`, `/risk`, `/trade-plan`, `/trader`, `/copy-trade-eval`, `/opportunities/stream` | `backend/app/api/routes/analysis.py` |
| Markets | `/api/markets` | `/categories`, `/search`, `/browse`, `/trader-analysis/stream` | `backend/app/api/routes/markets.py` |
| Portfolio | `/api/portfolio` | `/balance`, `/positions`, `/summary`, `/markets`, `/markets/prices`, `/positions/prices` | `backend/app/api/routes/portfolio.py` |
| Trades | `/api/trades` | `/history`, `/execute`, `/cash-out`, `/leaderboard`, follow/notification routes, stop-loss/take-profit, emergency/resume, arbitrage endpoints | `backend/app/api/routes/trades.py` |
| Settings | `/api/settings` | base get/put, `/copy-trading`, `/profile`, `/llm/*`, admin LLM endpoints | `backend/app/api/routes/settings.py` |
| Inverse Bot | `/api/inverse-bot` | `/positions`, `/positions/{id}/evaluate`, `/metrics` | `backend/app/api/routes/inverse_bot.py` |
| Market Maker | `/api/market-maker` | `/configs`, `/configs/{id}/start`, `/stop`, `/sync`, `/metrics` | `backend/app/api/routes/market_maker.py` |
| Backtesting | `/api/backtesting` | `/runs`, `/runs/{id}`, `/strategies` | `backend/app/api/routes/backtesting.py` |
| Binance Signals | `/api/binance-signals` | smart money/ranking/token endpoints + dashboard | `backend/app/api/routes/binance_signals.py` |
| Debug | `/api/debug` | `/logs`, `/stats`, `/health` | `backend/app/api/routes/debug.py` |

## Frontend Service -> Backend Endpoint Map

| Frontend Service | Main Methods | Backend Endpoints |
|---|---|---|
| `authService.ts` | challenge, verify, logout, me | `/api/auth/*` |
| `analysisService.ts` | analyze/scan/stream/trader evaluations | `/api/analysis/*` |
| `marketsService.ts` | category/search/browse + trader SSE stream | `/api/markets/*` |
| `portfolioService.ts` | balance/positions/summary/prices | `/api/portfolio/*` |
| `tradesService.ts` | history/execution/follow/feeds/risk/arbitrage | `/api/trades/*` |
| `settingsService.ts` | user settings/profile/llm preferences | `/api/settings/*` |
| `marketMakerService.ts` | config CRUD + control actions | `/api/market-maker/*` |
| `backtestingService.ts` | run CRUD + strategies | `/api/backtesting/*` |
| `inverseBotService.ts` | position tracking/evaluate/metrics | `/api/inverse-bot/*` |
| `binanceSignalsService.ts` | smart money/ranking/token data | `/api/binance-signals/*` |
| `debugService.ts` | logs/stats/health | `/api/debug/*` |

## Backend Endpoint -> Service/Model Map (Representative)

| Endpoint | Service Modules | Key Models |
|---|---|---|
| `/api/auth/verify` | `security/auth.py`, `security/refresh_tokens.py`, `security/credential_store.py` | `User`, `RefreshToken` |
| `/api/analysis/market` | `grpc_clients/analysis_client.py`, `services/llm_gateway.py` | `UserSettings` |
| `/api/markets/search` | `services/polymarket_service.py` | `Market` (indirect cache usage) |
| `/api/trades/execute` | `services/copy_trade_service.py`, `services/polymarket_service.py` | `UserTrade`, `TradeHistory`, risk/order models |
| `/api/trades/follow/{wallet}` | `services/trade_monitor.py` (watchlist side effects) | `FollowedTrader` |
| `/api/trades/stop-loss` | `services/stop_loss_monitor.py` | `StopLossOrder` |
| `/api/settings/llm/*` | `services/llm_gateway.py` | `UserSettings` |
| `/api/market-maker/configs/{id}/start` | `services/market_maker_service.py` | `MarketMakerConfig` |
| `/api/backtesting/runs` | `services/backtesting_service.py` | `BacktestRun` |

## Optional gRPC Catalog (AI Contract)

| RPC | Purpose |
|---|---|
| `AnalyzeMarket` | Full market analysis |
| `AnalyzeMarketStream` | Chunked market analysis stream |
| `QuickAnalysis` | Fast short analysis |
| `ScanMarkets` | Multi-market opportunity scan |
| `AssessRisk` | Risk evaluation |
| `GenerateTradePlan` | Suggested execution plan |
| `AnalyzeTrader` | Trader profile analysis |
| `EvaluateCopyTrade` | Copy-trade decision support |
| `EvaluateInversePosition` | Reverse/hold signal for open position |
| `AnalyzeSentiment` | Sentiment-focused analysis |
| `DiscoverBestTrade` | Autonomous best-trade selection |
| `IndexMarketsRAG` | Populate/refresh vector index |
| `HealthCheck` | Service health |

Proto source: `services/proto/analysis.proto`

## “If Changing X, Touch Y” Matrix

| Change Goal | Must Touch | Usually Also Touch |
|---|---|---|
| Add a backend API endpoint | route file + request/response models | service module, tests, frontend service |
| Change auth/session behavior | `api/routes/auth.py`, `security/auth.py` | frontend `apiClient.ts`, `authService.ts`, auth tests |
| Change trade execution logic | `api/routes/trades.py`, relevant service | risk-setting models, integration tests |
| Add user setting | `models/user_settings.py`, settings route schemas | migration, frontend settings service/UI |
| Add new AI provider | gateway/factory mappings + config enums | proto/provider enums, settings endpoints/UI |
| Add new gRPC analysis call | `analysis.proto`, backend analysis client | both AI service servers + frontend integration |
| Add market maker field | `market_maker_config` model | migration, route schema, UI form |
| Change backtest persistence | `backtest_run` model + backtesting routes | migration + result rendering in frontend |

## Fast Navigation Tips
- Use `rg` by endpoint string (example: `rg "/api/trades/execute" backend frontend`).
- Use `rg` by model class name to find writers/readers.
- Start from frontend service method name, then jump to route file.
