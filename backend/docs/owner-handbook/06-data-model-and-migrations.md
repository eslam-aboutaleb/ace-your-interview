# 06 - Data Model and Migrations

## Goal
Understand core entities, schema evolution, and safe change workflow from model to migration to API.

## Core Ownership Domains

### Identity and Auth
Key models:
- `User`
- `RefreshToken`
- `AdminAuditLog`

Ownership concerns:
- wallet identity
- session lifecycle
- token revocation and auditing

### User Preferences and Risk
Key models:
- `UserSettings`
- `StopLossOrder`
- `TakeProfitOrder`
- `InverseBotPosition`
- `InverseBotAction`

Ownership concerns:
- trading/risk configs
- automation thresholds
- per-user controls and cooldown/halts

### Trading and Monitoring
Key models:
- `TradeHistory`
- `UserTrade`
- `TraderPositionState`
- `FollowedTrader`
- `NotificationFollowedTrader`
- `NotificationFeedEvent`

Ownership concerns:
- observed trader activity
- copied/user execution records
- open/close event feeds

### Market and Analysis
Key models:
- `Market`
- `Winner`
- `Assessment`

Ownership concerns:
- market metadata cache
- AI assessment persistence/history
- leaderboard support

### Strategy Modules
Key models:
- `MarketMakerConfig`
- `BacktestRun`

Ownership concerns:
- market maker lifecycle and metrics
- backtest runs/results

## Migration System
Files:
- Alembic config: `backend/alembic.ini`
- Environment script: `backend/migrations/env.py`
- Revisions: `backend/migrations/versions/*.py`

### Observed Revision History
- `20260228_0001_baseline_bootstrap`: baseline schema bootstrap via model metadata create_all
- `20260228_0002_security_hardening`: token hash migration + check constraints
- `20260228_0003_timezone_aware_datetimes`: datetime standardization
- `20260301_0001_add_llm_provider_prefs`: preferred provider/model fields in settings
- `20260301_0002_add_market_maker_and_backtesting`: new strategy tables

## Migration Safety Practices Present
- non-destructive downgrade choices in critical revisions
- explicit data migration for sensitive token changes
- DB check constraints for range/non-negative validations
- environment-driven secret requirement for migration safety path

## Schema Change Lifecycle (How To Evolve Safely)
1. Update/introduce SQLAlchemy model.
2. Create Alembic revision.
3. Backfill/transform data if needed.
4. Add/adjust route/service/schema behavior.
5. Update tests for new constraints/behavior.
6. Validate migration path on clean DB and existing-like DB.

## API Impact Mapping
When schema changes, verify:
- request/response models in route modules
- service logic assumptions (nullability/defaults)
- frontend type definitions and service adapters
- background jobs reading/writing affected columns

## Common Schema Risks
- adding non-null columns without safe defaults/backfill
- breaking enum/value contracts used by frontend
- changing precision/units for size/price/percent fields
- forgetting migration for model changes

## Practical Commands
From `backend/`:
```bash
uv run alembic revision --autogenerate -m "describe_change"
uv run alembic upgrade head
```

In Docker backend startup, migrations are also applied before app launch.

## Ownership Checklist Before Merging Schema Changes
- [ ] Migration upgrades successfully on fresh DB
- [ ] Migration upgrades successfully on pre-change DB snapshot
- [ ] API behavior remains compatible or versioned intentionally
- [ ] Tests added/updated
- [ ] Rollback/downgrade strategy understood
