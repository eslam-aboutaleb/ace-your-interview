# 11 - Change Playbooks

## Goal
Use repeatable, safe procedures for common codebase changes.

---

## Playbook A: Add a New Backend Endpoint Safely

### Use When
You need a new HTTP API endpoint or to extend an existing route domain.

### Steps
1. Choose route module by domain (`auth`, `trades`, `settings`, etc.).
2. Define request/response Pydantic models in that route file (or shared schema module if needed).
3. Add endpoint with `Depends(get_db)` and auth dependency if protected.
4. Delegate business logic to service module (avoid heavy route logic).
5. Add/update model interactions if persistence is needed.
6. Add tests for success and failure paths.
7. Add frontend service method and UI integration if required.
8. Update handbook navigation doc if endpoint is user-facing.

### Validation
- Endpoint works in `/docs` and from frontend service call.
- Auth/permission behavior is correct.
- Error handling returns expected status and shape.

### Rollback Notes
- If endpoint is additive and unused, rollback is route removal.
- If endpoint introduced side effects, revert service/model changes together.

---

## Playbook B: Add a New Frontend Page + Service Integration

### Use When
You need a new route/page that consumes backend APIs.

### Steps
1. Add or extend service function in `frontend/src/services/*`.
2. Add page/component under `pages/` or `components/` by existing pattern.
3. Register route in `frontend/src/App.tsx`.
4. Add navigation entry if page should be reachable from menu.
5. Wire loading/error states and auth assumptions.
6. For streaming endpoints, include abort/cancel behavior.
7. Add/update relevant TS types.

### Validation
- Page renders under expected auth state.
- API calls succeed and recover from 401 refresh path.
- Type-check/build passes.

### Rollback Notes
- Revert route + navigation + service method together.
- Ensure no orphaned links/menu items remain.

---

## Playbook C: Add Model + Migration + API Changes

### Use When
A new field/table is required for feature behavior.

### Steps
1. Update SQLAlchemy model in `backend/app/models/`.
2. Generate migration revision (`alembic revision --autogenerate ...`).
3. Review migration for nullability/default/backfill correctness.
4. Apply migration locally (`alembic upgrade head`).
5. Update route/service schemas and logic.
6. Update frontend types/services if response contract changed.
7. Add tests around new constraints and behavior.

### Validation
- Fresh DB migration succeeds.
- Existing DB migration path succeeds.
- Endpoint contract remains backward-compatible unless intentionally versioned.

### Rollback Notes
- Have explicit downgrade or forward-fix strategy.
- Avoid irreversible destructive migration without backup plan.

---

## Playbook D: Add New AI Provider or Provider Setting

### Use When
You need to expose a new LLM provider/model option.

### Steps
1. Extend provider enum/mapping in backend gateway (`llm_gateway.py`).
2. Extend provider handling in AI service factory (`llm_factory.py` for llm-chain).
3. If needed, extend `analysis.proto` enum and regenerate stubs.
4. Update settings endpoints/validation exposing provider options.
5. Update frontend settings service/UI provider lists.
6. Add provider-specific configuration/env docs.
7. Add tests for routing precedence (request override -> user setting -> default).

### Validation
- Provider appears in settings APIs/UI.
- Request-level override reaches correct backend/provider.
- Missing key/failure behavior is explicit and safe.

### Rollback Notes
- Remove provider mapping from all layers (gateway, service, UI).
- Keep proto compatibility in mind when removing enum values.

---

## Pre-Merge Global Checklist (Any Playbook)
- [ ] Unit/integration checks pass for touched modules
- [ ] No duplicate logic introduced against existing service layers
- [ ] Security and risk controls still enforced
- [ ] Logging is sufficient to debug new path
- [ ] Docs updated where behavior contract changed
