# 10 - Debugging, Testing, and Operations

## Goal
Operate the project locally, debug issues quickly, and run high-value validations before merging.

## Local Run Modes

### Mode A: Manual (recommended for development)
Backend:
```bash
cd backend
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:
```bash
cd frontend
pnpm install
pnpm dev
```

### Mode B: Docker Compose (full stack)
```bash
docker-compose up
```

Services include:
- postgres
- redis
- llm-chain
- cli-agent
- backend
- frontend

## Highest-Value Checks Before/After Changes

### Backend tests
From `backend/`:
```bash
PYTHONPATH=. uv run pytest -q
```

### Frontend type/build checks
From `frontend/`:
```bash
pnpm type-check
pnpm build
```

### Health checks
- Backend health: `GET /health`
- Analysis route health: `GET /api/analysis/health`
- Optional debug health: `GET /api/debug/health` (when enabled)

## Common Failure Modes and Triage

### 1) Login/session fails
Check:
- CORS allowed origins configuration
- secure cookie behavior vs local/prod env
- refresh endpoint behavior (`/api/auth/refresh`)
- frontend interceptor path in `apiClient.ts`

### 2) AI analysis endpoints fail/time out
Check:
- gRPC host/port env vars in backend
- llm-chain/cli-agent service health and logs
- provider API key availability
- proto/stub sync mismatch

### 3) Trade monitor/copy behavior inconsistent
Check:
- followed traders in DB
- monitor task startup logs
- websocket fallback to polling status
- credential cache availability (Redis/encryption settings)

### 4) Migration/startup errors
Check:
- `JWT_SECRET_KEY` and other required env vars
- DB connectivity
- Alembic head consistency
- model import side effects for metadata

### 5) Frontend data appears stale
Check:
- poll intervals and in-flight guards
- SSE stream connection and abort handling
- service endpoint path correctness
- backend response timing/errors

## Logging and Debug Endpoints
- Request logging middleware runs globally.
- Debug routes are environment/flag gated.
- Do not rely on debug endpoints being present in production.
- Use structured route + service logs for incident triage.

## Practical Incident Workflow
1. Reproduce with one endpoint/user flow.
2. Confirm backend route reached.
3. Check service-layer logs/external calls.
4. Confirm DB read/write effects.
5. Validate frontend rendering/state transitions.
6. Add regression test for discovered bug.

## Deployment-Safety PR Checklist
- [ ] Scope is isolated and reviewed by domain (auth, trading, ai, settings)
- [ ] No secret-hardcoded/config-regression risk
- [ ] Required migrations included and tested
- [ ] Backend tests pass
- [ ] Frontend type/build checks pass
- [ ] Relevant manual happy-path test executed
- [ ] Rollback strategy defined for risky changes

## Recommended Verification Scenarios
1. Wallet login + token refresh + logout
2. One market analysis request (and stream variant)
3. One trade history fetch and one simulated/real execution path
4. One settings update round trip (backend + UI)
5. One background monitor-related scenario (stop-loss/inverse/copy follow)
