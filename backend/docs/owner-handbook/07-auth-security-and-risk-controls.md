# 07 - Auth, Security, and Risk Controls

## Goal
Understand how authentication, credential protection, security hardening, and trading risk controls work together.

## Wallet Login Flow (Challenge/Verify)
Primary route file:
- `backend/app/api/routes/auth.py`

Flow:
1. `POST /api/auth/login`
2. Backend generates challenge (wallet + timestamp + nonce).
3. Challenge stored in cache with TTL.
4. Client signs challenge with wallet.
5. `POST /api/auth/verify` with signature.
6. Backend verifies signature and challenge freshness.
7. Access and refresh tokens are created.
8. Tokens are set as HttpOnly cookies.
9. User session state is available via `/api/auth/me`.

## Cookie Token Lifecycle
- Cookie names: `pm_access_token`, `pm_refresh_token`
- Access token: short-lived (minutes)
- Refresh token: longer-lived (days, with keep-logged-in extension)
- Logout revokes/invalidates refresh token and clears cookies
- Frontend refreshes transparently on 401 using interceptor

## Refresh Token Hashing Model
Files:
- `backend/app/security/refresh_tokens.py`
- `backend/app/models/token.py`

Security posture:
- refresh tokens persisted as HMAC-SHA256 hash, not plaintext
- dedicated hash secret preferred (`REFRESH_TOKEN_HASH_SECRET`)
- fallback to JWT secret for backward compatibility

## Credential Encryption and Rotation
Files:
- `backend/app/security/crypto.py`
- `backend/app/security/credential_store.py`

Key points:
- wallet private keys and related credentials are encrypted before caching
- keyring supports multi-key decryption and primary-key encryption (rotation model)
- `ENCRYPTION_MASTER_KEYS` is the preferred source
- local/dev fallback can derive a key from JWT secret
- malformed/undecryptable payloads are deleted (fail-closed behavior)

## Caching and Redis Security Posture
File:
- `backend/app/utils/cache.py`

Behavior:
- challenge cache and opportunity cache can use Redis or in-memory fallback
- credential cache can require Redis outside local env (config-driven)
- fallback behavior is explicit, not silent in strict-required cases

## Security Headers and Origin Controls
File:
- `backend/app/main.py`
- `backend/app/api/routes/auth.py`

Controls include:
- CSP for non-doc routes
- `X-Frame-Options`, `X-Content-Type-Options`, referrer and permissions policies
- origin validation against configured allowed origins for auth endpoints
- strict cookie flags + secure behavior tied to environment

## Config Guardrails
File:
- `backend/app/config.py`

Examples:
- rejects weak/placeholder JWT secrets
- requires debug salt when debug endpoints are explicitly enabled
- validates positive integer performance tuning knobs

## Debug Endpoint Controls
- Debug router registration is environment/flag aware.
- Admin wallet sync logic controls privileged users.
- Avoid enabling debug endpoints in production without explicit hardened config.

## Risk Controls in Trading Flows
Main files:
- `backend/app/api/routes/settings.py`
- `backend/app/api/routes/trades.py`
- services around copy trading, stop-loss, inverse bot, market maker

Exposed controls include:
- max position size
- daily/monthly loss controls
- drawdown and halt controls
- simulation mode
- inverse bot confidence/cooldown/reversal limits
- emergency stop and optional panic sell flow

## Security/Control Change Checklist
- [ ] Does this change affect cookies, token validation, or refresh behavior?
- [ ] Are secret/env requirements updated and documented?
- [ ] Are failure modes fail-closed for credentials and trade execution?
- [ ] Are debug/admin paths still protected?
- [ ] Are risk limits enforced at execution points, not only UI level?
