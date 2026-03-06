# Polymarket Ownership Handbook

This handbook is for a junior engineer becoming responsible for this repository.
It is written as a maintainership path: understand architecture, trace flows, change code safely, and operate/debug confidently.

## Who This Is For
- New or junior fullstack engineers onboarding to this codebase
- Engineers taking ownership of day-to-day fixes and feature delivery
- Engineers preparing to handle incidents, hotfixes, and safe schema/API changes

## How To Use This Handbook
1. Read in order once.
2. During real tasks, jump to the relevant module (routing, migrations, auth, AI services, etc.).
3. Use the playbooks before making non-trivial code changes.
4. Use the 30-day plan as your onboarding execution checklist.

## Suggested Reading Order
1. [01-system-map.md](./01-system-map.md)
2. [02-tech-stack.md](./02-tech-stack.md)
3. [03-backend-architecture.md](./03-backend-architecture.md)
4. [04-frontend-architecture.md](./04-frontend-architecture.md)
5. [05-ai-services-and-grpc.md](./05-ai-services-and-grpc.md)
6. [06-data-model-and-migrations.md](./06-data-model-and-migrations.md)
7. [07-auth-security-and-risk-controls.md](./07-auth-security-and-risk-controls.md)
8. [08-design-patterns-in-this-codebase.md](./08-design-patterns-in-this-codebase.md)
9. [09-endpoint-and-service-navigation.md](./09-endpoint-and-service-navigation.md)
10. [10-debugging-testing-and-operations.md](./10-debugging-testing-and-operations.md)
11. [11-change-playbooks.md](./11-change-playbooks.md)
12. [12-30-day-ownership-plan.md](./12-30-day-ownership-plan.md)
13. [13-glossary.md](./13-glossary.md)

## If You Only Have 1 Hour
1. Read [01-system-map.md](./01-system-map.md) to understand moving parts.
2. Read [03-backend-architecture.md](./03-backend-architecture.md) and [04-frontend-architecture.md](./04-frontend-architecture.md) for navigation.
3. Skim [09-endpoint-and-service-navigation.md](./09-endpoint-and-service-navigation.md) to know where to edit.
4. Skim [10-debugging-testing-and-operations.md](./10-debugging-testing-and-operations.md) before running or changing anything.

## Cross-Reference Index
- Architecture map: [01-system-map.md](./01-system-map.md)
- Tech catalog: [02-tech-stack.md](./02-tech-stack.md)
- Backend deep dive: [03-backend-architecture.md](./03-backend-architecture.md)
- Frontend deep dive: [04-frontend-architecture.md](./04-frontend-architecture.md)
- AI/gRPC/MCP: [05-ai-services-and-grpc.md](./05-ai-services-and-grpc.md)
- Data model + Alembic: [06-data-model-and-migrations.md](./06-data-model-and-migrations.md)
- Auth + security + risk: [07-auth-security-and-risk-controls.md](./07-auth-security-and-risk-controls.md)
- Design patterns: [08-design-patterns-in-this-codebase.md](./08-design-patterns-in-this-codebase.md)
- Endpoint/service map: [09-endpoint-and-service-navigation.md](./09-endpoint-and-service-navigation.md)
- Debug/ops/tests: [10-debugging-testing-and-operations.md](./10-debugging-testing-and-operations.md)
- Change playbooks: [11-change-playbooks.md](./11-change-playbooks.md)
- 30-day path: [12-30-day-ownership-plan.md](./12-30-day-ownership-plan.md)
- Glossary: [13-glossary.md](./13-glossary.md)

## Scope and Non-Scope
- Scope: maintainership learning, code navigation, safe change process.
- Non-scope: replacing existing project docs at repo root.

For setup commands and product-level intro docs, also see:
- `README.md`
- `QUICKSTART.md`
- `AUTH_GUIDE.md`
