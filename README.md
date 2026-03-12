# Ace Your Interview

Adaptive interview-prep app for Backend, Frontend, System Design (including infrastructure/cloud), and AI Stack topics.

## What Changed
- Replaced project-specific handbook with a generic interview curriculum (`22` core topics + handbook overview).
- Added explicit candidate levels: `junior`, `mid`, `senior`.
- Added topic metadata and filtering by `track`, `level`, and query text.
- Rebranded frontend, backend app name, and handbook copy to **Ace Your Interview**.
- Added curriculum version reset (`v3_interview_core_infra_cloud_diagrams`) so legacy local progress is reset once.
- Added markdown rendering support for Mermaid diagrams and fenced code panels across study, quiz, chat, and interview report surfaces.

## Features
- Level-aware question and quiz generation.
- Strict v2 grounded generation (`source_quote`, `source_section`, validation and retries).
- Adaptive learning endpoints for attempts, review queue, weak areas, and mastery.
- Track/level filters in topics list.
- Configurable provider/model in settings.
- Personalized mock interview sessions with rubric feedback and report.
- Infrastructure and cloud exam panels (Docker, Terraform, Kubernetes, AWS, GCP, Azure) under the `system_design` track.

## Regenerate Handbook Curriculum
The static handbook can be regenerated to maintain deep topic coverage (`>=1000` sections total across 22 built-in topics).

Generate/update handbook files:
```bash
cd study-app/backend
PYTHONPATH=. uv run python scripts/generate_owner_handbook.py --llm-mode auto
```

Required provider setup:
- Configure provider credentials in environment (for example `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`).
- Ensure `STUDY_DEFAULT_PROVIDER` / `STUDY_DEFAULT_MODEL` map to an available provider/model when using `--llm-mode auto` or `--llm-mode always`.
- Use deterministic generation when no credentials are available:
```bash
PYTHONPATH=. uv run python scripts/generate_owner_handbook.py --llm-mode never
```

Validate handbook depth and per-topic section counts before commit:
```bash
cd study-app/backend
PYTHONPATH=. uv run python scripts/generate_owner_handbook.py --check-only
PYTHONPATH=. uv run python -m unittest -q tests.test_doc_parser tests.test_owner_handbook_depth
```

## Quick Start
### Docker
```bash
cd study-app
docker compose up --build
```
- Frontend: http://localhost:5174
- Backend API: http://localhost:8001

### Local
Backend:
```bash
cd study-app/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e "."
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Frontend:
```bash
cd study-app/frontend
npm install
npm run dev
```

## API Endpoints
| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/topics` | List topics (supports `track`, `level`, `q`) |
| GET | `/api/topics/{id}` | Topic details |
| POST | `/api/questions/generate` | Legacy question generation |
| POST | `/api/questions/generate-v2` | Strict grounded question generation |
| POST | `/api/questions/quiz/generate` | Legacy quiz generation |
| POST | `/api/questions/quiz/generate-v2` | Strict grounded quiz generation |
| POST | `/api/learning/attempts` | Record attempt event |
| GET | `/api/learning/review-queue` | Due review items |
| GET | `/api/learning/weak-areas` | Weak topic summary |
| GET | `/api/learning/mastery` | Mastery by topic |
| POST | `/api/interview-sessions` | Create mock interview session |
| GET | `/api/interview-sessions` | List interview sessions |
| GET | `/api/interview-sessions/stats` | Interview summary metrics |
| GET | `/api/interview-sessions/{id}` | Interview session state + turns |
| POST | `/api/interview-sessions/{id}/answer` | Submit interview answer + rubric |
| POST | `/api/interview-sessions/{id}/next-question` | Generate next interview question |
| GET | `/api/interview-sessions/{id}/report` | Interview report |
| GET | `/api/llm/providers` | Provider availability |
| GET | `/api/llm/health` | Backend health |
| GET | `/api/user-settings` | User LLM settings + credential status |
| PUT | `/api/user-settings/preferences` | Save user provider/model/auth + `llm_source` |
| PUT | `/api/user-settings/api-key/{provider}` | Save user API key |
| DELETE | `/api/user-settings/api-key/{provider}` | Delete user API key |
| GET | `/api/user-settings/google/connect` | Start Gemini OAuth connect |
| GET | `/api/user-settings/google/callback` | Gemini OAuth callback |
| POST | `/api/user-settings/google/disconnect` | Disconnect Gemini account |
| GET | `/api/auth/llm-service-users` | Admin: backend-funded Study App access list |
| POST | `/api/auth/llm-service-users` | Admin: approve backend-funded Study App access |
| DELETE | `/api/auth/llm-service-users/{provider}/{identifier}` | Admin: revoke backend-funded Study App access |
| GET | `/api/auth/llm-assignments/users` | Admin: list allowed-login users + Study App assignment |
| PUT | `/api/auth/llm-assignments/{provider}/{identifier}` | Admin: set fixed Study App provider/model for a user |
| DELETE | `/api/auth/llm-assignments/{provider}/{identifier}` | Admin: remove user Study App assignment |
| GET | `/api/auth/llm-assignments/me` | Admin: get own Study App assignment |
| PUT | `/api/auth/llm-assignments/me` | Admin: set own Study App assignment |
| GET | `/health` | Service health |

## Auth for Local Development
If you want localhost-only bypass while preserving auth for deployment:
```bash
STUDY_DEV_AUTH_BYPASS_LOCALHOST=true
```
For deployment, keep it disabled:
```bash
STUDY_DEV_AUTH_BYPASS_LOCALHOST=false
```

## Admin and User Settings
- `/user-settings`: available to all authenticated users for personal LLM settings.
- `/settings`: admin-only page for allowed users and platform checks.

Configure admins via:
```bash
STUDY_ADMIN_USERS=google:admin@example.com,ops@example.com
```

Configure encrypted user credential storage:
```bash
STUDY_ENVIRONMENT=development
STUDY_CREDENTIALS_ENCRYPTION_KEY=change-me
STUDY_USER_SETTINGS_FILE=user_llm_settings.json
```

Configure backend-funded Study App access and per-user assignments:
```bash
STUDY_LLM_SERVICE_USERS_FILE=llm_service_users.json
STUDY_LLM_ASSIGNMENTS_FILE=llm_assignments.json
```

When `STUDY_ENVIRONMENT` is not `development`, `STUDY_CREDENTIALS_ENCRYPTION_KEY` is required at startup.

Runtime policy:
- `Personal LLM` mode uses only the user credential path (no fallback).
- `Study App LLM` mode requires admin approval + admin assignment, then uses backend-funded provider/model.

## Mock Interview Feature Flag
Mock interview endpoints are gated by:
```bash
STUDY_ENABLE_MOCK_INTERVIEW_V1=true
```
Default is `true`. Set it to `false` only if you explicitly want to hide or disable mock interviews.
