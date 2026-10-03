# Ace Your Interview

[![Backend CI](https://github.com/eslam-aboutaleb/ace-your-interview/actions/workflows/backend.yml/badge.svg)](https://github.com/eslam-aboutaleb/ace-your-interview/actions/workflows/backend.yml)
[![Frontend CI](https://github.com/eslam-aboutaleb/ace-your-interview/actions/workflows/ci.yml/badge.svg)](https://github.com/eslam-aboutaleb/ace-your-interview/actions/workflows/ci.yml/badge.svg)

Adaptive, LLM-powered interview preparation platform. Level-aware question and quiz generation, FSRS-4.5 spaced repetition, citation-grounded RAG chat, flashcard decks with Anki interop, and full mock-interview sessions with rubric feedback — across Backend, Frontend, System Design (including infrastructure/cloud), and AI Stack tracks.

## Features

### Curriculum & study
- **22 built-in topics** across four tracks — `backend` (5), `frontend` (4), `system_design` (9, incl. Docker, Terraform, Kubernetes, AWS, GCP, Azure), `ai_stack` (4) — each with `junior` / `mid` / `senior` levels and 1000+ handbook sections.
- Track / level / full-text filtering, per-topic preferences, autosave, and generated topic content.
- Optional YouTube topic-video recommendations (quota-aware, cached, admin-toggleable).

### Generation & grounding
- Question and quiz generation in legacy and **strict v2** modes: v2 emits `source_quote` / `source_section`, validates against the curriculum, retries on failure, and streams (`/stream` variants).
- Citation-grounded **RAG chat** over user-ingested documents (sqlite-vec vector search), with follow-up turns and custom-topic generation.
- On-demand hints for any generated question.

### Adaptive learning
- **FSRS-4.5** spaced-repetition scheduler (`fsrs>=4.1,<4.2`) driving the review queue, mastery, weak-area analysis, study plan, forecast, and calibration endpoints.
- Flashcard **decks** with Anki import/export, duplicate detection, and per-card review recording.
- Progress trends and personalized recommendations.

### Mock interviews
- Multi-turn mock interview sessions with rubric-scored answers, streaming answer/next-question endpoints, session stats, trends, and a generated report.
- **Interview+ personalization**: resume parsing (PDF/DOCX), job-description analysis, company packs, and a STAR-story bank.
- Voice-to-voice interviews (edge-tts + browser VAD over WebSocket) behind the `STUDY_ENABLE_VOICE_INTERVIEW_V1` flag (default off).

### Platform
- Multi-provider LLM routing via LiteLLM: Groq (default, free tier), OpenAI, Anthropic, Google, GitHub Models, and local Ollama.
- Google + GitHub OAuth login with allow-lists, JWT sessions, and admin surfaces.
- Two-tier LLM policy: **Personal LLM** (user's own key, no fallback) and **Study App LLM** (backend-funded, admin-approved + admin-assigned).
- Per-scope sliding-window rate limiting (OAuth / session / LLM scopes; optional Redis backend with strict mode), per-user LLM call budget and concurrency caps.
- Fernet-encrypted credential storage at rest; untrusted-input containment in the generation pipeline.

## Architecture

```
┌────────────────────────────┐        ┌──────────────────────────────────┐
│  React 18 + TypeScript     │  HTTP  │  FastAPI (Python 3.11+)          │
│  Vite · Tailwind · Zustand │ ─────► │  routers/  (14 API routers)      │
│  react-router · PWA        │ ◄───── │  services/ (45+ services)        │
│  react-markdown · mermaid  │  WS    │  schemas/models.py               │
└────────────────────────────┘        └───────────┬──────────────────────┘
                                                  │
        ┌─────────────────┬───────────────────────┼────────────────────┐
        ▼                 ▼                       ▼                    ▼
  SQLite (+sqlite-vec)  LiteLLM providers   MCP gateway        edge-tts / VAD
  learning.db           groq/openai/         agentic tools      voice stream
  (progress, cards,     anthropic/google/
  docs, vectors)        ollama
```

- **Backend** (`backend/app/`): `routers/` expose the REST/WebSocket API; `services/` hold the domain logic (LLM client + policy, FSRS scheduler, RAG service, document pipeline, interview store, voice session, rate limiter, MCP gateway, agentic tools); `schemas/models.py` defines the Pydantic contract; `config.py` centralizes settings.
- **Frontend** (`frontend/src/`): `pages/` (17 routes), `components/` (layout, voice, common), `store/` (Zustand: auth, voice, etc.), `hooks/`, `services/` (API client).
- **Evals** (`backend/evals/`): golden fixtures, prompt snapshots, and a deterministic quality harness (`scripts/eval_agent_quality.py`) gating schema validity, prompt contracts, untrusted-input containment, duplicate/grounding/repair/degraded budgets.

## Repository structure

```
ace-your-interview/
├── backend/
│   ├── app/
│   │   ├── routers/        # 14 API routers (topics, questions, learning, cards, ...)
│   │   ├── services/       # LLM, FSRS, RAG, interview, voice, auth, rate-limit, ...
│   │   ├── schemas/        # Pydantic models (API contract)
│   │   ├── data/           # static_curriculum.json (22 topics, 1000+ sections)
│   │   ├── generated/      # vendored protobuf stubs (gRPC client)
│   │   ├── config.py       # all settings + feature flags
│   │   └── main.py         # app factory, lifespan wiring, middleware
│   ├── docs/owner-handbook/  # generated curriculum handbook
│   ├── evals/                # golden fixtures + prompt snapshots
│   ├── scripts/              # eval harness, handbook generator, company-pack seeder
│   ├── tests/                # pytest suite (coverage gate ≥ 90%, currently ~98%)
│   └── pyproject.toml
├── frontend/               # React 18 + Vite + TypeScript + Tailwind
├── .github/workflows/      # backend.yml · ci.yml · eval-live.yml
├── docker-compose.yml      # full-stack local stack
├── render.yaml             # Render backend service
└── start.sh                # one-command launcher
```

## Quick start

### Docker
```bash
git clone https://github.com/eslam-aboutaleb/ace-your-interview.git
cd ace-your-interview
docker compose up --build
```
- Frontend: http://localhost:5174
- Backend API: http://localhost:8001

### Local
Backend:
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill in at least one provider key
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Frontend:
```bash
cd frontend
npm install
npm run dev
```

## Configuration

Copy `backend/.env.example` to `backend/.env`. Key variables:

| Variable | Purpose |
| --- | --- |
| `STUDY_DEFAULT_PROVIDER` / `STUDY_DEFAULT_MODEL` | Default LLM routing (default: `groq` / `llama-3.3-70b-versatile`) |
| `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GITHUB_TOKEN` | Provider credentials |
| `STUDY_AUTH_SECRET_KEY` | JWT session signing (32+ chars, required) |
| `STUDY_CREDENTIALS_ENCRYPTION_KEY` | Fernet key for per-user credentials at rest (required outside `development`) |
| `STUDY_GITHUB_CLIENT_ID/SECRET`, `STUDY_GOOGLE_CLIENT_ID/SECRET` | OAuth apps |
| `STUDY_ALLOWED_GITHUB_USERS`, `STUDY_ALLOWED_GOOGLE_EMAILS` | Login allow-lists |
| `STUDY_ADMIN_USERS` | Admin surface access (`google:admin@example.com`) |
| `STUDY_CORS_ORIGINS`, `STUDY_FRONTEND_URL` | Frontend origin for cookies/CORS |

Feature flags (all `STUDY_ENABLE_*`, default `true` unless noted): `MOCK_INTERVIEW_V1`, `FSRS_V1`, `FLASHCARDS_V1`, `INTERVIEW_PLUS_V1`, `VOICE_INTERVIEW_V1` (default `false`), `TOPIC_VIDEOS` (default `false`).

Local-dev auth bypass (localhost only — keep `false` in deployment):
```bash
STUDY_DEV_AUTH_BYPASS_LOCALHOST=true
```

## API reference

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Service health |
| **Topics** | | |
| GET | `/api/topics` | List topics (`track`, `level`, `q` filters) |
| GET | `/api/topics/{id}` | Topic details |
| GET/PUT | `/api/topics/{id}/preferences` | Per-topic study preferences |
| POST | `/api/topics/{id}/save`, `/autosave` | Persist topic progress |
| POST | `/api/topics/{id}/content/generate/stream` | Stream generated topic content |
| GET | `/api/topics/{id}/videos` | Topic video recommendations |
| GET/POST | `/api/features/topic-videos/*` | Video feature status / metrics / events |
| **Questions & quizzes** | | |
| POST | `/api/questions/generate`, `/generate-v2`, `/generate-v2/stream` | Question generation (legacy, strict, streaming) |
| POST | `/api/questions/quiz/generate`, `/generate-v2`, `/generate-v2/stream` | Quiz generation (legacy, strict, streaming) |
| POST | `/api/questions/{question_id}/hint` | On-demand hint |
| **Learning** | | |
| POST | `/api/learning/attempts` | Record an attempt event |
| GET | `/api/learning/review-queue` | Due review items (FSRS) |
| GET | `/api/learning/weak-areas` | Weak topic summary |
| GET | `/api/learning/mastery` | Mastery by topic |
| GET | `/api/learning/study-plan`, `/forecast`, `/calibration`, `/recommendations` | Planning endpoints |
| GET/PUT/POST | `/api/learning/profile`, `/profile/diagnostic`, `/review` | Learning profile & review recording |
| **Flashcards** | | |
| GET/POST | `/api/decks` | List / create decks |
| GET/PUT/DELETE | `/api/decks/{deck_id}` | Deck management |
| GET | `/api/decks/{deck_id}/cards`, `/export` | Deck cards / Anki export |
| POST | `/api/decks/import`, `/find-duplicates` | Anki import / duplicate detection |
| GET/POST/PUT/DELETE | `/api/cards/*` | Card CRUD and review |
| **Documents & chat** | | |
| GET/POST | `/api/documents` | List / ingest documents |
| GET/DELETE | `/api/documents/{document_id}` | Document management |
| POST | `/api/chat/ask`, `/follow-up` | Citation-grounded RAG chat |
| POST | `/api/topics/custom`, `/custom/stream` | Custom topic generation |
| **Mock interviews** | | |
| POST | `/api/interview-sessions` | Create session |
| GET | `/api/interview-sessions`, `/stats`, `/trends` | List / metrics |
| GET | `/api/interview-sessions/{id}` | Session state + turns |
| POST | `/api/interview-sessions/{id}/answer`, `/answer/stream` | Submit answer + rubric |
| POST | `/api/interview-sessions/{id}/next-question`, `/next-question/stream` | Next question |
| GET | `/api/interview-sessions/{id}/report` | Interview report |
| POST/DELETE | `/api/interview-sessions/resume` | Resume parsing |
| POST | `/api/interview-sessions/jd-analysis` | Job-description analysis |
| GET | `/api/interview-sessions/companies` | Company packs |
| **STAR stories** | | |
| GET/POST | `/api/star-stories`, `/suggest` | Story bank / suggestions |
| GET/PUT/DELETE | `/api/star-stories/{story_id}` | Story management |
| **LLM & settings** | | |
| GET | `/api/llm/providers`, `/api/llm/health` | Provider availability / health |
| GET | `/api/llm/ollama/models`, `/ollama/test` | Local Ollama integration |
| GET/PUT | `/api/user-settings`, `/preferences` | User LLM settings |
| PUT/DELETE | `/api/user-settings/api-key/{provider}` | Per-provider API key management |
| GET/POST | `/api/user-settings/google/connect`, `/google/disconnect` | Gemini OAuth |
| **Auth** (`/api/auth`) | | |
| GET | `/github`, `/github/callback`, `/google`, `/google/callback` | OAuth flows |
| GET | `/me`, `/profile` · POST `/logout` | Session management |
| GET/POST/DELETE | `/allowed-users/*` | Login allow-list admin |
| GET/POST/DELETE/PUT | `/llm-service-users/*`, `/llm-assignments/*` | Backend-funded access admin |
| **Voice** (`/api/voice`) | | |
| GET | `/config` | Voice config (tier, provider) |
| PUT | `/settings` | Admin voice settings |
| WS | `/stream` | Voice-to-voice interview stream |

## Auth & security

- **OAuth**: Google and GitHub login with configurable allow-lists; sessions are signed JWTs (`STUDY_AUTH_SECRET_KEY`).
- **Admin**: `STUDY_ADMIN_USERS` gates the `/settings` admin page and the backend-funded LLM access surfaces.
- **Credential storage**: per-user provider keys are Fernet-encrypted (`STUDY_CREDENTIALS_ENCRYPTION_KEY`); outside `development` the key is required at startup.
- **Rate limiting**: per-scope sliding windows (OAuth 20/5min, session 90/min, LLM 30/min) keyed by user identity when authenticated; optional Redis backend (`STUDY_RATE_LIMIT_BACKEND=redis`) with `STUDY_RATE_LIMIT_STRICT` fail-fast for multi-replica deployments.
- **LLM budget**: per-user call budget (120/min) and concurrency cap (4) bound provider fan-out per request.
- **State files** (`user_llm_settings.json`, `allowed_users.json`, `llm_service_users.json`, `llm_assignments.json`, `*.db`) live in `backend/data/` and are git-ignored — never commit them.

## Quality & evals

- **Backend CI** (`.github/workflows/backend.yml`, every push): pytest with coverage (gate: ≥90% statement+branch, currently ~98%), `ruff` lint, and the deterministic eval harness — production-validator agreement, prompt contracts, untrusted-input containment, golden prompt digests, and duplicate/grounding/repair/degraded budgets.
- **Frontend CI** (`.github/workflows/ci.yml`, main): `npm ci` + build + `vitest`.
- **Live eval** (`.github/workflows/eval-live.yml`, weekly cron + manual): regenerates golden fixtures against real providers and reports a reviewable drift diff; deliberately not on push (nondeterministic, costs tokens).

Regenerate the handbook curriculum (maintains ≥1000 sections across the 22 topics):
```bash
cd backend
PYTHONPATH=. uv run python scripts/generate_owner_handbook.py --llm-mode auto   # or: never
PYTHONPATH=. uv run python scripts/generate_owner_handbook.py --check-only
PYTHONPATH=. python -m unittest -q tests.test_doc_parser tests.test_owner_handbook_depth
```

## Deployment

- **Backend**: Render (`render.yaml`) — Python service, health check on `/health`.
- **Frontend**: Vercel (`frontend/vercel.json`) — set `STUDY_CORS_ORIGINS` and `STUDY_FRONTEND_URL` to the deployed origin.
- **Docker**: `docker-compose.yml` (production-ish) and `docker-compose.dev.yml` (dev mounts); `start.sh` checks optional gRPC backends (ports 50051/50052) before launching.

## License

MIT — see [LICENSE](LICENSE).
