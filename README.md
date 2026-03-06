# Polymarket Study Hub

AI-powered study application that generates interview questions from the Polymarket system documentation.

## Features

- **13 Study Topics** — parsed from the owner-handbook docs covering system architecture, backend, frontend, AI services, data models, security, design patterns, and more
- **LLM-Generated Q&A** — real-time interview question generation via gRPC connection to `llm-chain` or `cli-agent`
- **Configurable LLM** — switch between OpenAI, Anthropic, Google, Groq, Ollama, or GitHub Models at any time
- **Quiz Mode** — flash-card style study with self-scoring and a completion ring
- **Udemy-Themed UI** — purple accent, card layout, progress tracking, smooth Framer Motion animations
- **Progress Persistence** — localStorage tracks completed topics and answered questions

## Prerequisites

At least one of the gRPC LLM backends must be running:

```bash
# From the project root
docker compose up -d llm-chain    # port 50051
# OR
docker compose up -d cli-agent    # port 50052
```

## Quick Start

### Option 1: Docker Compose (recommended)

```bash
cd study-app
docker compose up --build
```

- Frontend: http://localhost:5174
- Backend API: http://localhost:8001

### Option 2: Local development

**Backend:**

```bash
cd study-app/backend

# Create venv & install
python3 -m venv .venv && source .venv/bin/activate
pip install -e "."

# Compile proto stubs
mkdir -p app/generated
python -m grpc_tools.protoc \
  -I./proto \
  --python_out=./app/generated \
  --grpc_python_out=./app/generated \
  proto/analysis.proto

# Fix import path
sed -i '' 's/^import analysis_pb2/from app.generated import analysis_pb2/' \
  app/generated/analysis_pb2_grpc.py

touch app/generated/__init__.py

# Set env vars for local gRPC
export STUDY_GRPC_LLM_CHAIN_HOST=localhost
export STUDY_GRPC_CLI_AGENT_HOST=localhost
export STUDY_DOCS_PATH=docs

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Frontend:**

```bash
cd study-app/frontend
npm install
npm run dev
```

Open http://localhost:5174

## API Endpoints

| Method | Path                      | Description                    |
| ------ | ------------------------- | ------------------------------ |
| GET    | `/api/topics`             | List all study topics          |
| GET    | `/api/topics/{id}`        | Get topic detail with sections |
| POST   | `/api/questions/generate` | Generate interview Q&A via LLM |
| POST   | `/api/questions/generate-v2` | Generate grounded Q&A (strict schema) |
| POST   | `/api/questions/quiz/generate` | Generate quiz questions (legacy) |
| POST   | `/api/questions/quiz/generate-v2` | Generate grounded quiz (strict schema) |
| POST   | `/api/learning/attempts` | Record study/quiz attempt for adaptive scheduling |
| GET    | `/api/learning/review-queue` | Get due review items |
| GET    | `/api/learning/weak-areas` | Get weakest topics summary |
| GET    | `/api/learning/mastery` | Get topic mastery scores |
| GET    | `/api/llm/providers`      | List available LLM providers   |
| GET    | `/api/llm/health`         | Check gRPC backend health      |
| GET    | `/health`                 | App health check               |

## LLM Configuration

From the Settings page in the UI, you can:

1. **Select a provider** — OpenAI, Anthropic, Google Gemini, Groq, Ollama, or GitHub Models
2. **Choose a model** — model list updates per provider
3. **Adjust temperature** — 0 (precise) to 1.5 (creative)
4. **Check health** — verify gRPC connectivity to the LLM backends

Providers route to the correct backend automatically:

- OpenAI / Anthropic / Google / Groq / Ollama → `llm-chain` (port 50051)
- GitHub Models → `cli-agent` (port 50052)

## Localhost Auth Bypass (Dev)

For local-only development, you can bypass login while still keeping auth enabled elsewhere:

```bash
STUDY_DEV_AUTH_BYPASS_LOCALHOST=true
```

When enabled, requests coming from `localhost` / `127.0.0.1` are treated as an authenticated local dev user.

For deployment, keep this disabled:

```bash
STUDY_DEV_AUTH_BYPASS_LOCALHOST=false
```

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Frontend   │────▶│   Backend    │────▶│   llm-chain      │
│  React+Vite  │     │   FastAPI    │     │   (gRPC:50051)   │
│  :5174       │     │   :8001      │     └──────────────────┘
└──────────────┘     │              │     ┌──────────────────┐
                     │  reads docs/ │────▶│   cli-agent      │
                     │              │     │   (gRPC:50052)   │
                     └──────────────┘     └──────────────────┘
```
