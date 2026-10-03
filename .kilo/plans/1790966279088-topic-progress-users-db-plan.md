# Users DB + per-topic AI progress that drives next-session question generation

## Goal

A real `users` table in SQLite, plus one cumulative progress document per `(user_id, topic_id)` consisting of
**(1)** a header with the topic name and **(2)** an AI-generated summary. The summary is produced when the learner
presses **Save progress** or automatically when the session ends (including closing the tab). On the next session,
that stored progress is injected into the question-generation prompt so the LLM continues where the learner left
off instead of repeating the same questions.

## Decisions (already agreed — do not re-litigate)

| # | Decision |
|---|---|
| 1 | Add a `users` table to SQLite. Per-user **LLM config stays in the existing Fernet-encrypted JSON store** (`user_settings_store.py`) — do not migrate credentials. |
| 2 | Closing the client: cheap `pagehide` + `fetch(keepalive:true)` beacon that writes raw data only. The AI summary is generated after, never awaited by the browser. |
| 3 | **One row per `(user_id, topic_id)`.** Each save overwrites the summary (previous summary fed back in as context for continuity) and unions new questions into a cumulative `questions_asked_json` capped at 200. |
| 4 | Enforce no-repeats **both** ways: inject a `progress_block` (AI summary) into the prompt **and** union the server-stored `questions_asked` into `existing_questions` → the existing `uniqueness_block`. |
| 5 | Scope: **Phase 1 = topic study + quiz.** Interviews are Phase 2. Voice is out (see Out of scope). |
| 6 | No progress display page, card, or modal. |
| 7 | Three write triggers: server-side capture on every generate call, `pagehide` auto-save, and a **Save progress** button. |
| 8 | The Save progress button saves in the **background** and shows only a brief inline status on the button itself. |
| 9 | All five pieces of backend state (progress DB + four JSON files) move to a mounted `/app/data` via Task 0. This is a prerequisite, not follow-up cleanup. |

## Context the implementer needs

- **No ORM, no Alembic, no migrations.** Schema is `CREATE TABLE IF NOT EXISTS` inside each store's `_init_db()`.
  Copy the house style from `learning_store.py:66-259`; use the `PRAGMA table_info` + conditional `ALTER TABLE`
  patcher idiom from `interview_store.py:102-112` if any column is ever added later.
- **There is no `users` table today.** Identity is the OAuth `sub` string from the JWT cookie, returned by
  `require_auth` (`dependencies.py:62-87`) as `{"user", "provider"}` and threaded everywhere as `user_id`.
- **`learning_attempts` stores no question text** and has no `SELECT` anywhere in the codebase. `question_id` is a
  truncated sha1 and irreversible. The new `topic_progress` table becomes the source of truth for "questions asked"
  going forward; there is no historical backfill.
- **`existing_questions` is currently client-only** — passed straight through at `questions.py:166`, `:217`, `:270`.
  The server has never stored which questions it already asked.
- **`LLMClient.completion()`** (`llm_client.py:98-217`) resolves provider/model/credential through the policy layers
  and returns `{success, analysis, metadata, error, error_code}`. It sends a single `{"role":"user"}` message, so
  context injection is plain prompt text. Policy-blocked calls return `error_code` in `llm_policy.py` rather than
  raising.
- **`run_async` uses a 1-worker `ThreadPoolExecutor`** (`learning_store.py:56-64`). **Never hold the executor across
  an LLM call** — do the LLM work first, then a short `run_async` write.
- **No background-job infrastructure exists.** No `BackgroundTasks`, no task queue. `asyncio.create_task` appears
  only inside `interview_sessions.py` streaming generators. If a fire-and-forget task is needed it must be held in a
  module-level set to avoid GC and guarded by an in-flight key set.

## Schema

New store `backend/app/services/progress_store.py`, own `sqlite3` connection to `settings.learning_db_path`,
mirroring `learning_store.py:45-64` (own connection, `threading.Lock`, `ThreadPoolExecutor`, `run_async`,
`CREATE TABLE IF NOT EXISTS`, ISO-8601 UTC `TEXT` timestamps, `*_json` columns, keyword-only public methods).

```sql
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    provider       TEXT NOT NULL DEFAULT '',
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    session_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topic_progress (
    user_id              TEXT NOT NULL,
    topic_id             TEXT NOT NULL,
    topic_title          TEXT NOT NULL,
    summary_text         TEXT NOT NULL DEFAULT '',
    summary_status       TEXT NOT NULL DEFAULT 'empty',   -- empty|pending|ready|failed
    summary_error        TEXT NOT NULL DEFAULT '',
    questions_asked_json TEXT NOT NULL DEFAULT '[]',     -- newest first, capped 200
    sections_json        TEXT NOT NULL DEFAULT '[]',     -- section headings seen
    attempt_stats_json   TEXT NOT NULL DEFAULT '{}',
    preferred_language   TEXT NOT NULL DEFAULT '',
    question_count       INTEGER NOT NULL DEFAULT 0,
    revision             INTEGER NOT NULL DEFAULT 0,
    provider_used        TEXT NOT NULL DEFAULT '',
    model_used           TEXT NOT NULL DEFAULT '',
    generation_source    TEXT NOT NULL DEFAULT '',       -- ai|fallback
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    PRIMARY KEY (user_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_topic_progress_user_updated
    ON topic_progress(user_id, updated_at DESC);
```

Populate `users` from the progress write path (`upsert_topic_progress` calls `touch_user`), so `users` covers
learners who have studied at least one topic. Hooking `require_auth` would need store injection into
`dependencies.py` — out of scope.

## Task 0 — Persist backend state outside the container filesystem (do this first)

Nothing this feature writes survives a container rebuild today, and the problem is **not** limited to the progress
DB. Five pieces of state are written to unmounted container paths:

| State | Setting | Default | Env var |
|---|---|---|---|
| SQLite (learning, interview, planner, **new progress**) | `learning_db_path` (`config.py:29`) | `/tmp/study_hub_learning.db` | `STUDY_LEARNING_DB_PATH` |
| Encrypted user LLM settings + Google OAuth tokens | `user_settings_file` (`config.py:56`) | `user_llm_settings.json` → `/app/` | `STUDY_USER_SETTINGS_FILE` |
| LLM service allowlist | `llm_service_users_file` (`config.py:57`) | `llm_service_users.json` → `/app/` | `STUDY_LLM_SERVICE_USERS_FILE` |
| Admin LLM assignments | `llm_assignments_file` (`config.py:58`) | `llm_assignments.json` → `/app/` | `STUDY_LLM_ASSIGNMENTS_FILE` |
| Login allowlist | **hardcoded** `ALLOWED_USERS_FILE = Path("allowed_users.json")` (`services/auth.py:16`) | `allowed_users.json` → `/app/` | **none — not configurable** |

`docker-compose.dev.yml` mounts only `./backend/app`, `./backend/docs:ro` and `./backend/main.py`.
`docker-compose.yml` mounts only `./backend/docs:ro`. Both Dockerfiles set `WORKDIR /app`, so today
`docker compose up --build` destroys the progress DB **and** every learner's saved API keys, Google OAuth tokens,
the admin login allowlist and all LLM assignments.

Steps:

1. **`app/config.py`** — add `allowed_users_file: str = "allowed_users.json"` alongside the other three file
   settings (`config.py:56-58`), which yields `STUDY_ALLOWED_USERS_FILE`.
2. **`app/services/auth.py`** — replace the three direct uses of the module constant (lines 23, 24, 45) with a
   lazy accessor that calls `get_settings().allowed_users_file`.
   **Do not resolve it at import time.** `app/main.py:83-102` copies `.env` into `os.environ` *after* module
   imports have begun, so an import-time `get_settings()` would silently miss `.env` values.
   Update the single patch target in `backend/tests/test_auth_admin.py:40` from
   `app.services.auth.ALLOWED_USERS_FILE` to the new accessor.
3. **`docker-compose.dev.yml`** — use a host bind mount so the DB is inspectable from the host:
   ```yaml
       volumes:
         - ./backend/app:/app/app
         - ./backend/docs:/app/docs:ro
         - ./backend/main.py:/app/main.py
         - ./backend/data:/app/data
       environment:
         STUDY_LEARNING_DB_PATH: /app/data/study_hub_learning.db
         STUDY_USER_SETTINGS_FILE: /app/data/user_llm_settings.json
         STUDY_LLM_SERVICE_USERS_FILE: /app/data/llm_service_users.json
         STUDY_LLM_ASSIGNMENTS_FILE: /app/data/llm_assignments.json
         STUDY_ALLOWED_USERS_FILE: /app/data/allowed_users.json
   ```
   **Never nest a volume under `/app/app`.** `./backend/app:/app/app` already shadows the app directory, so a
   nested `/app/app/data` mount would hide `app/data/static_curriculum.json` and break every topic load.
4. **`docker-compose.yml`** — same five env vars plus a named volume rather than a host path, since a prod
   container has no meaningful host filesystem:
   ```yaml
       volumes:
         - ./backend/docs:/app/docs:ro
         - study_data:/app/data
   # file scope:
   volumes:
     study_data:
   ```
5. **`.gitignore`** — add `backend/data/`, `*.db`, `user_llm_settings.json`, `allowed_users.json`,
   `llm_service_users.json`, `llm_assignments.json`. None are ignored today, so running locally from `backend/`
   drops Fernet-encrypted credentials and the login allowlist into the repo working tree as committable untracked
   files.
6. **`.env.example`** — document `STUDY_LEARNING_DB_PATH` and the four file-path settings so a non-Docker local
   run does not silently write to `backend/`.

No `RUN mkdir` is needed in the Dockerfiles: the parent directory is already created by `learning_store.py:46-48`,
`llm_assignments_store.py:42`, `llm_service_access.py:48` and `user_settings_store.py:81`. Note that
`python:3.11-slim` runs as root, so files created in the dev host bind mount will be root-owned on macOS/Linux.

**Render deployment blocker — needs a decision, not code.** `render.yaml` uses `plan: free`, sets no
`STUDY_LEARNING_DB_PATH`, and declares no disk. Render's free plan has an ephemeral filesystem and does not support
persistent disks, so the deployed backend loses all five pieces of state on every deploy. Either move to a paid
plan with a disk mounted at `/app/data`, or accept that production progress is ephemeral and say so explicitly.

## Backend — ordered tasks

1. **`app/services/progress_store.py`** — `ProgressStore` with the DDL above and these methods:
   `touch_user(*, user_id, provider)`, `upsert_topic_progress(*, user_id, provider, topic_id, topic_title,
   questions, sections, attempt_stats, preferred_language)`, `record_asked_questions(*, user_id, topic_id, topic_title,
   questions, section_title, preferred_language)`, `get_progress(*, user_id, topic_id)`,
   `list_progress(*, user_id, limit)`, `get_asked_questions(*, user_id, topic_id, limit)`, `set_summary(*, user_id,
   topic_id, text, provider_used, model_used, source)`, `mark_summary_pending(*, user_id, topic_id)`,
   `mark_summary_failed(*, user_id, topic_id, error)`.
   - Normalize question text with the **same** normalizer the generator uses (`question_generator.py:128`
     `_normalise_question`) when de-duplicating, so stored and client-supplied lists dedupe consistently.
   - `questions_asked_json` is **newest-first**, de-duplicated, hard cap **200**. New questions are prepended.
   - Every write: `with self._lock:` then `with self._conn:`.
   - `get_asked_questions` returns plain question-text strings for the generator, not ids.

2. **`app/services/progress_summarizer.py`** — `ProgressSummarizer(llm_client)`:
   - `build_summary_prompt(*, topic_title, sections, questions, attempt_stats, previous_summary) -> str`
   - `async generate(*, user_id, topic_title, sections, questions, attempt_stats, previous_summary, user_identity,
     llm_config) -> dict` returning `{text, provider_used, model_used, source}`.
   - Call `llm_client.completion(prompt, llm_config, user_identity, task="final")`. Use `task="final"` so the
     learner's configured model is used instead of an eval/planner route.
   - **Never raise and never block the write.** On `success=False`, a policy `error_code`, an empty `analysis`, or
     unparseable output, return a **deterministic fallback** summary built from section headings, per-difficulty
     counts, and the correctness/confidence stats. Mark `source="fallback"`. This mirrors the existing
     deterministic-fallback pattern in `question_generator.py`.
   - Cap the summary at ~2000 chars.

3. **`app/routers/progress.py`** — `APIRouter(prefix="/api/progress")` + module-level `init(...)`, matching the
   `questions.py:42-52` pattern. All endpoints `Depends(require_auth)`:
   - `GET /api/progress/topics` → list of documents for the user.
   - `GET /api/progress/{topic_id}` → one document.
   - `POST /api/progress/{topic_id}/save` → **explicit save**. Write raw data first (durable), then attempt the AI
     summary awaited. Return the full document. This is the synchronous path the Save progress button calls.
   - `POST /api/progress/{topic_id}/autosave` → **the beacon target**. Write raw data, mark `summary_status='pending'`,
     then attempt the summary within the same request (the browser does not wait). On summary failure the record
     stays `pending` for later retry. Keep the payload cap at 60 questions (~15 KB) — `keepalive` bodies are
     limited to 64 KB.
   - New pydantic models in `app/schemas/models.py`: `SaveProgressRequest`, `TopicProgressDocument`,
     `TopicProgressListResponse`, and a `ProgressQuestion` item carrying
     `{question_id, question, difficulty, revealed, is_correct, confidence}`.

4. **`app/main.py`** — construct `ProgressStore(settings.learning_db_path)` and `ProgressSummarizer(_llm_client)` in
   the `lifespan` block near `main.py:125`, call `progress.init(progress_store, summarizer, llm_client)` alongside
   the existing `init()` calls (`main.py:129-145`), and register the router in `create_app()` with
   `auth_dep = [Depends(require_auth)]` (`main.py:212-225`).

5. **`app/services/question_generator.py`** — prompt injection:
   - Add `prior_progress: str = ""` and `additional_existing_questions: Optional[list[str]] = None` to
     `generate_v2_stream` (`:2346-2363`), `generate_v2` (`:2838`), `generate` (`:2733`),
     `generate_quiz_v2_stream` (`:3052-3063`), `generate_quiz_v2` (`:3373`) and `generate_quiz` (`:3331`).
   - In `_build_prompt` (`:1219-1234`) build a `progress_block` next to `uniqueness_block` (`:1317-1323`) using the
     existing `optional_context_block` helper from `app/services/prompt_blocks.py`, and interpolate it into the
     final f-string near line `1382`, just before `uniqueness_block`.
   - Same for `_build_quiz_prompt` (`:1394`). The quiz path currently has **no** dedup state — seed
     `generated_question_texts` from the prior list and apply `_is_near_duplicate_question` (`:152-172`).
   - Merge `existing_questions` + `additional_existing_questions` **client-first**, then de-duplicate via
     `_normalise_existing_questions` (`:194-208`, caps at 80). Client-first preserves today's per-checkpoint
     behaviour exactly; the stored history fills the remaining budget newest-first.
     *Documented limitation:* once the 80-item normalizer cap and the 60-item `uniqueness_block` slice are hit,
     the oldest historical questions are dropped from the prompt. The `progress_block` summary is the mitigation,
     since it covers the whole history semantically.

6. **`app/routers/questions.py`** — server-side capture + prompt wiring:
   - In each of the three `/generate*` call sites (`:166`, `:217`, `:270`) and the three quiz sites, first call
     `progress_store.get_asked_questions(user_id, topic_id, limit=200)` and pass the result as
     `additional_existing_questions`.
   - After a successful stream/batch, call `record_asked_questions` with the returned `QuestionAnswerV2.question`
     texts, `body.section_title`, and `body.preferred_language`. This is the zero-frontend-change capture layer and
     makes the stored list a superset of anything the client sends.
   - Fetch `summary_text` from the same row and pass it as `prior_progress`. If `summary_status == 'pending'`,
     kick off a **tracked, non-blocking** regeneration guarded by an in-flight `(user_id, topic_id)` set held in a
     module-level `set`, so a dropped `keepalive` request is repaired on the next generation for that topic.
   - `questions.py` needs the progress store + summarizer passed through its `init()`.

## Frontend — ordered tasks

7. **`src/types/index.ts`** — add `ProgressQuestion`, `SaveProgressPayload`, `TopicProgressDocument`,
   `TopicProgressListResponse`.

8. **`src/services/api.ts`** — add `saveTopicProgress(topicId, payload)` (axios POST to
   `/progress/{topicId}/save`) and `autosaveTopicProgress(topicId, payload)` — the latter must **not** use axios,
   because it needs `fetch(..., { keepalive: true, credentials: "include" })` to survive unload.

9. **`src/pages/TopicStudy.tsx`** — add a **Save progress** button and the auto-save handler:
   - Build the payload from the existing state already in the component: `questions` (`:127`),
     `submittedAttempts` (`:134`), `confidenceByIdx` (`:138`), `revealedAnswers` (`:131`), `topic` (for the title).
   - **Button:** calls `saveTopicProgress`; shows `Saving…` → `Progress saved` → idle on the button itself only. No
     modal, no panel, no toast library. On failure show `Save failed` on the button. Place it near the existing
     generation controls (`:1378-1497`) and in the "Topic Complete" panel (`:1755-1784`).
   - **Auto-save:** a `useEffect` registering a `pagehide` listener (deps: `topicId`, `questions`,
     `submittedAttempts`, `confidenceByIdx`, `revealedAnswers`) that fires `autosaveTopicProgress` once with the
     last 60 questions. Clean up the listener on unmount. This covers "session ends" and "client closed".
   - No progress display anywhere.

## Failure modes — all handled, none block a save

| Case | Behaviour |
|---|---|
| Learner has no personal credential / not approved for Study App LLM | Summary is policy-blocked → deterministic fallback summary, `source='fallback'`, `status='ready'`. Raw progress is still saved. |
| `keepalive` request dropped (>60 s, browser kills it) | Record stays `summary_status='pending'`; repaired by the non-blocking regeneration in `questions.py` on the next generation for that topic. |
| Browser payload exceeds 64 KB | Capped at 60 questions in the beacon. |
| Learner is anonymous (`local-dev` bypass in `dependencies.py`) | Allowed; `users` row created with `provider='local'`. |
| LLM returns malformed or empty output | Fallback summary; never an exception out of the router. |
| Rate limiter | `/api/progress/**` falls into the default `llm` scope. The autosave beacon is cheap but a reload loop could 429. Confirm `classify_rate_limit_scope` in `app/services/rate_limit.py` handles this acceptably; if not, add autosave to a cheaper scope. |
| Empty/absent progress | `summary_status='empty'`, no `progress_block` in the prompt, `existing_questions` behaviour identical to today. |

## Risks

- **No backend state is currently durable.** The progress DB defaults to `/tmp/study_hub_learning.db` and neither
  compose file mounts a data volume, so progress — plus every learner's encrypted API keys, OAuth tokens, the
  login allowlist and all LLM assignments — is destroyed by `docker compose up --build`. This defeats the whole
  feature. **Handled by Task 0**, which must land before anything writes state.
- **Render free plan has no persistent disk** (`render.yaml:4`, `plan: free`). Production state remains ephemeral
  regardless of the compose work. Needs a user decision — see Task 0.
- **Prompt growth.** Adding `progress_block` + up to 60 more question texts to every prompt increases token use on
  an already large prompt. Cap the progress block at ~1200 chars and state the cap in the plan's code comments.
- **`LLMClient` has no system prompt / multi-turn support.** The summary and the progress block both go in as user
  text. Acceptable; do not attempt to add system-message support here.

## Validation

Backend tests are `unittest.TestCase` (no conftest, no pytest config). Pattern: save `os.environ`, point the
settings files at a `tempfile.TemporaryDirectory()`, `get_settings.cache_clear()`, build a bare `FastAPI()`,
`app.dependency_overrides[require_auth] = lambda: {"user": "alice", "provider": "google"}`, and inject fakes via
the router's `init()`. DB tests use a real temp SQLite file.

- `backend/tests/test_progress_store.py` — DDL creation, upsert, `questions_asked_json` newest-first cap at 200,
  cross-list dedupe using the generator's normalizer, `summary_status` transitions, `users` touch, `revision`
  increments.
- `backend/tests/test_progress_summarizer.py` — `FakeLLM` success; policy `error_code` → fallback;
  empty/malformed analysis → fallback; summary char cap.
- `backend/tests/test_progress_router.py` — save/autosave/list/get contracts; autosave leaves `pending` when the
  summarizer raises.
- `backend/tests/test_questions_router_prior_progress.py` — asserts the router passes `prior_progress` and a
  non-empty `additional_existing_questions`, and that server-side capture records generated question text.
- `backend/tests/test_question_generator_prior_progress.py` — `_build_prompt` and `_build_quiz_prompt` contain the
  progress block; merged `existing_questions` respect the client-first ordering and the 80-item cap; the quiz path
  now seeds dedup state.
- `backend/tests/test_auth_admin.py` — after the `ALLOWED_USERS_FILE` change, point the existing patch at the new
  accessor and confirm the admin allowlist endpoints still round-trip.
- Regressions to re-run unchanged: `test_learning_store.py`, `test_questions_router_custom_topics.py`,
  `test_questions_stream_router.py`, `test_questions_quiz_stream_router.py`,
  `test_question_generator_validation.py`, `test_question_generator_streaming.py`, `test_models_contracts.py`.

Run from `backend/`: `PYTHONPATH=. uv run python -m unittest discover tests`.

**Persistence smoke test (Task 0).** `docker compose -f docker-compose.dev.yml up --build`, study a topic, then
`docker compose -f docker-compose.dev.yml down && docker compose -f docker-compose.dev.yml up` (no `--build`) and
confirm `backend/data/study_hub_learning.db` on the host still contains the `users` and `topic_progress` rows.
Also confirm `backend/data/user_llm_settings.json` still holds the learner's saved provider. Verify the negative
case too: `curl localhost:8001/api/topics` must still return all curriculum topics, proving the new mount did not
shadow `app/data/static_curriculum.json`.

Frontend has **no test runner** — verify manually: click Save progress, check the status label cycles; reload and
confirm `GET /api/progress/{topicId}` returns the record; regenerate questions for the same topic and confirm the
prompt no longer offers previously-asked questions.

## Out of scope

- **Mock interviews (Phase 2).** Needs a real `topic_id` first: add `topic_id: Optional[str]` to
  `CreateInterviewSessionRequest` (`models.py:699-710`) and a `topic_id` column to `interview_sessions`, then
  prefer it over `_track_topic_id(track)` (`interview_sessions.py:84-88`), which today maps **every** `backend`
  interview to `01-backend-fundamentals-and-http` regardless of content. Also switch the synthetic
  `interview:{session_id}:{turn_index}` question id (`interview_sessions.py:353`) to a content hash like
  `_question_id` (`question_generator.py:123-125`) so cross-session dedup becomes possible.
- **Voice chat.** `VoiceSession` (`voice_session.py:44-53`) has no topic field, no store reference and no
  persistence. The WebSocket `start` handler (`voice.py:191-209`) reads only
  `tier/session_type/session_id/system_prompt/llm_config` and silently drops every other key;
  `useVoice.startSession()` has no `topicId`; `VoicePanel` is mounted with no props so it is always a global
  `"chat"` session. Also, "avoid repeating questions" is meaningless for free-form voice Q&A.
- Progress display UI (page, card, modal) — explicitly deferred.
- Migrating LLM credentials out of the encrypted JSON store.
- Populating `users` for every authenticated request rather than only on first progress write.