# Plan: Interview Personalization (Resume/JD, Hints, STAR Bank, Company Packs, Voice-to-Voice)

## Goal
Make mock interviews adaptive to the candidate: parse resumes and job descriptions, run JD-vs-resume gap analysis, add a 4-level hint system, a STAR story bank, company-specific question packs, and wire the existing voice agent into voice-to-voice mock interviews.

## Context
- `backend/app/routers/interview_sessions.py` already accepts `target_role`, `job_description_text`, `resume_summary_text`, `focus_areas` as raw text — no parsing, no comparison.
- `backend/app/services/voice_session.py` already supports `session_type: "interview"` with a default persona — but the UI interview flow is text-only; voice and interview are not connected.
- Depends on `1790987112911-fsrs-spaced-repetition-engine.md` (interview attempts already flow into the learning store) and optionally on `1790987112911-document-ingestion-rag-chat.md` (resume text could also come from uploaded documents).

## Decisions (resolved)
- Resume/JD parsing = LLM structured extraction (reuse `LLMClient` + Pydantic output schema), not a dedicated NLP library.
- Hints: 4 levels (gentle nudge → direction → partial solution → full walkthrough), LLM-generated, cost-capped at 4 per question; in interviews, using a hint lowers the rubric's `independent_reasoning` sub-score.
- STAR bank: user-owned stories, retrieved by semantic similarity (Jaccard now, embeddings when plan 3 lands).
- Company packs: seeded config table, top-10 companies at launch, extensible via admin.
- Voice interview: new WebSocket message `interview_start` on the existing voice WS endpoint; the voice session calls the interview generator/evaluator directly.

## Data model
```sql
CREATE TABLE IF NOT EXISTS resume_profiles (
    user_id TEXT PRIMARY KEY,
    skills_json TEXT NOT NULL DEFAULT '[]',
    projects_json TEXT NOT NULL DEFAULT '[]',
    experience_years REAL,
    roles_json TEXT NOT NULL DEFAULT '[]',
    strengths_json TEXT NOT NULL DEFAULT '[]',
    weak_spots_json TEXT NOT NULL DEFAULT '[]',
    raw_text_hash TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS star_stories (
    user_id TEXT NOT NULL, story_id TEXT NOT NULL,
    title TEXT NOT NULL, situation TEXT NOT NULL, task TEXT NOT NULL,
    action TEXT NOT NULL, result TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, story_id)
);
CREATE TABLE IF NOT EXISTS company_packs (
    company TEXT PRIMARY KEY,
    track TEXT NOT NULL,
    style_config_json TEXT NOT NULL,   -- interviewer style, question tendencies, difficulty bias
    updated_at TEXT NOT NULL
);
```
Resume raw text is never stored — only the extracted profile (PII minimization); the hash is used for change detection.

## API changes
- `POST /api/interview-sessions/resume` (multipart PDF/DOCX/TXT) → 201 `ResumeProfile` (extract → LLM structured output).
- `POST /api/interview-sessions/jd-analysis` `{jd_text}` → `{matched[], gaps[], weak_spots[], likely_followups[], recommended_focus_areas[]}`; compares JD requirements against `resume_profiles`.
- `CreateInterviewSessionRequest` extended: `company?: string`, `resume_profile?: bool` (auto-attach), `jd_text?: string` (auto gap analysis → focus areas).
- `POST /api/questions/{question_id}/hint?level=1..4` → `{level, hint}`; level 4 includes the full walkthrough; interview context reduces the rubric sub-score.
- STAR bank CRUD: `POST/GET/PUT/DELETE /api/star-stories[/:id]`; behavioral sessions auto-suggest the top-3 stories as optional context.
- `GET /api/interview-sessions/companies` → available packs; company selector in `InterviewSetup.tsx`.
- Voice: WS message `{type: "interview_start", config: CreateInterviewSessionRequest}` → server creates the session and TTS-reads the first question; audio answers → STT → evaluator → TTS feedback; the existing per-connection turn budget (`_TURN_BUDGET_CLOSE_CODE`) applies.

## Implementation tasks (ordered)
1. `backend/app/services/resume_parser.py`: text extraction (pypdf/mammoth) + LLM structured extraction into the `ResumeProfile` schema; store in `resume_profiles`.
2. `backend/app/services/jd_analyzer.py`: requirement extraction + gap analysis against the profile; output feeds `focus_areas` in `interview_generator.py` prompts.
3. `backend/app/services/hint_service.py`: 4-level hint generation with level-appropriate prompt templates; cost-cap enforcement via `feature_events`; rubric sub-score adjustment in `interview_generator.evaluate_answer`.
4. `backend/app/services/star_store.py` + router; retrieval at behavioral session creation.
5. `company_packs` seed script (`backend/scripts/seed_company_packs.py`) + loader in `interview_generator` question prompts.
6. Voice interview: extend `services/voice_session.py` with interview orchestration (create session via `InterviewStore`, generate question via `InterviewGenerator`, evaluate via the same); extend the WS handler in `routers/voice.py` with the `interview_start` message type.
7. Frontend: resume upload + JD textarea in `InterviewSetup.tsx`; gap-analysis panel; company selector; hint button in `QuizMode.tsx` and `InterviewSessionPage.tsx` with level indicator; STAR bank page (`/star-stories`); voice-interview entry in voice settings.
8. Tests: `test_resume_parser.py`, `test_jd_analyzer.py` (gap precision on a fixture resume+JD), `test_hint_service.py` (level escalation, cost cap, no-answer-leak at levels 1–2), `test_star_store.py`, `test_voice_interview.py` (null STT/TTS providers, E2E).

## Failure modes
- Resume PII → raw text never persisted; the profile is encrypted at rest like API keys (`UserSettingsStore` encryption helper); GDPR delete endpoint `DELETE /api/interview-sessions/resume`.
- LLM extraction failure → 422 with field-level errors; the user can paste text manually (existing fields).
- Hint leakage → level 1–2 prompts explicitly forbid revealing the answer; a test asserts the answer string is absent.
- Voice interview cost → the per-connection turn budget already exists; interview sessions are additionally capped at `turn_count`.
- Company pack missing → fall back to the generic style config (existing behavior).

## Rollout
1. Ship resume/JD/hints/STAR behind `STUDY_ENABLE_INTERVIEW_PLUS_V1` (default `true`).
2. Voice interview behind `STUDY_ENABLE_VOICE_INTERVIEW_V1` (default `false`; enable after the null-provider E2E passes).
3. Seed company packs via script; admins can add more.

## Validation
- Gap analysis: fixture resume + JD → ≥ 80% of planted gaps detected, no invented skills.
- Hints: levels 1–2 never contain the answer substring; level 4 contains a walkthrough.
- Voice interview E2E with NullSTT/NullTTS: session created, question TTS'd, answer evaluated, rubric returned.
- `pytest tests/test_resume_parser.py tests/test_jd_analyzer.py tests/test_hint_service.py tests/test_voice_interview.py`
