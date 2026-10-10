# Plan H — Voice Interview & JD-Driven Preparation (P1-product)

**Wave 4 · Depends on Plan A (interview contract) and Plan D (FSRS / planner reads)**

## Goal

Voice interviews exist but have no barge-in/interruption handling and hardcode `response_time_ms=0`; JD analysis produces gaps that never reach the study plan. Add response-time measurement, client-side barge-in using the already-declared (unused) VAD dependencies, and a JD→preparation-plan path.

## Current state (verified)

- `voice_session.py:341` — `record_turn(..., response_time_ms=0)` hardcoded; the DB column (`interview_store.py:91`) and REST contract (`body.response_time_ms`) already exist.
- `routers/voice.py:438-440, 454-462` — the interview path collapses all latency into one `answer_ms`; the chat path reports the full `stt_ms/llm_ms/tts_ms` breakdown (`:520-530`).
- `voice_providers.py:94-103` — `TTSProvider.synthesize_stream` exists but is dead code (only `synthesize` is called); no provider supports streaming STT or interruption.
- `frontend/package.json:13,18` — `@ricky0123/vad-web` and `onnxruntime-web` are declared with **zero imports in `frontend/src`** — pre-wired for barge-in.
- `voiceService.ts:166, 192-195` — `speechSynthesis.cancel()` / `stopPlayback()` are the only stop mechanisms.
- `jd_analyzer.py:139-152` — `analyze` returns `{matched, gaps, weak_spots, likely_followups, recommended_focus_areas}` with a real safety invariant (every reported skill must literally appear in the JD, `:104-120`); gaps flow into interview `focus_areas` via `personalize_session_inputs` (`interview_sessions.py:206-245`) but **never into the study plan** (`learning_planner.py:842-915` takes only `user_id/days/daily_items`).

## Changes (ordered)

1. **Response-time measurement** — `useVoice.ts` records mic-start→mic-stop duration per answer; `WebSocketVoiceService.sendAudio` gains an optional `response_time_ms` field; `routers/voice.py` audio handler (`:414-471`) passes it to `VoiceInterviewSession.answer()`; `voice_session.py:316-388` accepts and stores it in `record_turn` (replacing the hardcoded `0` at `:341`). Also report the per-stage latency breakdown for interview turns (mirror `:520-530`).
2. **Barge-in (client-side)** — wire `@ricky0123/vad-web` in `useVoice.ts`: while TTS is playing, VAD-detected speech triggers `stopPlayback()`/`speechSynthesis.cancel()` and immediately sends the recorded audio turn. Graceful degradation when VAD is unsupported (feature-detect; push-to-talk remains the fallback). Server-side: add an `interrupt` WS message that cancels the in-flight turn task (wrap STT/LLM/TTS in cancellable `asyncio` tasks in `process_audio_turn`, `voice_session.py:80-153`).
3. **Streaming TTS** — wire the existing `synthesize_stream` (`voice_providers.py:99-103`) for chunked playback so interruption can cut mid-utterance; keep `synthesize` as the fallback.
4. **JD → preparation plan** — new `POST /api/learning/plan-from-jd` (or extend the jd-analysis response): map `gaps`/`recommended_focus_areas` to topic IDs via `DocParser.list_topics` matching (the same prefix-matching mechanism as `_track_topic_id`, `interview_sessions.py:164-168`), persist high-confidence matches as `focus_topic_ids` on the learner profile (`learning_planner.py:103-120`), and return `build_study_plan` output. Unmapped gaps stay as plain focus-area strings. Frontend `InterviewSetup.tsx`: "Build prep plan from JD" button next to the existing "Use N recommended focus areas" (`:737-746`), navigating to `/study-plan`.
5. **Tests** — response-time round-trip through `record_turn`; `interrupt` message cancels an in-flight turn; VAD feature-detection fallback; JD→topic mapping (high-confidence match, unmapped gap, no-JD case); plan-from-jd endpoint contract.

## Failure modes

- VAD browser support varies — barge-in must be optional and never block the interview flow.
- JD→topic mapping is fuzzy — only map high-confidence matches; never invent a topic ID.
- `analyze_job_description` (`interview_sessions.py:541-544`) does not pass `llm_config`/`user_identity` unlike the session-creation paths — align it while touching this area.
- Voice WS turn budget (`voice.py:151-178`, close code 4429) must not consume budget on interrupted turns that never reached the LLM — verify.

## Validation

- Backend suite + ruff (voice tests: `test_voice_interview.py`, `test_voice_ws_auth.py`, `test_coverage_routers_voice.py`, `test_jd_analyzer.py`).
- Frontend build + tests; manual: cloud-tier voice interview with barge-in, timed answer recorded in the turn, JD plan appears in `/study-plan`.

## Out of scope

Server-side VAD, streaming/partial STT transcripts, response-time *analysis* (scoring speed against correctness — only measurement + storage here), real-time collaborative interviews.
