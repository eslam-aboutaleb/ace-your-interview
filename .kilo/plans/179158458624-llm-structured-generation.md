# Plan E — Reusable LLM Layer: Latency Telemetry, Schema Enforcement, Prompt Registry, Run Store (P1)

**Wave 3 · Depends on Plan C (both edit `question_generator.py` call sites)**

## Goal

The LLM client reports usage/provider/model/finish-reason but not latency or retry outcomes; structured output is a boolean `json_object` flag with no schema enforcement; prompts are inline unversioned constants duplicated across five generators; and generation quality counters are computed per-call and discarded. Make the layer genuinely reusable without introducing premature abstraction.

## Current state (verified)

- `llm_client.py:266-279` — single generic `completion()`; no `time.monotonic()` around `acompletion` (latency unmeasured); retries live in caller loops and are never reported.
- `llm_client.py:48-50, 498-499` — `response_format = {"type": "json_object"}` only, gated by `JSON_MODE_CAPABLE_PROVIDERS`; no JSON schema is ever sent.
- `_supported_options` is byte-identically duplicated at `question_generator.py:82`, `custom_topic_generator.py:151`, `problem_solving_generator.py:49`, `topic_language_advisor.py:121` (signature introspection, not capability detection).
- Prompts are inline constants: `_QUIZ_SYSTEM_PROMPT` (`question_generator.py:2381`), `_question_system_prompt` (`:2111`), `_CARD_SYSTEM_PROMPT` (`:4791`), interview `_question_prompt`/`_evaluate_prompt` (`interview_generator.py:383`, `:517`), plus `custom_topic_generator.py:36`, `problem_solving_generator.py:42`, `topic_language_advisor.py:28`, `rag_service.py:186`. No `prompt_version`/`generation_version` field exists anywhere.
- Quality counters (`malformed_items`, `retries_used`, `issue_counter`) are local variables in `_collect_with_retries` (`:3210-3399`) and never persisted.

## Changes (ordered)

1. **`llm_client.py`**
   - Measure per-call latency (`time.monotonic()` around `acompletion`) and add `latency_ms` to `base_metadata` (`:441-446`).
   - Add an optional `response_json_schema: dict | None = None` parameter: when the provider supports it (extend the capability sets — OpenAI `response_format: {"type": "json_schema", json_schema: {...}}`, Gemini `response_schema`), send it; otherwise fall back to `json_object` mode, then to the existing salvage parser. Anthropic stays on the salvage path (documented asymmetry, `:498`).
   - Add `attempts` and `terminal_reason` to the result metadata so callers no longer lose retry outcomes (callers still own the retry loop — do not add retries inside the client).
2. **Deduplicate `_supported_options`** into one shared helper in `llm_client.py`; delete the four copies.
3. **New `backend/app/services/prompt_registry.py`** — `PromptTemplate` dataclass (`name`, `version`, `system_prompt`, `build_user_prompt(**ctx)`) and a `PromptRegistry` with `register`/`get(name, version)`. Migrate the six prompt constants above into named, versioned templates (start with the quiz and interview flows — the two with eval gates). Record `prompt_name` + `prompt_version` in generation metadata and in the `stats` dict returned by `_collect_with_retries`.
4. **New generation run store** — table `generation_runs` in `learning_store.py` (or a small dedicated store): `run_id, user_id, flow, provider, model, prompt_name, prompt_version, attempts, terminal_reason, input_tokens, output_tokens, finish_reason, error_code, created_at`. Write one row per `_collect_with_retries` / `evaluate_answer` completion. No UI in this plan (analytics dashboard is out of scope).
5. **Thread prompt version into responses** — add `generation_version: str` (format `"<prompt_name>@<version>"`) to `GenerateQuestionsV2Response`, `GenerateQuizV2Response`, and the interview turn payload, so quality metrics can be broken down by prompt version.
6. **Evals** — `prompt_snapshots.json` regeneration via `--blame-prompts`; the harness's `prompt_digest_match_rate` gate must stay green. Optionally have the live tier write `evals/runs/llm_calls.jsonl` rows (currently empty — retry gates report `unverified`).

## Failure modes

- Any prompt text change flips the deterministic prompt-digest gate — expected; rebaseline deliberately with `--blame-prompts`, never suppress the derived gates.
- Schema enforcement is provider-dependent — every path must degrade to `json_object` → salvage parser without failing the request.
- Keep `completion()` as the single entry point; do not add a `StructuredGenerator` class hierarchy (the review's Protocol sketch is conceptual — an optional parameter is the right-sized abstraction here).

## Validation

- Backend suite + ruff; deterministic eval gates all green.
- Unit tests: latency present in metadata; schema passed to a fake provider that supports it and ignored by one that doesn't; prompt registry returns the registered version; run store row written per generation.

## Out of scope

Cost/latency/capability-based model routing (keep the 4-key task map at `llm_client.py:42`), LLM-as-judge `QualityEvaluator`, semantic answer-key validation.
