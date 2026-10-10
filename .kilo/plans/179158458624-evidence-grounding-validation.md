# Plan B — Evidence-Based Grounding & Quiz Validation Contract (P0)

**Wave 1 · Parallel with Plan A · No dependencies**

## Goal

Grounding is pure lexical token overlap (one shared token passes), `source_quote` is never verified against topic content, the quiz output contract lacks `learning_objective`/`expected_answer`/`distractor_rationales`, and nothing balances answer positions. Fix all four.

## Current state (verified)

- `question_generator.py:526-548` — `_question_is_grounded_to_anchors` returns `True` on `len(question_tokens & anchor_tokens) >= 1`.
- `question_generator.py:3031-3032` — `_validate_quiz_item` checks `source_quote` is a string ≥8 chars; it is never compared to topic content.
- `question_generator.py:2486-2500` — the quiz JSON contract requests only `question, type, choices, correct_answer, explanation, difficulty, topic_id, source_quote, reasoning_summary, target_level`.
- `question_generator.py:2954-2966` — `_validate_question_item` fills missing `source_quote` with `"Generated from the current topic content."` (invented evidence).
- Grep confirms: no `shuffle`/`position_balance`/`answer_distribution` logic anywhere; `_build_fallback_quiz_items` hardcodes `correct = "A"` (`:3140`, `:3154`).
- Uploaded documents are **not** reachable from the quiz flow (`QuestionGenerator.__init__` takes only `llm_client` + `mcp_gateway`, `:3406-3408`) — evidence is restricted to topic content already in `topics_content`.

## Changes (ordered)

1. **`_validate_quiz_item`** (`question_generator.py:2989-3080`)
   - Add required keys: `learning_objective`, `expected_answer`, `distractor_rationales`.
   - Accept a `topic_content: str` parameter (validator closure at `:4630-4637` passes the resolved content per `topic_id`); reject `source_quote_not_in_content` when the quote is not a substring of the topic content (skip the check for problem-solving topics where content is code-heavy — reuse `is_problem_solving_topic`).
   - Validate `distractor_rationales` covers every non-correct choice label and each rationale is ≥12 chars.
   - Validate `expected_answer` is non-empty and, for MCQ, equals the correct choice text.
2. **Answer-position normalization** — new `_normalise_answer_position(item) -> dict`: after validation, deterministically rotate `choices` so `correct_answer` lands at the position derived from a hash of the normalized question text (distributes A/B/C/D without relying on the model). Apply in `_collect_with_retries` (`:3275` area) and in the quiz assembly loops (`:4350-4360`, `:4668-4686`). Unit-test the distribution over a synthetic batch.
3. **`_question_is_grounded_to_anchors`** (`:526-548`) — require a *meaningful* overlap: shared token of length ≥5, or ≥2 shared tokens, or a shared token with the topic title. Keep deterministic; keep the `problem_solving_mode` parameter but it stays unused (or remove it — pick one, note in PR).
4. **`_validate_question_item`** (`:2927-2986`) — stop inventing `source_quote`: when topic content is available and the item is not problem-solving, require the quote to be a substring of the content (issue `source_quote_not_in_content`); otherwise keep the current default only for the deterministic fallback path (which sets `source_quote` to a real content fragment at `:2903`).
5. **`_build_quiz_prompt`** (`:2427-2543`) — extend the contract block with `learning_objective`, `expected_answer`, `distractor_rationales: [{"label": "B", "rationale": "..."}]`; add rules: "exactly one defensibly correct answer", "distractors must be plausible and technically distinct", "vary the position of the correct answer across items", "source_quote must be a verbatim excerpt of the topic content — never invent a quotation".
6. **`schemas/models.py`** — `QuizQuestionV2` (`:447-457`) gains `learning_objective: str = ""`, `expected_answer: str = ""`, `distractor_rationales: list[QuizDistractorRationale] = []` (new model `{label: str, rationale: str}`). Populate in `_to_quiz_question_v2` (`:3441-3457`) and the batch assembly (`:4668-4686`).
7. **Tests** — update the `_quiz_item()` helper (`test_coverage_question_generator_deep.py:670-689`) and `test_question_generator_validation.py:86` with the new fields; add tests for: substring grounding rejection, invented-quote rejection, distractor-rationale validation, position normalization distribution, legacy item without new fields → `missing_learning_objective`.
8. **Evals** — regenerate `evals/quiz.jsonl` + `evals/golden/quiz.jsonl` + `prompt_snapshots.json` in one deliberate step (`--blame-prompts` to review the prompt diff). `FLOW_CONTRACTS["quiz"].fields` auto-derives from the validator's required keys (`eval_agent_quality.py:326-351`) — verify the derivation picks up the new keys and the `prompt_contract_rate` gate still passes (new field names must literally appear in `_build_quiz_prompt` output).

## Failure modes

- Stricter validation drops more items → `malformed_items_dropped` rises; watch the `schema_valid_rate` gate (must stay 100% after fixture regeneration).
- Substring check is strict about whitespace — normalize whitespace on both sides before comparing.
- Position rotation changes existing golden outputs — expected; regenerate goldens deliberately, never in a commit that also changes prompts.

## Validation

- Backend suite + ruff (as Plan A).
- `python scripts/eval_agent_quality.py` — all deterministic gates green.

## Out of scope

Document-chunk evidence refs (`source_refs` → `(document_id, chunk_index)` as `rag_service.py` does), semantic answer-key checking, difficulty calibration.
