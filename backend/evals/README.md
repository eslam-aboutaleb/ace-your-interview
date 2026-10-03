# Eval fixtures and golden outputs

Stage 4.1 of `.kilo/plans/1790987112911-*` / `1790972917415-llm-pipeline-reliability-plan.md`.

Two tiers live here:

| Tier | Entry point | Network | Cost | When |
| --- | --- | --- | --- | --- |
| deterministic | `python scripts/eval_agent_quality.py` | none | none | every push (CI) |
| live | `python scripts/eval_agent_quality.py --live` | provider calls | real tokens | `workflow_dispatch` / weekly cron |

The deterministic tier is what CI gates on. It imports the production parsers
and validators, so a number it reports is a number production would agree with.

## Fixture files

| File | Flow | What it covers |
| --- | --- | --- |
| `questions.jsonl` | question generation | conceptual + problem-solving, every language in `PROBLEM_SOLVING_LANGUAGE_OPTIONS`, every level, `_attempt_budget` fan-out, prior-progress uniqueness, hostile `mcp_context`, the unsupported-language degraded path |
| `quiz.jsonl` | quiz generation | single/multi topic, mcq-only and true-false-only, problem-solving topic, large `count`, hostile topic content |
| `interview_question.jsonl` | interview question generation | `coding` / `behavioral` / `mixed` x `junior` / `mid` / `senior`, opening/middle/late turn phases, hostile job description + resume, a duplicate question production must reject |
| `interview.jsonl` | interview evaluation | the same 3x3 type/level matrix, notes that need Stage 1.4 repair, two Stage 1.3 degraded outputs, a hostile answer |
| `chat.jsonl` | chat follow-up | normal markdown reply, a reply to a fence-breakout attempt |
| `custom_topics.jsonl` | roadmap generation | opening/middle batches, hostile topic name, a 100-section aggregate |

### Matrix coverage is gated

`matrix_coverage` in the report checks that the fixtures actually cover every
`junior` / `mid` / `senior` level, every `coding` / `behavioral` / `mixed`
interview type, and every language in
`PROBLEM_SOLVING_LANGUAGE_OPTIONS`. Stage 1.10 adds a language → the gate fails
until a fixture covers it, so the matrix cannot silently rot.

### Case shape

```json
{
  "case_id": "questions.problem_solving.go.senior.count3",
  "flow": "questions",
  "tags": ["problem_solving", "go", "senior", "fence_language"],
  "input": { "...flow-specific request parameters..." },
  "output": "...raw model text, exactly as the provider returned it...",
  "expected": { "valid": true, "item_count": 3 }
}
```

* `input` is what makes a prompt regression detectable. The harness feeds it to
  the production prompt builder and checks the rendered prompt, so a fixture
  without one is rejected outright — the old fixtures had no input at all and
  could not fail for a prompt reason.
* `output` is raw text, not a pre-parsed object, so the salvage path
  (`parse_json_object`, `_parse_questions_json`) is exercised the way a real
  provider response exercises it.
* `expected.valid` declares the intended outcome. `false` marks a case that
  production *must* reject; the harness fails if such a case starts validating.
  This is how the unsupported-language problem-solving path and the degraded
  interview evaluations are asserted.
* `expected.item_count` is optional. When present the harness enforces it.
* **Never** `retries_used`. The loader raises on it. Retry counts come from the
  run store (below), not from a value typed into a fixture.

The `questions` / `quiz` / `custom_topics` outputs in this directory were
produced by the production deterministic fallbacks
(`_build_fallback_question_items`, `_build_fallback_quiz_items`) and are then
re-checked through `_validate_question_item` / `_validate_quiz_item`, so every
golden item is one the shipping validators accept.

## `prompt_snapshots.json`

Golden `sha256` of every input prompt the harness rebuilds from a fixture.
A prompt edit changes the digest and fails the deterministic gate — that is the
"a prompt edit that breaks output shape fails the build" behaviour. After an
*intentional* prompt change:

```bash
python scripts/eval_agent_quality.py --blame-prompts
```

and review the diff. The snapshot never suppresses the two derived gates
(`prompt_contract_rate`, `prompt_output_field_rate`), so re-baselining cannot
hide a contract that production no longer describes.

## `golden/`

Checked-in reference outputs the live tier diffs against. `--live` replays the
fixture inputs through the real generators and reports added / removed /
changed cases plus a unified diff per change. Regenerate deliberately, never as
part of a commit that also changes prompts.

## `runs/`

`llm_calls.jsonl` — **one row per real `LLMClient.completion` call**, carrying the
`usage` / `finish_reason` / `error_code` metadata that `completion` already
returns:

```json
{"run_id":"live-…","group_id":"questions-ps-java-mid","sequence":1,"flow":"questions","provider":"groq","model":"…","finish_reason":"stop","error_code":"","usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}
```

Attempts are derived by grouping rows on `group_id`, so a file cannot simply
declare that it used three retries. The file is **empty**: no instrumented run
against a real provider has been recorded here yet, and the retry gates report
`unverified` rather than a pass they have not earned. Populate it with
`--live`.

`llm_calls.synthetic.jsonl` — a small mechanism check for the grouping
arithmetic. It is counted but never trusted: while only synthetic records
exist, `retry_distribution.verified` is `false` and the retry gates stay in
`unverified_gates`.
