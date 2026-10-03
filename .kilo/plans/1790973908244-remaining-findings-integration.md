# Remaining findings & integration tasks

## Context

The LLM pipeline reliability plan is implemented and green: 2494 tests pass, branch coverage 98.41% (gate `fail_under = 90` in `backend/pyproject.toml`), ruff clean, frontend `tsc`/build/vitest green, eval harness 52 cases / 0 gate failures. The working tree holds 75 staged/modified changes plus 41 **untracked** `test_coverage_*.py` files — the entire coverage work is uncommitted.

Four findings were verified against the code (two are real bugs in the plan's own scope; two are dead code):

1. **Quiz fallback does not alternate types** — `question_generator.py:3107` sets `idx = len(out) + guard` and line 3182 unconditionally does `guard += 1`, so `idx` advances by 2 per *accepted* item. Parity never flips, so `idx % 2 == 0` (line 3121) is always true: a mixed fallback quiz emits `[mcq, mcq, true_false, true_false]` pairs. The same root cause skews topic selection (line 3108): with 2 topics, `idx % 2` is always 0, so only the first topic is ever used. Characterized (as current buggy behaviour) in `tests/test_coverage_question_generator_deep.py:1882`.
2. **Non-canonical difficulty empties the fallback batch** — the filler canonicalizes an unknown difficulty to `"medium"` (lines 3099-3101) but validates against the *raw* value (line 3177 passes `difficulty=difficulty`), so every item fails `difficulty_mismatch` (line 3075) and the batch comes back empty. The REST API regex-blocks non-canonical values (`schemas/models.py:433`), so this is reachable only via internal/service-level calls (e.g. the legacy `generate_quiz` wrapper, line 4554), but the service layer is not defensive. Characterized at `tests/test_coverage_question_generator_deep.py:1898`.
3. **`grpc_client.py` is dead and broken** — committed on main, but `GRPCClient.__init__` reads four settings that do not exist in `config.py` (construction raises `AttributeError`), `grpcio` is not a declared dependency, and nothing imports it. Commit `c3e2e60` already removed its test from main once because it breaks *collection of the entire suite* when grpc is not installed; the test is back as an untracked file. Full work is preserved on `wip/agentic-tools`.
4. **`AsyncStoreProtocol` is dead code** — `store_protocol.py` defines `run_async`/`init_tables`/`migrate`, but no store implements `migrate()` and no router imports the protocol; its docstring falsely claims "Routers depend on this protocol".

## Decisions (confirmed with user)

- **grpc_client.py: complete and wire it** — add the four config settings, declare `grpcio`, commit the coverage test. Request-path integration (using `quick_analysis`/`health_check` in a router) stays out of scope: no llm-chain/cli-agent gRPC service exists in the deployment.
- **store_protocol.py: delete the protocol and its test.** The Postgres migration notes in the module docstring survive in git history.
- **CI: consolidate into `backend.yml`** — drop the backend job from `ci.yml` (keep frontend), add lint to `backend.yml`.

## Tasks

### 1. Fix quiz fallback alternation and topic distribution
`backend/app/services/question_generator.py`, `_build_fallback_quiz_items` (lines 3082-3184):
- Add `attempt = 0` next to `guard = 0` (line 3105); delete `idx = len(out) + guard` (line 3107).
- Topic selection (line 3108) and both stem selections (lines 3129, 3145): use `attempt % len(...)` instead of `idx % len(...)`.
- Type selection (line 3121): base alternation on accepted count — `... or len(out) % 2 == 0)`.
- Loop tail (line 3182): add `attempt += 1` beside `guard += 1`.
- Tests: rewrite `test_the_mcq_and_true_false_alternation_never_reaches_the_first_pair` (`test_coverage_question_generator_deep.py:1882-1896`) to assert strict alternation (count=2 → `["mcq", "true_false"]`; count=6 → alternating); add a test that with 2 topics both `topic_id` values appear in the output.

### 2. Fix non-canonical difficulty emptying the batch
- `question_generator.py:3177`: pass the canonicalized value — `difficulty=diff` instead of `difficulty=difficulty`.
- `_validate_quiz_item` (line 3075): normalize the requested value before comparing: `requested = str(difficulty or "").strip().lower()` then `if requested and diff != requested:`. This also hardens the two LLM-path call sites (lines 4344, 4625) against case/whitespace.
- Tests: rewrite `test_a_non_canonical_difficulty_silently_empties_the_batch` (lines 1898-1915) — the filler with `difficulty="impossible"` now serves `count` items, all with `difficulty="medium"`. Keep the standalone assertion at lines 1913-1915 unchanged (it still returns `difficulty_mismatch`: `_quiz_item()` is `"medium"` ≠ `"impossible"`). Add a validator test that requested `" Hard "` matches an item with `"hard"`.

### 3. Complete the gRPC client wiring
- `backend/app/config.py` (after the Document ingestion block, line 136):
  ```python
  # gRPC analysis backends (llm-chain / cli-agent services)
  grpc_llm_chain_host: str = "localhost"
  grpc_llm_chain_port: int = 50051
  grpc_cli_agent_host: str = "localhost"
  grpc_cli_agent_port: int = 50052
  ```
  (`env_prefix = "STUDY_"` yields `STUDY_GRPC_LLM_CHAIN_HOST` etc. automatically.)
- `backend/pyproject.toml`: add `grpcio>=1.60.0` to `[project.dependencies]` (lines 10-27). **Must land in the same commit as the test file** — committing `test_coverage_grpc_client.py` without the dependency reintroduces the suite-collection failure that `c3e2e60` fixed.
- `backend/tests/test_coverage_grpc_client.py`: invert `GRPCClientSettingsContractTests` (lines 87-107) — assert the four fields ARE in `Settings.model_fields` and that `GRPCClient()` constructs successfully with real `Settings()`; update the `_settings_stub` docstring (lines 29-41), which currently claims config declares no grpc settings.
- `git add` the previously untracked test file.

### 4. Delete the dead store protocol
- Delete `backend/app/services/store_protocol.py` and `backend/tests/test_coverage_store_protocol.py` (verified: nothing else imports either; no store implements `migrate()`).
- Migration constraints to carry forward (preserved in git history): async driver replaces `run_async`; Alembic versioned migrations replace `CREATE TABLE IF NOT EXISTS`; preserve composite primary keys; multi-instance deployment additionally needs the Redis rate-limiter backend and a shared database.
- Risk: the `plan/rag-ingestion` worktree originated this module — if that branch still references it, a later merge will conflict. Re-check `git ls-tree plan/rag-ingestion` before merging.

### 5. Consolidate CI
- `.github/workflows/ci.yml`: remove the `backend` job (lines 10-28); keep the `frontend` job.
- `.github/workflows/backend.yml`: add a `lint` job running `python -m ruff check .` (the workflow already defaults `working-directory: backend`).
- `backend/pyproject.toml`: add `ruff>=0.6.0` to `[project.optional-dependencies] dev` (lines 29-34) so `pip install -e ".[dev]"` (backend.yml's install step) provides ruff — it currently lives only in `[dependency-groups] dev`, which plain pip ignores.

### 6. Integration commit
- Run the full validation suite below, then commit the working tree (75 staged/modified changes + 41 untracked test files) on a feature branch.

## Risks

- **Collection-break regression**: the grpc test imports `grpc` and `app.generated` at module level; without `grpcio` declared it aborts collection of the whole suite on a fresh install (see `c3e2e60`). Task 3's dependency and test must ship together.
- **Deterministic-output drift**: the alternation fix changes fallback quiz output. The eval harness uses scripted LLM responses (fallback only triggers on terminal failure), so golden fixtures should be unaffected — confirm via the eval gate in validation.
- **Worktree conflicts**: deleting `store_protocol.py` may conflict with the `plan/rag-ingestion` worktree that created it.

## Validation

```bash
cd backend && PYTHONPATH=. python -m pytest tests/ -q                      # 0 failures
cd backend && PYTHONPATH=. python -m pytest tests/ -q --cov=app            # TOTAL >= 90% (was 98.41%)
cd backend && python -m ruff check app/ tests/ scripts/                    # clean
cd frontend && npx tsc --noEmit && npm test                                # clean
cd backend && python scripts/eval_agent_quality.py --json                  # 0 gate failures, 52 cases
```
Targeted: the two rewritten quiz tests (Task 1/2) and the inverted `GRPCClientSettingsContractTests` (Task 3) must pass; `python -c "from app.config import Settings; Settings()"` must construct with the new fields.

## Out of scope

- Voice-to-voice mock interviews (interview UI is text-only; the voice agent already accepts `session_type: "interview"`).
- Cross-replica state: SQLite single connection + in-memory rate limiter do not share state; multi-instance needs Postgres + Redis (documented constraint, not a bug).
- `sqlite-vec` is not installed in this venv — `document_store.py` vec0 paths are env-gated and unreachable locally.
- GRPCClient request-path wiring (requires deployed llm-chain/cli-agent gRPC services).

## Open questions

None blocking.
