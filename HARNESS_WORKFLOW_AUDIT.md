# Harness Workflow Audit

## Scope and Method

This audit used only local source inspection. No WorldQuant endpoints, secrets, auth scripts, CLI run/resume paths, backtests, or database writes were executed.

## Workflow Map

1. Candidate generation writes local candidates to `alphas.csv`.
   - Core CSV append/generation paths include `alpha_factory/generator.py`, `alpha_factory/refit_generator.py`, `alpha_factory/decorrelate_generator.py`, and script families such as `scripts/repair_candidates.py`, `scripts/fresh_supply_generator.py`, `scripts/d1_generator.py`, `scripts/multi_dataset_generator.py`, `scripts/superalpha_builder.py`, and `scripts/adaptive_sampler.py`.
   - `scripts/repair_candidates.py:184` applies the submitted-alpha similarity pre-write filter, then `scripts/repair_candidates.py:733` filters candidates before `append_rows()` at `scripts/repair_candidates.py:737`.
   - `scripts/fresh_supply_generator.py:203` generates fresh low-collision supply; it scores candidates against submitted features at `scripts/fresh_supply_generator.py:239` before writing rows at `scripts/fresh_supply_generator.py:243`.

2. `alphas.csv` is imported into the SQLite queue.
   - Paths are configured in `config.yaml:59`: `input_csv` is `alphas.csv`, `sqlite_db` is `data/backtests.sqlite3`, and report paths are under `reports/`.
   - `alpha_factory/cli.py:47` handles `import`.
   - `alpha_factory/backtester.py:26` parses CSV rows into `AlphaTask` objects.
   - `alpha_factory/database.py:58` upserts tasks into `alpha_tasks`.

3. The SQLite queue drives run/resume.
   - Queue schema lives in `alpha_factory/database.py:12`.
   - `alpha_factory/database.py:81` selects `PENDING`/`RETRY` rows.
   - `alpha_factory/cli.py:52` runs pending tasks, guarded by `Database.active_launch_rows()` at `alpha_factory/database.py:104`.
   - The launch guard itself is `active_backtest_rows()` in `alpha_factory/sqlite_utils.py:28`; `RUNNING`/`SUBMITTING` always block, and recent or simulation-backed `RETRY` rows also block.
   - `BacktestWorker.run_pending()` is at `alpha_factory/backtester.py:75`; `submit_pending()` is at `alpha_factory/backtester.py:86`; `run_one()` is at `alpha_factory/backtester.py:133`.
   - `alpha_factory/cli.py:61` handles `resume`, calling `BacktestWorker.resume_running()` at `alpha_factory/backtester.py:112`.

4. Results and reports are saved locally.
   - Terminal payloads are normalized by `parse_result()` at `alpha_factory/backtester.py:204`.
   - Results are persisted by `Database.save_result()` at `alpha_factory/database.py:133`.
   - `alpha_factory/reporter.py:17` exports CSV and Markdown reports; Markdown treats only `COMPLETE` as completed at `alpha_factory/reporter.py:47`.

5. Scorer and autosubmit consume results.
   - `scripts/submitted_similarity.py:158` builds submitted-reference features, and `score_against_submitted()` at `scripts/submitted_similarity.py:234` computes nearest-submitted collision scores.
   - `scripts/ml_candidate_scorer.py:318` scores candidates using quality, family stats, structural novelty, truth-table evidence, submitted similarity, and predicted self-correlation risk. Submitted-similarity penalties are applied around `scripts/ml_candidate_scorer.py:399`.
   - The scorer writes `state/ml_candidate_scorer_state.json` and `reports/ml_candidate_scorer_report.md` at `scripts/ml_candidate_scorer.py:636`.
   - `scripts/auto_submit.py:234` selects result rows, `passes()` at `scripts/auto_submit.py:247` enforces quality thresholds, and the main submit gate starts at `scripts/auto_submit.py:257`.
   - Autosubmit blocks high submitted-reference collisions at `scripts/auto_submit.py:308`, predicted self-correlation risk at `scripts/auto_submit.py:312`, performs alpha detail checks at `scripts/auto_submit.py:401`, and only calls `client.submit_alpha()` at `scripts/auto_submit.py:436`.

## WARNING Terminal-Status Bug

WorldQuant can return simulation status `WARNING`. Older terminal handling only recognized `COMPLETE`, `DONE`, `ERROR`, `FAILED`, and `FAIL`. That caused a `WARNING` simulation to remain locally `RUNNING`; once persisted, the active launch guard saw the row as active and blocked further runs indefinitely.

Current source is partially corrected:

- `resume_running()` now treats `WARNING` and `WARN` as terminal at `alpha_factory/backtester.py:120`.
- `parse_result()` maps `WARNING`/`WARN` to local `ERROR` at `alpha_factory/backtester.py:217`, which clears the active guard once resume saves the result and marks the task.

Remaining gap:

- `BrainClient.wait_for_simulation()` still uses the older terminal set at `alpha_factory/brain_client.py:161`. The synchronous `run_one()` path can still wait forever when `max_poll_seconds` is unset, or until timeout if configured, when WQ returns `WARNING`. Terminal-status logic should be centralized and reused by both `wait_for_simulation()` and `resume_running()`.

## Fragile Points

- Active guard: `active_backtest_rows()` in `alpha_factory/sqlite_utils.py:28` is correctly conservative, but it depends on terminal statuses being recognized everywhere. `scripts/strategy_monitor.py:52` resets stale `RUNNING` rows older than four hours, which is useful but also masks terminal-state bugs after a long delay.
- Auth recovery: `scripts/auth_recover.py:77` probes auth, `scripts/auth_recover.py:109` applies cooldowns, and `scripts/auth_recover.py:194` can use JSON credentials. `scripts/healthcheck.py:161` performs capability probes and `scripts/healthcheck.py:146` can invoke recovery. This is operationally helpful but has broad side effects, so controller paths should keep auth probes separated from pure local planning.
- Network/DNS errors: `BrainClient._request()` raises for request exceptions and HTTP errors. `BacktestWorker.run_one()` catches broad exceptions and retries at `alpha_factory/backtester.py:133`, while `submit_pending()` also turns any exception into `RETRY` at `alpha_factory/backtester.py:86`. DNS/connectivity failures can therefore create recent `RETRY` rows that block launches through the active guard for 45 minutes.
- Submitted-similarity pre-write filter: `scripts/repair_candidates.py:184` and `scripts/fresh_supply_generator.py:239` block obvious near-submitted candidates before `alphas.csv`, while `scripts/ml_candidate_scorer.py:399` and `scripts/auto_submit.py:308` add later-stage penalties. This layered design is good, but thresholds differ (`repair` default 0.65, fresh-supply 0.58, autosubmit 0.85), so reports should keep showing which threshold rejected each candidate.
- Family watchdog: `scripts/review_and_correct.py:375` computes pause/downweight state from collision rate, predicted self-correlation, D1 readiness, pending self-correlation, and stale pre-write generation. This protects the queue, but `previous_watchdog` is sticky at `scripts/review_and_correct.py:421`; recovery/decay criteria should be explicit so a family is not suppressed after conditions improve.
- `fresh_supply` lane: `scripts/review_and_correct.py:862` always schedules fresh-supply generation/import and can schedule a `fresh_` run at `scripts/review_and_correct.py:865`. The lane is valuable as low-collision supply, but it still appends to shared `alphas.csv`; duplicate/id handling and submitted-similarity state freshness are key.
- Git hygiene: README explicitly says script changes should be mirrored into the repo. Current workflow stores many generated artifacts under `state/`, `data/`, and `reports/`; keep source, config, tests, and docs under version control, but avoid committing runtime DBs, logs, credentials, cookies, and generated state unless intentionally snapshotting.

## Prioritized Action Plan

### Immediate safety/fix items

1. Centralize terminal simulation statuses in one helper/constant and include `WARNING`/`WARN`. Use it in `BrainClient.wait_for_simulation()` and `BacktestWorker.resume_running()`.
2. Ensure `run_one()` cannot wait forever on any unknown/non-progressing terminal-like state: configure `max_poll_seconds` and persist timeout/error results consistently.
3. Add an explicit local cleanup path for known-stale `RUNNING` rows whose remote status is already terminal, rather than relying only on the four-hour reset in `strategy_monitor`.
4. Make auth/capability probes opt-in for planning/report-only commands; local reporting should not require live auth.

### Medium cleanup/refactor

1. Extract shared status normalization from `parse_result()` so reports, active guards, resume, and run paths agree on status meaning.
2. Normalize submitted-similarity thresholds and reason names across repair, fresh-supply, scorer, and autosubmit reporting.
3. Add decay/recovery logic to family watchdog state in `scripts/review_and_correct.py` so paused families can recover after fresh evidence.
4. Separate generator append logic from shared `alphas.csv` mutation where practical, or add clearer per-lane import prefixes and reports.

### Tests/smoke checks

1. Unit-test that `WARNING` and `WARN` are terminal in both resume and wait paths, and that they save a non-active result status.
2. Unit-test `active_backtest_rows()` for `RUNNING`, `SUBMITTING`, recent `RETRY`, old `RETRY`, and terminal statuses.
3. Unit-test submitted-similarity filter decisions for low/weak/medium/high collision boundaries.
4. Smoke-test report generation against a tiny local SQLite fixture without importing `BrainClient` or touching auth.

### Documentation/git hygiene

1. Document terminal status policy in `docs/runbook.md` and the workflow map in `docs/architecture.md`.
2. Document which commands are local-only versus WQ-touching.
3. Keep credentials, cookies, DBs, logs, and generated runtime state out of normal commits.
4. Add a short maintenance checklist: run local tests, inspect `git diff`, verify no secrets/runtime artifacts, then commit source/docs/tests together.
