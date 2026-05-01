# WorldQuant Self-Correlation Strategy Refactor Plan

## Background
The WorldQuant alpha factory has recovered auth and is running cycles, but repeated "optimization" has not produced durable improvement against self-correlation. The current pipeline has accumulated multiple generators (`repairsc_`, `repairsc2_`, `repairsc3_`, `super_`, `supersc_`, `multi_`, `d1*`, adaptive sampler, ML scorer), but most apparent improvements are pass-quality improvements (Sharpe/Fitness/Turnover), not proven submit/self-correlation-clear improvements.

Recent evidence from local data/logs:
- `backtest_results` has ~833 rows.
- `repairsc_`: 36 rows, 18 pass-quality.
- `repairsc2_`: 88 rows, 16 pass-quality.
- `repairsc3_`: 19 rows, 6 pass-quality.
- `supersc_`: 10 rows, 4 pass-quality.
- `super_`: 6 rows, 3 pass-quality.
- `multi_`: 12 rows, 0 pass-quality.
- Recent autosubmit skip reasons include roughly:
  - `pre-submit self-corr gate`: 3050
  - `self-correlation cooldown active`: 1027
  - `unsafe checks: SELF_CORRELATION:PENDING`: 51
- Important: `backtest_results.fail_reasons` usually does not contain SELF_CORRELATION for these families. The true self-correlation signal mostly lives in autosubmit/detail-check logs and cooldown state, not in the main result table.
- Recent expressions are concentrated in analyst/earnings lineage (`earnings_momentum_analyst_score`, `change_in_eps_surprise`, `earnings_torpedo_indicator`, `analyst_revision_rank_derivative`, etc.). Function diversity exists, but economic/dataset lineage diversity is weak.

## Current Failure Mode
The current pipeline is stuck in a loop:
1. Generate many formula variants around the same effective analyst/earnings signals.
2. Many pass Sharpe/Fitness/Turnover.
3. ML scorer predicts high self-correlation risk.
4. Autosubmit pre-gate skips them before real WQ detail checks.
5. Few new authoritative self-correlation labels are collected.
6. More generators are added, but they optimize pass-quality rather than true self-correlation clear rate.

## Objective
Refactor the strategy from "add more generators" to a measured self-correlation optimization loop:
- Treat self-correlation outcome as first-class data.
- Build a truth table from autosubmit/detail logs, cooldowns, submissions, and backtest rows.
- Add lineage/economic-theme classification instead of relying on prefix family names.
- Use a small exploration budget to collect real labels even when ML predicts high risk.
- Shift budget away from pass-quality-only strategies toward submit-clear/self-corr-clear evidence.
- Remove or bypass obsolete code paths that depend on `backtest_results.fail_reasons like '%SELF_CORRELATION%'` as the primary source of truth.

## Required Implementation

### 1. Add a self-correlation truth table module/script
Create `scripts/self_corr_truth_table.py`.

It should:
- Read `data/backtests.sqlite3` (`backtest_results`, `alpha_tasks`).
- Parse `logs/auto_submit.log` JSON lines.
- Read `data/auto_submissions.jsonl` if present.
- Read `state/auto_submit_state.json` cooldowns if present.
- Emit `state/self_corr_truth_table.json` and `reports/self_corr_truth_table.md`.

For each local alpha id, include:
- `alpha_id`
- `family` prefix
- `expression`
- `sharpe`, `fitness`, `turnover`
- `pass_quality` boolean using existing thresholds 1.6/1.0/0.45
- `worldquant_alpha_id` if extractable from `raw_json`
- `self_corr_status`: one of `clear`, `pending`, `blocked`, `predicted_blocked`, `cooldown`, `unknown`
- `evidence`: list of concise strings, e.g. `auto_submit: unsafe checks SELF_CORRELATION:PENDING`, `auto_submit: pre-submit gate p=0.305`, `cooldown_state`, `submitted`, `already_submitted`
- `last_seen_at`
- `skip_counts` by reason category
- `lineage`: a dict with at least:
  - `theme`: coarse theme such as `analyst_earnings`, `price_volume`, `liquidity_volatility`, `fundamental_valuation`, `market_size`, `mixed`, `unknown`
  - `datasets_or_fields`: extracted important field tokens
  - `uses_returns_or_price_corr`: boolean
  - `parent_ids` when available from `superalpha_state.json` / `d1_generator_state.json` / `multi_dataset_state.json`

Also include aggregate summary:
- counts by `self_corr_status`
- counts/pass_quality by family
- counts/pass_quality by lineage theme
- submit-clear or known-clear rate where evidence exists
- top repeated blocked local ids

### 2. Modify `scripts/ml_candidate_scorer.py`
Use the truth table if available.

Required behavior:
- Load `state/self_corr_truth_table.json`.
- Use authoritative/persistent statuses (`blocked`, `pending`, `cooldown`, `clear`) to adjust risk more directly than current heuristic-only log parsing.
- Add lineage concentration penalty by theme, especially when recent pass-quality candidates are dominated by one theme.
- Do not blindly mark every high-quality but high-risk candidate as non-actionable. Add fields:
  - `exploration_candidate`: boolean
  - `exploration_reason`
- Exploration should prefer candidates that are either:
  - quality pass and lineage theme under-sampled, or
  - quality pass with predicted risk high but structurally novel / cross-theme.
- Keep existing output schema compatible, but add the new fields.

### 3. Modify `scripts/auto_submit.py`
Add small controlled exploration budget.

Required behavior:
- Keep existing safety rules: no submit if WQ detail shows SELF_CORRELATION fail/pending or other unsafe checks.
- Keep normal pre-submit gate for most candidates.
- But allow a limited number of detail checks per run for candidates with `exploration_candidate=true`, even if `p_self_corr_block > AUTO_SUBMIT_MAX_P_SELF_CORR`.
- Add env knobs with defaults:
  - `AUTO_SUBMIT_SELF_CORR_EXPLORATION_CHECKS=2`
  - `AUTO_SUBMIT_MAX_EXPLORATION_P_SELF_CORR=0.75`
- Exploration means only detail-checking to collect authoritative labels. It must not submit unsafe candidates.
- Log exploration decisions distinctly, e.g. `exploration detail check despite p_self_corr=... reason=...`.

### 4. Modify generator selection logic instead of adding another generator
Prefer minimal changes:
- In `scripts/repair_candidates.py`, stop using `fail_reasons like '%SELF_CORRELATION%'` as a primary source. Use truth table blocked/cooldown/pending sources instead.
- In `scripts/superalpha_builder.py`, penalize pairing candidates with the same lineage theme even if their prefix families differ.
- In `scripts/multi_dataset_generator.py` or adaptive sampler, do not expand `multi_` blindly if recent `multi_` has 0 pass-quality; keep it but cap/slow it unless lineage theme is under-sampled.

### 5. Integrate into main review/correct flow
Find `scripts/review_and_correct.py` / `scripts/run_review_correct.sh` orchestration.
- Run `self_corr_truth_table.py` before `ml_candidate_scorer.py` and before autosubmit.
- Ensure reports mention truth table summary.

### 6. Delete or retire obsolete content safely
Do NOT delete:
- `data/`, `logs/`, `state/`, `secrets/`, `.venv/`, `reports/` historical files.
- Cookie or auth files.

Allowed cleanup:
- Remove dead helper functions only if replaced and unused.
- Remove report text that claims strategy activation without evidence, if generated by static code and safe to update.
- Prefer deprecating over deleting if unsure.

### 7. Tests / validation
There is no existing test suite. Add lightweight tests under `tests/` using temp files/sqlite where practical.
Minimum tests:
- truth table parser correctly classifies:
  - pre-submit gate as `predicted_blocked`
  - unsafe SELF_CORRELATION:PENDING as `pending`
  - cooldown state as `cooldown`
  - submitted/already_submitted as `clear`
- lineage classifier detects analyst/earnings vs price/volume vs liquidity/volatility.
- auto_submit exploration logic can allow detail-check budget for exploration candidates without bypassing unsafe check.

Run:
- `./.venv/bin/python -m py_compile scripts/self_corr_truth_table.py scripts/ml_candidate_scorer.py scripts/auto_submit.py scripts/repair_candidates.py scripts/superalpha_builder.py scripts/review_and_correct.py`
- any added tests with `./.venv/bin/python -m pytest` if pytest exists, otherwise run simple unittest scripts.
- Dry run/report generation: `./.venv/bin/python scripts/self_corr_truth_table.py` and inspect generated summary.

## Expected Outcome
Not immediate guaranteed submissions. Expected improvement is a better learning loop:
- Fewer repeated variants from the same analyst/earnings lineage.
- More authoritative self-correlation labels collected per cycle.
- Better distinction between pass-quality and submit-clear quality.
- Reduced proliferation of unvalidated generator layers.
- Budget allocation begins to favor lineage themes and methods with actual self-correlation evidence.

## Important Constraints
- Do not run destructive shell commands.
- Do not delete data/logs/state/secrets.
- Do not submit anything manually outside existing `auto_submit.py` safety logic.
- Preserve existing auth/cookie behavior.
- Make minimal, reviewable code changes.
