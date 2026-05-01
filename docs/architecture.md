# WorldQuant Alpha Factory Architecture

_Last updated: 2026-04-30_

## Purpose

WorldQuant Alpha Factory is a local orchestration layer for generating, simulating, repairing, ranking, and guarded-submit testing WorldQuant Brain alpha candidates.

The architecture has evolved from a simple backtest worker into a feedback controller: simulation results and submit blockers are converted into the next generation of repair candidates.

## High-level data flow

```text
candidate generators
  -> alphas.csv
  -> alpha_factory.cli import
  -> data/backtests.sqlite3 alpha_tasks
  -> alpha_factory.cli run / Brain simulations
  -> backtest_results
  -> reports + state summaries
  -> strategy_monitor / repair generators / ML scorer
  -> autosubmit safety gate
```

The loop is coordinated by `scripts/review_and_correct.py`, usually launched through `scripts/run_review_correct.sh`.

## Controller stages

The review/correct controller can run these stages depending on current auth, queues, and family metrics:

1. `strategy-monitor` — summarize current family quality, blockers, and strategy posture.
2. `repair-generate` — generate targeted repairs from high-turnover and self-correlation blockers.
3. `repair-import` — import generated candidates into SQLite.
4. `repair-turnover-run` — run `repairto_` candidates.
5. `repair-selfcorr-run` — run `repairsc_` candidates.
6. `repair-submitgated-run` — run `repairsc2_` candidates.
7. `repair-submitgated-v2-run` — run `repairsc3_` structural field/operator variants.
8. `adaptive-sampler` / `adaptive-run` — generate and test `arm_` candidates.
9. `multi-generate` / `multi-run` — generate and test `multi_` cross-dataset candidates.
10. `super-generate` / `super-run` — generate and test `super_` SuperAlpha candidates.
11. `supersc-generate` / `supersc-run` — generate and test `supersc_` SuperAlpha self-correlation repairs.
12. `d1-panel` / D1 validation — inspect low-self-correlation D1-family candidates and their readiness signals.
14. `self-corr-truth-table` — refresh first-class self-correlation evidence from autosubmit logs, cooldown state, submitted records, and backtest rows.
15. `ml-score` — rank recent candidates with a lightweight scorer using truth-table evidence, expression-structure features, lineage priors, D1 metadata, and FDE flags.
16. `autosubmit` — attempt guarded submit only when candidate quality, D1 readiness, daily submit limit, and safety checks pass. Optional exploration/FDE/D1 channels may spend bounded detail-check budget for label collection only; they do not bypass submit safety.

Stages are recorded in `state/review_correct_state.json`, summarized in `reports/review_correct_latest.md`, and logged in `logs/review_correct.log`.

## Candidate families

| Family | Role | Current interpretation |
| --- | --- | --- |
| `decor_` | Primary discovery/decorrelation family | Still the highest controller focus score. |
| `refit_` | Fitness-first refits | Useful but not the main current bottleneck. |
| `repairto_` | High-turnover repairs | Secondary repair lane. |
| `repairsc_` | Self-correlation repairs | Stable pass/strong source, often still submit-blocked. |
| `repairsc2_` | Submit-gated second-order self-correlation repairs | Important source of high raw quality. |
| `repairsc3_` | Phase 3 field/operator self-correlation mutator | Tests field equivalence and operator-family migration. |
| `arm_` | Adaptive single-dataset sampler | Low-volume structural exploration. |
| `multi_` | Cross-dataset interaction candidates | Active but first batches were weak; keep budget small until value density improves. |
| `d1_` | First D1-readiness candidates | Submit-aware low-self-correlation exploration. |
| `d1v2_` | Cleaner D1 anchor variants | Produced better low-self-corr candidates, but still above the submit threshold. |
| `d1v22_` | Whole-anchor proxy variants | Scorer/schema visibility worked, but first validation batch lost too much signal. |
| `d1v23_` | Surgical/helper-target D1 variants | Current follow-up direction to preserve D1v2 signal while reducing self-correlation. |
| `super_` | SuperAlpha blends/structural composites | Strong raw-quality track; cross-lineage constraints are now preferred to avoid same-lineage composites. |
| `supersc_` | SuperAlpha self-correlation repairs | Main repair path for strong `super_` candidates blocked by self-correlation. |

## Self-correlation truth table and exploration lanes

`scripts/self_corr_truth_table.py` makes self-correlation evidence first-class instead of relying only on `backtest_results.fail_reasons`. It combines SQLite backtest rows, autosubmit JSON logs, submission records, cooldown state, retired repair metadata, lineage metadata, and D1/FDE state into:

```text
state/self_corr_truth_table.json
reports/self_corr_truth_table.md
```

Each row records quality metrics, `pass_quality`, `self_corr_status`, evidence strings, skip counts, lineage theme, parent ids, and `repair_depth`. The aggregate summary includes status counts, family/lineage pass-quality counts, known-clear rate, repeated blocked ids, and repair-depth distribution.

The current optimization loop uses three bounded label-collection channels:

- **Standard exploration**: limited detail checks for scorer-marked `exploration_candidate` rows under `AUTO_SUBMIT_MAX_EXPLORATION_P_SELF_CORR`.
- **Forced Diversity Exploration (FDE)**: bounded detail checks for `fde_candidate=true` rows from under-sampled lineages. FDE is controlled by `AUTO_SUBMIT_FDE_ENABLED` and `AUTO_SUBMIT_FDE_CHECKS_PER_ROUND`.
- **D1 relaxed exploration**: bounded detail checks for D1-family candidates with pass-quality and `p_self_corr_block <= AUTO_SUBMIT_D1_MAX_P_SELF_CORR`; results are mirrored into `state/d1_truth_table.json`.

These channels collect authoritative labels only. A candidate that clears an exploration detail check still must pass the normal submit gate, including `AUTO_SUBMIT_MAX_P_SELF_CORR=0.20`, D1-ready when required, unsafe-check clearance, daily submit cap, and auth/detail-budget checks.

Repair generation now tracks `repair_depth` and can retire repeatedly blocked deep repair chains. The generator side also applies lineage quota/diversity pressure so analyst/earnings variants do not monopolize the budget.

## Self-correlation repair design

The durable bottleneck is not raw Sharpe/Fitness alone. Many candidates pass raw thresholds but remain non-submittable due to `SELF_CORRELATION:PENDING`, self-correlation cooldown, or detail-check budget limits.

The repair system therefore emphasizes structural change rather than only parameter tweaks:

- equivalent data-field substitutions: close/open/high/low/vwap/overnight/intraday;
- operator-family migration: rank/zscore/ts_rank/median/decay/residual forms;
- group/bucket residuals and market/liquidity gates;
- helper-orthogonalized SuperAlpha variants;
- second- and third-order repair families (`repairsc2_`, `repairsc3_`, `supersc_`).

The key quality test is whether a variant preserves enough Sharpe/Fitness while clearing self-correlation gates.

## Multi Data Set track

`multi_dataset_generator.py` creates cross-dataset candidates. The interaction space currently includes:

- `rank_spread`
- `zscore_spread`
- `conditional_gate`
- `residual_helper`
- `regime_interaction`
- `neutralized_product`
- `correlation_residual`
- `delta_decay_combo`
- `volatility_scaled`
- `group_rank_contrast`
- `nonlinear_squash`

Current strategy: keep `multi_` exploration small-batch until results show better value density. Weak early results should not be overgeneralized until interaction/function-family coverage is sufficiently broad.

## D1 readiness track

`scripts/d1_generator.py` and `scripts/multi_d1_panel.py` support the D1-family lane: `d1_`, `d1v2_`, `d1v22_`, and `d1v23_`. This lane exists because the effective submit blocker has shifted from raw performance to whether an alpha can clear pre-submit self-correlation and D1-readiness gates.

Current interpretation as of 2026-04-29:

- `d1_` started the explicit D1-readiness exploration.
- `d1v2_` produced the strongest low-self-correlation anchors so far, including candidates with materially improved `p_self_corr_block`, but still not below the conservative `0.20` threshold.
- `d1v22_` tested whole-anchor proxy substitution. The first six-candidate validation batch completed and was visible to the scorer, but Sharpe/Fitness collapsed, so this exact proxy design should not be scaled.
- `d1v23_` is the next surgical/helper-target direction: preserve internal D1v2 signal components while replacing only high-risk correlation targets or helpers.

The D1 lane should be evaluated by `d1_ready`, `p_self_corr_block`, dry-run autosubmit blockers, and retained Sharpe/Fitness together. A low self-correlation score is not enough if signal quality collapses; high raw quality is not enough if D1 readiness stays false.

## SuperAlpha track

`superalpha_builder.py` builds `super_` candidates from strong or self-correlation-blocked parent alphas. The track intentionally trades some lineage simplicity for potentially stronger combined signal.

As of 2026-04-29, examples such as `super_78a04d1302` and `super_053cae76ee` show high raw Sharpe/Fitness, but autosubmit remains blocked by self-correlation pending/cooldown. This makes `supersc_` repair the primary next step, not blind expansion of more near-identical parent blends.

## ML scorer

`scripts/ml_candidate_scorer.py` is currently a lightweight heuristic/logistic-style candidate scorer. It ranks recent candidates by raw quality, structural expression features, family value density, lineage theme, truth-table evidence, predicted self-correlation risk, and D1-family metadata.

The scorer prefix list includes D1-family candidates (`d1v23_`, `d1v22_`, `d1v2_`, `d1_`) alongside `repairfit_`, `multi_`, `super_`, `arm_`, `repairsc2_`, `repairsc3_`, `decor_`, `refit_`, and `supersc_`. Its key submit-awareness outputs are `p_self_corr_block`, `d1_ready`, `exploration_candidate`, `fde_candidate`, `repair_depth`, lineage priors, and expression-structure features such as operator entropy and field diversity.

It is useful for prioritization, but self-correlation risk calibration must be checked against actual autosubmit outcomes. High predicted pass probability does not override `SELF_CORRELATION:PENDING`, D1-readiness failure, daily submit cap, or dry-run autosubmit blockers.

## State, reports, and logs

Important runtime artifacts:

```text
state/review_correct_state.json       # Latest controller state
state/self_corr_truth_table.json      # First-class self-correlation evidence/lineage/repair-depth table
state/d1_truth_table.json             # D1 relaxed exploration detail-check labels
state/auto_submit_state.json          # Autosubmit state, cooldowns, daily submit cap, FDE counters
state/multi_dataset_state.json        # Multi Data Set sampler state
state/superalpha_state.json           # SuperAlpha builder state
state/super_repair_state.json         # supersc_ repair state
state/ml_candidate_scorer_state.json  # ML scorer output, including D1 readiness/risk/FDE fields
state/d1_generator_state.json         # D1/D1v2/D1v22/D1v23 generator metadata
state/multi_d1_panel.json             # Multi/D1 panel state when generated
state/review_correct.lock             # Live controller lock, may become stale after killed runs
reports/review_correct_latest.md      # Human-readable latest controller summary
reports/self_corr_truth_table.md      # Self-correlation truth-table report
reports/backtest_report.md            # General backtest report
logs/review_correct.log               # Controller log
logs/auto_submit.log                  # Autosubmit decisions and blockers
```

## Safety and constraints

- Auth is session-based and can expire while monitors still look healthy. `scripts/auth_recover.py` can refresh from the private local `secrets/worldquant_credentials.json` when configured, then writes `secrets/worldquant.cookie`.
- Never expose or commit `secrets/worldquant.cookie` or `secrets/worldquant_credentials.json`.
- Do not remove `state/review_correct.lock` unless process inspection confirms it is stale.
- Avoid short interactive timeouts for long controller runs; they can kill the process and leave a stale lock.
- Autosubmit is intentionally guarded and should remain conservative.
- Keep `AUTO_SUBMIT_MAX_P_SELF_CORR=0.20`, `AUTO_SUBMIT_REQUIRE_D1_READY=1`, and `AUTO_SUBMIT_DAILY_LIMIT=1` unless the user explicitly approves a policy change after dry-run evidence.
- Exploration, FDE, and D1 relaxed channels are label-collection mechanisms only; they must not submit unsafe candidates.
