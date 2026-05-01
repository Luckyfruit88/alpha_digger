# alpha_digger

A WorldQuant alpha research workflow focused on generating, filtering, backtesting, and supervising candidate alphas with strong safety gates around self-correlation and submitted-lineage collisions.

## What this repo contains

- Candidate generation pipelines for repair, multi-dataset, D1, SuperAlpha, and related branches
- Submitted-alpha reference library fetch + similarity scoring
- Family-level supervision to downweight or pause unproductive lineages
- Autosubmit guardrails with strict self-correlation and D1 readiness checks
- Reports and state artifacts for monitoring each cycle

## Core workflow

1. **Auth check / recovery**
   - Validate WorldQuant auth state
   - If cookies expire, recover via local JSON credentials when configured, then refresh the cookie file

2. **Candidate generation**
   - Generate new candidates from multiple families:
     - `repairto_`, `repairsc_`, `repairsc2_`, `repairsc3_`
     - `multi_`
     - `d1*`
     - `super_`, `supersc_`
     - `arm_`, `refit_`, `decor_`, others

3. **Upstream filtering before write**
   - Lineage quota limits for overcrowded themes
   - Submitted-alpha similarity filtering to block obvious collisions before candidates enter `alphas.csv`

4. **Import and backtest**
   - Import candidates into the local task DB
   - Run staged backtests for turnover, self-correlation, submit-gated screening, and related panels

5. **Scoring and supervision**
   - Score candidates with ML-style heuristics
   - Estimate `p_self_corr_block`, D1 readiness, submitted-collision level, and exploration value
   - Apply family-level watchdog logic to downweight or pause branches that are repeatedly unproductive

6. **Autosubmit gating**
   - Require quality thresholds
   - Require low enough self-correlation risk
   - Require D1 readiness where configured
   - Respect cooldowns, detail-check budgets, and daily submit limits

## Why high self-correlation can still appear

Submitted-alpha filtering does **not** eliminate every high-self-correlation alpha. It only removes a major subset of near-submitted or structurally crowded candidates. Some high-self-correlation outcomes are still objectively unavoidable because different expressions can map to similar economic signals.

The workflow therefore uses **two layers**:

- **Submitted filter**: blocks obvious near-submitted collisions early
- **Supervision/watchdog**: detects branches that keep producing low-novelty or high-self-correlation candidates and downweights/pauses them

## New supervision logic in this repo

The current version includes a family-level watchdog that looks for patterns such as:

- repeated pre-write filtering (`generated=0`, fully filtered before `alphas.csv`)
- high submitted-collision concentration
- high predicted self-correlation concentration
- near-zero D1-ready rate
- persistent pending self-correlation backlog

When a family crosses thresholds, it is automatically **downweighted** or **paused** in focus selection so the workflow shifts toward better supply lines like `multi_` or other less crowded branches.

## Important files

- `scripts/review_and_correct.py` — main cycle planner / supervisor
- `scripts/repair_candidates.py` — repair candidate generation and upstream filtering
- `scripts/ml_candidate_scorer.py` — candidate scoring and self-correlation risk estimation
- `scripts/auto_submit.py` — submission gating logic
- `scripts/fetch_submitted_alpha_library.py` — fetch submitted-alpha references
- `scripts/submitted_similarity.py` — similarity features and nearest-submitted matching
- `scripts/run_review_correct.sh` — main loop launcher

## Safety / operating principles

- Passing Sharpe/Fitness/Turnover is **not enough**
- Submitted similarity is a proxy, not the full self-correlation universe
- High-risk or repeatedly hopeless lineages should be deprioritized quickly
- Prefer structural novelty over superficial parameter tweaks

## Maintenance rule

From now on, script changes for this workflow should be mirrored into this repository so GitHub remains the clean external source of truth.
