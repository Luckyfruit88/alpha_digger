# WorldQuant Alpha Factory Runbook

_Last updated: 2026-04-30_

## Quick status check

Run from the project root:

```bash
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory

# Auth/session status
.venv/bin/python scripts/auth_status.py --json

# Latest controller state/report
cat state/review_correct_state.json
sed -n '1,220p' reports/review_correct_latest.md

# Recent controller log
tail -120 logs/review_correct.log

# ML scorer and D1 readiness snapshot
python - <<'PY'
import json
for path in ['state/ml_candidate_scorer_state.json', 'state/d1_generator_state.json']:
    try:
        data=json.load(open(path))
    except FileNotFoundError:
        print(path, 'missing')
        continue
    print('##', path)
    for key in ['timestamp_utc', 'summary', '_schema_version']:
        if key in data:
            print(key, data[key])
PY

# Self-correlation truth table
.venv/bin/python scripts/self_corr_truth_table.py
sed -n '1,220p' reports/self_corr_truth_table.md

# Autosubmit state, daily cap, and exploration counters
python - <<'PY'
import json
for path in ['state/auto_submit_state.json', 'state/d1_truth_table.json']:
    try:
        data=json.load(open(path))
    except FileNotFoundError:
        print(path, 'missing')
        continue
    print('##', path)
    for key in ['utc_date', 'daily_submit_count', 'daily_submit_limit', 'fde_total_checks', 'd1_total_checks', 'total', 'clear', 'blocked', 'pending']:
        if key in data:
            print(key, data[key])
PY

# Process check
ps aux | grep -E 'review_correct|alpha_factory|worldquant|d1v2|d1v22|d1v23' | grep -v grep || true
```

## Start or resume the controller

Preferred launcher:

```bash
/bin/zsh scripts/run_review_correct.sh
```

For long runs, use a detached launch or a tool timeout longer than the controller's possible runtime:

```bash
nohup /bin/zsh scripts/run_review_correct.sh >> logs/review_correct.nohup.log 2>&1 &
```

Do **not** start a long controller run with a short interactive timeout. A forced termination can leave `state/review_correct.lock` behind and cause later cron cycles to skip with `previous run still active`.

## Lock handling

The lock path is:

```text
state/review_correct.lock
```

Before touching it, always inspect processes:

```bash
ps aux | grep -E 'review_correct|alpha_factory|worldquant' | grep -v grep || true
```

If no matching process exists and the log shows repeated skips, treat the lock as stale. Prefer moving it aside instead of deleting it:

```bash
mv state/review_correct.lock state/review_correct.lock.stale-$(date -u +%Y%m%dT%H%M%SZ)
```

Never clear a lock while a real controller or `alpha_factory.cli run` process is active. Independent D1 validation chains such as `alpha_factory.cli run --id-prefix d1v23_` can coexist with a stale controller lock; distinguish them before moving the lock aside.

## Auth handling

Check auth:

```bash
.venv/bin/python scripts/auth_status.py --json
.venv/bin/python -m alpha_factory.cli check-auth
```

If auth expires or `BrainAuthError` appears during simulation polling, refresh through the approved local recovery workflow:

```bash
.venv/bin/python scripts/auth_recover.py
.venv/bin/python scripts/auth_status.py --json
```

The current private recovery config lives in `secrets/autonomy.env`; `WQ_CREDENTIALS_FILE=secrets/worldquant_credentials.json` points to the local JSON credentials file. The credentials file supports `email`, `username`, or `account` plus `password`, must stay mode `600`, and must never be copied into chat, docs, commits, or logs. `scripts/auth_recover.py` writes a refreshed cookie to `secrets/worldquant.cookie` and falls back to the existing Safari/Keychain path if JSON credentials are absent/placeholders.

`run_auto_submit.sh` now attempts `scripts/auth_recover.py` automatically when `auth_status.py --json` fails, then re-checks auth before deferring autosubmit.

Symptoms of auth problems:

- `BrainAuthError`
- `unauthorized/expired`
- `auth_error` family counts increasing
- simulations failing shortly after a state report said auth was OK

Auth can be inconsistent: a status check may pass while a later simulation endpoint fails. Treat simulation failures as stronger evidence.

## Autosubmit checks

Inspect autosubmit decisions:

```bash
tail -160 logs/auto_submit.log
```

Important blocker strings:

- `SELF_CORRELATION:PENDING`
- `self-correlation cooldown active`
- `detail_budget_exhausted`
- `unsafe checks`
- `already recorded`

A candidate is not submit-clear just because Sharpe/Fitness/Turnover pass. Self-correlation checks, detail budget, pre-submit ML self-correlation risk, and D1-ready status must also clear.

Real autosubmit should remain disabled unless a dry-run shows all of the following: passing quality metrics, `p_self_corr_block <= 0.20`, `d1_ready=1` when the D1 gate is required, no auth/detail/self-correlation blockers, and a non-empty submit-clear decision path.

Current conservative knobs:

```text
AUTO_SUBMIT_MIN_SHARPE=1.65
AUTO_SUBMIT_MIN_FITNESS=1.05
AUTO_SUBMIT_MAX_TURNOVER=0.40
AUTO_SUBMIT_MAX_DETAIL_CHECKS=8
AUTO_SUBMIT_SELF_CORR_COOLDOWN_HOURS=8
AUTO_SUBMIT_MAX_P_SELF_CORR=0.20
AUTO_SUBMIT_REQUIRE_D1_READY=1
AUTO_SUBMIT_DAILY_LIMIT=1
AUTO_SUBMIT_SELF_CORR_EXPLORATION_CHECKS=2
AUTO_SUBMIT_MAX_EXPLORATION_P_SELF_CORR=0.75
AUTO_SUBMIT_FDE_ENABLED=1
AUTO_SUBMIT_FDE_CHECKS_PER_ROUND=2
AUTO_SUBMIT_D1_EXPLORATION_ENABLED=1
AUTO_SUBMIT_D1_MAX_P_SELF_CORR=0.35
AUTO_SUBMIT_D1_CHECKS_PER_ROUND=2
```

`AUTO_SUBMIT_DAILY_LIMIT` is enforced by UTC date in `state/auto_submit_state.json`. The current default is `1`; raise to `2` only after explicit user approval. Dry-runs and detail checks do not increment the real-submit counter.

Exploration/FDE/D1 relaxed checks are label collection only. Even if one of these checks finds a clear detail result, real submit still requires the normal strict submit gate and daily cap.

## Candidate family priorities

As of 2026-04-29:

1. `decor_`, `repairsc_`, `repairsc2_` remain the most stable/high-volume sources.
2. `super_` is the strongest new raw-quality track but remains self-correlation blocked.
3. `supersc_` and `repairsc3_` are the primary structural repair tracks for self-correlation.
4. Forced Diversity Exploration, lineage priors, and repair-depth retirement are intended to reduce repeated same-lineage/self-corr-blocked variants.
5. `multi_` is active but should stay low-budget until value density improves.
6. D1/D1v2 candidates are the main submit-aware low-self-correlation lane; `d1v22_` whole-anchor proxy should not be scaled after first-batch signal collapse.
7. `d1v23_` surgical/helper-target variants are the next small-batch direction to test, with relaxed D1 detail checks used only for labels.
8. ML scoring is useful for ranking but must be calibrated against real autosubmit blockers and truth-table outcomes.

## Recommended monitoring questions

During heartbeat/status checks, report only meaningful changes:

- Did auth expire or recover?
- Did a controller cycle complete or fail?
- Did a live process get stuck, or is a lock stale?
- Did `super_`, `supersc_`, `repairsc2_`, `repairsc3_`, or D1-family candidates produce a new strong/pass candidate?
- Did FDE or D1 relaxed exploration collect new truth-table labels?
- Did autosubmit succeed or produce a new blocker pattern?

If none of these changed, stay quiet.

## Common failure modes

### Stale lock

Cause: controller killed after creating `state/review_correct.lock`.

Fix: verify no live process, then move lock aside as a timestamped stale backup.

### Auth appears OK but runs fail

Cause: status endpoint/session check can lag behind simulation endpoint validity.

Fix: refresh cookie/session and rerun a small simulation batch.

### High raw metrics but no submission

Cause: `SELF_CORRELATION:PENDING`, cooldown, detail budget, or unsafe checks.

Fix: prioritize structural `supersc_` / `repairsc3_` variants over simple window/weight tweaks.

### Multi Data Set weak early results

Cause: interaction/function-family search may be too narrow or early sample too small.

Fix: continue only small-batch exploration while broadening interactions; do not spend large budget until value density improves.

### D1 proxy signal collapse

Cause: a low-self-correlation rewrite can remove too much of the original signal, especially whole-anchor proxy substitution.

Fix: avoid scaling failed `d1v22_` proxy patterns. Prefer small `d1v23_` surgical/helper-target batches that preserve internal D1v2 signal components while reducing only the high-risk correlation target.

### Auth expired during autosubmit

Cause: cookie/session expired before an autosubmit cron or wrapper run.

Fix: `run_auto_submit.sh` should call `scripts/auth_recover.py` automatically. If testing manually, run `scripts/auth_recover.py`, then `scripts/auth_status.py --json`. Confirm the credentials JSON exists, is non-placeholder, and has mode `600`; never print its contents.

### Daily submit cap reached

Cause: `AUTO_SUBMIT_DAILY_LIMIT` for the UTC day has already been used.

Fix: do nothing unless the user explicitly approves raising the cap to `2`. Inspect `state/auto_submit_state.json` for `utc_date` and `daily_submit_count`.

### FDE/D1 exploration not producing labels

Cause: no candidate meets `fde_candidate` or relaxed D1 criteria, candidates exceed exploration risk caps, or detail budget/cooldowns are exhausted.

Fix: inspect `state/ml_candidate_scorer_state.json`, `state/auto_submit_state.json`, `state/d1_truth_table.json`, and `logs/auto_submit.log`. Do not loosen normal submit gates; adjust generation/scoring only after evidence.

## Safe operating principles

- Prefer small batches and controller-managed runs.
- Preserve evidence: reports, state files, and logs are useful for debugging.
- Move stale locks aside rather than deleting.
- Never reveal secrets/cookies/API keys.
- Do not bypass platform access controls or safety checks.
