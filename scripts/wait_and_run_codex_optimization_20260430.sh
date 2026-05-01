#!/usr/bin/env bash
set -euo pipefail
ROOT="/Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory"
LOG="$ROOT/logs/codex_optimization_20260430.log"
LAST="$ROOT/logs/codex_optimization_20260430.last.md"
cd "$ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for active WorldQuant workflow to finish" >> "$LOG"
while true; do
  active_proc=$(ps ax -o command | egrep 'alpha_factory|run_review_correct|strategy_monitor|auto_submit|decorrelate_pipeline|self_corr_truth_table|ml_candidate_scorer' | grep -v egrep || true)
  active_rows=$(sqlite3 data/backtests.sqlite3 "select count(*) from alpha_tasks where status in ('RUNNING','SUBMITTING','RETRY');")
  if [[ -z "$active_proc" && "${active_rows:-0}" == "0" ]]; then
    break
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] still active rows=$active_rows" >> "$LOG"
  sleep 120
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] active workflow clear; starting Codex" >> "$LOG"
PROMPT=$(cat <<'EOF'
You are Codex CLI running in Plan Mode using model gpt-5.5 with high/xhigh reasoning. Work only in this repository.

Read CODEX_OPTIMIZATION_PLAN_20260430.md fully and implement it with minimal, safe, reviewable changes.

Critical constraints:
- Do not delete or modify secrets/cookies/auth files.
- Do not delete data/logs/state/reports history.
- Do not run live submit. Validation must be dry-run/function tests/report generation only.
- Preserve AUTO_SUBMIT_MAX_P_SELF_CORR=0.20 for normal submit.
- FDE and D1 relaxed channels are label-collection only; they must never bypass unsafe detail checks or normal submit gates.
- Maintain backwards compatibility with existing state/*.json and SQLite.

Required final response:
- implementation plan
- files changed
- commands/tests run and results
- blockers
- expected operational effect
EOF
)
# Use non-PTY exec style because previous PTY run on this host could echo prompt and exit without work.
codex exec --full-auto --model gpt-5.5 -c 'model_reasoning_effort="high"' --cd "$ROOT" --output-last-message "$LAST" "$PROMPT" >> "$LOG" 2>&1
status=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Codex exited status=$status" >> "$LOG"
exit $status
