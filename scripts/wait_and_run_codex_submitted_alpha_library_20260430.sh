#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory"
cd "$ROOT"
LOG="logs/codex_submitted_alpha_library_20260430.log"
LAST="logs/codex_submitted_alpha_library_20260430.last.md"
PLAN="CODEX_SUBMITTED_ALPHA_LIBRARY_PLAN_20260430.md"

mkdir -p logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] queued submitted-alpha-library Codex optimization" | tee -a "$LOG"

active_count() {
  local proc_count db_count
  proc_count=$(ps ax -o command | egrep 'alpha_factory|run_review_correct|run_decorrelate_pipeline|strategy_monitor|auto_submit|decorrelate_pipeline|self_corr_truth_table|ml_candidate_scorer' | grep -v egrep | wc -l | tr -d ' ')
  db_count=$(sqlite3 data/backtests.sqlite3 "select count(*) from alpha_tasks where status in ('RUNNING','SUBMITTING','RETRY');" 2>/dev/null || echo 1)
  echo "$proc_count $db_count"
}

while true; do
  read -r pc dc < <(active_count)
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] guard check proc_count=$pc db_active=$dc" | tee -a "$LOG"
  if [[ "$pc" == "0" && "$dc" == "0" ]]; then
    break
  fi
  sleep 120
done

# Recheck auth, recover only if needed and local recovery is configured.
if ! ./.venv/bin/python scripts/auth_status.py --json | tee -a "$LOG" | grep -q '"ok": true'; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] auth not ok; trying local auth_recover.py" | tee -a "$LOG"
  ./.venv/bin/python scripts/auth_recover.py | tee -a "$LOG" || true
  ./.venv/bin/python scripts/auth_status.py --json | tee -a "$LOG"
fi

PROMPT=$(cat <<'PROMPT_EOF'
Read CODEX_SUBMITTED_ALPHA_LIBRARY_PLAN_20260430.md carefully and implement it.

Important execution mode:
- This is Plan Mode / high-reasoning implementation.
- Follow the hard safety constraints in the plan exactly.
- Do not perform any WorldQuant remote write or submit.
- Use read-only GET for submitted alpha fetch only.
- Preserve AUTO_SUBMIT_MAX_P_SELF_CORR=0.20 and all existing submit gates.
- Keep changes minimal, deterministic, and tested.
- After implementation, run the validation commands listed in the plan as far as possible.
- If a command cannot run due to auth/network/rate limit, report that clearly.

At the end, provide a concise final report with files changed, validations, fetched library count, dry-run summary, and blockers.
PROMPT_EOF
)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting codex" | tee -a "$LOG"
/opt/homebrew/bin/codex exec --full-auto --model gpt-5.5 -c 'model_reasoning_effort="xhigh"' --output-last-message "$LAST" "$PROMPT" 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] codex exited status=$status" | tee -a "$LOG"
exit "$status"
