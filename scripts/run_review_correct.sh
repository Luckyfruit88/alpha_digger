#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
mkdir -p logs state
LOCKDIR="state/review_correct.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) review_correct skipped: previous run still active" >> logs/review_correct.log
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a
python scripts/review_and_correct.py >> logs/review_correct.log 2>&1
