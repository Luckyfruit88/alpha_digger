#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a

if python scripts/auth_status.py --json >> logs/auto_submit.log 2>&1 || python scripts/auth_recover.py >> logs/auto_submit.log 2>&1; then
  python scripts/auto_submit.py >> logs/auto_submit.log 2>&1 || echo "$(date -u +%FT%TZ) auto-submit skipped/failed; candidates preserved" >> logs/auto_submit.log
else
  echo "$(date -u +%FT%TZ) WQ auth unavailable; auto-submit deferred" >> logs/auto_submit.log
fi
