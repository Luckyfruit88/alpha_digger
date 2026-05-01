#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a

python -m alpha_factory.cli refit-generate --count 6 >> logs/refit_pipeline.log 2>&1
python -m alpha_factory.cli import >> logs/refit_pipeline.log 2>&1

if python scripts/auth_status.py --json >> logs/refit_pipeline.log 2>&1; then
  python -m alpha_factory.cli run --limit 2 >> logs/refit_pipeline.log 2>&1 || echo "$(date -u +%FT%TZ) run skipped/failed; backlog preserved" >> logs/refit_pipeline.log
  python scripts/auto_submit.py >> logs/refit_pipeline.log 2>&1 || echo "$(date -u +%FT%TZ) auto-submit skipped/failed; candidates preserved" >> logs/refit_pipeline.log
else
  echo "$(date -u +%FT%TZ) WQ auth unavailable; generated/imported backlog only" >> logs/refit_pipeline.log
fi
