#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a

python -m alpha_factory.cli decorrelate-generate --count 6 >> logs/decorrelate_pipeline.log 2>&1
python -m alpha_factory.cli import >> logs/decorrelate_pipeline.log 2>&1

if python scripts/auth_status.py --json >> logs/decorrelate_pipeline.log 2>&1; then
  python -m alpha_factory.cli run --limit 2 >> logs/decorrelate_pipeline.log 2>&1 || echo "$(date -u +%FT%TZ) run skipped/failed; backlog preserved" >> logs/decorrelate_pipeline.log
  python scripts/auto_submit.py >> logs/decorrelate_pipeline.log 2>&1 || echo "$(date -u +%FT%TZ) auto-submit skipped/failed; candidates preserved" >> logs/decorrelate_pipeline.log
else
  echo "$(date -u +%FT%TZ) WQ auth unavailable; generated/imported backlog only" >> logs/decorrelate_pipeline.log
fi
