#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a
python scripts/healthcheck.py >> logs/healthcheck.log 2>&1
