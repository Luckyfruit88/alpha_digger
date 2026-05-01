#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
set -a
[ -f secrets/autonomy.env ] && source secrets/autonomy.env
set +a
python scripts/strategy_monitor.py >> logs/strategy_monitor.log 2>&1
