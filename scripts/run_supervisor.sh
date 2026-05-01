#!/bin/zsh
set -euo pipefail
cd /Users/FruityClaw/.openclaw/workspace/worldquant-alpha-factory
source .venv/bin/activate
python -m alpha_factory.supervisor --interval 120 --batch-size 1 >> logs/supervisor.log 2>&1
