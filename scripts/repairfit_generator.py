#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "repairfit_state.json"

DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "INDUSTRY",
    "truncation": 0.04,
}

VARIANTS = {
    "smooth_decay5": lambda e: f"ts_decay_linear(({e}), 5)",
    "subindustry_neutral": lambda e: f"group_neutralize(({e}), subindustry)",
    "rank_minus_mean3": lambda e: f"rank(({e})) - rank(ts_mean(({e}), 3))",
    "sector_zscore20": lambda e: f"ts_zscore(group_neutralize(({e}), sector), 20)",
    "rank_smooth_neutral": lambda e: f"group_neutralize(rank(ts_decay_linear(({e}), 4)), subindustry)",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_csv() -> tuple[set[str], set[str]]:
    exprs: set[str] = set()
    ids: set[str] = set()
    if not ALPHAS_CSV.exists():
        return exprs, ids
    with ALPHAS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expr = (row.get("expression") or "").strip()
            aid = (row.get("id") or "").strip()
            if expr:
                exprs.add(expr)
            if aid:
                ids.add(aid)
    return exprs, ids


def anchor_rows(anchor_ids: list[str]) -> list[sqlite3.Row]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if anchor_ids:
        qs = ",".join("?" for _ in anchor_ids)
        sql = f"""
            select r.alpha_id, r.sharpe, r.fitness, r.turnover, t.expression
            from backtest_results r join alpha_tasks t on r.alpha_id = t.id
            where r.alpha_id in ({qs}) and t.expression is not null
            order by r.sharpe * 0.4 + r.fitness * 0.6 desc
        """
        rows = conn.execute(sql, anchor_ids).fetchall()
    else:
        rows = conn.execute(f"""
            select r.alpha_id, r.sharpe, r.fitness, r.turnover, t.expression
            from backtest_results r join alpha_tasks t on r.alpha_id = t.id
            where {exact_prefix_clause('r.alpha_id')}
              and r.sharpe >= 1.2 and r.fitness >= 0.65 and r.turnover <= 0.55
              and t.expression is not null
            order by r.sharpe * 0.4 + r.fitness * 0.6 desc
            limit 5
        """, (exact_prefix_param("multi_"),)).fetchall()
    conn.close()
    return rows


def append_repairfit(rows: list[sqlite3.Row], max_add: int) -> list[dict[str, Any]]:
    state = load_json(STATE_PATH) or {"version": 1, "created_at": now_iso(), "candidates": {}, "history": []}
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"])
        if is_new:
            writer.writeheader()
        for row in rows:
            anchor_id = str(row["alpha_id"])
            base = str(row["expression"])
            for variant_name, builder in VARIANTS.items():
                expr = builder(base)
                aid = "repairfit_" + hashlib.sha1(f"{anchor_id}|{variant_name}|{expr}".encode("utf-8")).hexdigest()[:10]
                if expr in existing_exprs or aid in existing_ids:
                    continue
                writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(aid)
                meta = {
                    "track": "repairfit",
                    "anchor_id": anchor_id,
                    "variant_name": variant_name,
                    "anchor_sharpe": row["sharpe"],
                    "anchor_fitness": row["fitness"],
                    "anchor_turnover": row["turnover"],
                    "created_at": now_iso(),
                }
                state.setdefault("candidates", {})[aid] = meta
                added.append({"id": aid, **meta})
                if len(added) >= max_add:
                    state.setdefault("history", []).append({"timestamp_utc": now_iso(), "added": added})
                    state["history"] = state["history"][-100:]
                    save_json(STATE_PATH, state)
                    return added
    state.setdefault("history", []).append({"timestamp_utc": now_iso(), "added": added})
    state["history"] = state["history"][-100:]
    save_json(STATE_PATH, state)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fitness-repair variants for promising multi_ anchors")
    parser.add_argument("--anchor-id", action="append", default=[])
    parser.add_argument("--max-add", type=int, default=5)
    args = parser.parse_args()
    rows = anchor_rows(args.anchor_id)
    added = append_repairfit(rows, args.max_add)
    print(json.dumps({"timestamp_utc": now_iso(), "anchors": [r["alpha_id"] for r in rows], "added": added}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
