#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
MULTI_STATE_PATH = ROOT / "state" / "multi_dataset_state.json"
ML_STATE_PATH = ROOT / "state" / "ml_candidate_scorer_state.json"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
OUTPUT_PATH = ROOT / "state" / "multi_d1_panel.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def blocked_reasons() -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = defaultdict(list)
    if not AUTO_SUBMIT_LOG.exists():
        return reasons
    for line in AUTO_SUBMIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-2500:]:
        if '"skipped"' not in line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        for item in payload.get("skipped", []):
            if len(item) < 2:
                continue
            alpha_id, reason = str(item[0]), str(item[1])
            reasons[alpha_id].append(reason)
    return reasons


def submit_clear_ids() -> set[str]:
    clear: set[str] = set()
    if not AUTO_SUBMIT_LOG.exists():
        return clear
    for line in AUTO_SUBMIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-2500:]:
        if '"submitted"' not in line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        for item in payload.get("submitted", []):
            if isinstance(item, dict) and item.get("local_id"):
                clear.add(str(item["local_id"]))
    return clear


def summarize_bucket(alpha_ids: list[str], score_index: dict[str, dict[str, Any]], blocked: dict[str, list[str]], clear_ids: set[str], rows: dict[str, sqlite3.Row]) -> dict[str, Any]:
    tested_rows = [rows[aid] for aid in alpha_ids if aid in rows]
    tested = len(tested_rows)
    if tested == 0:
        return {
            "tested": 0,
            "pass": 0,
            "strong": 0,
            "avg_sharpe": 0.0,
            "self_corr_block_rate": 0.0,
            "d1_ready_rate": 0.0,
            "submit_clear_rate": 0.0,
        }
    sharpes = [to_float(r["sharpe"]) for r in tested_rows if r["sharpe"] is not None]
    pass_count = sum(1 for r in tested_rows if to_float(r["sharpe"]) >= 1.6 and to_float(r["fitness"]) >= 1.0 and to_float(r["turnover"], 99) <= 0.45)
    strong = sum(1 for r in tested_rows if to_float(r["sharpe"]) >= 1.8 and to_float(r["fitness"]) >= 1.15 and to_float(r["turnover"], 99) <= 0.30)
    self_corr = sum(1 for aid in alpha_ids if any("SELF_CORRELATION" in r or "self-correlation cooldown active" in r for r in blocked.get(aid, [])))
    d1_ready = sum(1 for aid in alpha_ids if (score_index.get(aid) or {}).get("d1_ready") == 1)
    submit_clear = sum(1 for aid in alpha_ids if aid in clear_ids)
    return {
        "tested": tested,
        "pass": pass_count,
        "strong": strong,
        "avg_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0,
        "self_corr_block_rate": round(self_corr / max(1, tested), 4),
        "d1_ready_rate": round(d1_ready / max(1, tested), 4),
        "submit_clear_rate": round(submit_clear / max(1, tested), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize local multi_ D1 readiness by function family and dataset pair")
    parser.parse_args()

    multi_state = load_json(MULTI_STATE_PATH)
    ml_state = load_json(ML_STATE_PATH)
    alpha_meta = multi_state.get("alphas") or {}
    score_index = {str(x.get("alpha_id")): x for x in (ml_state.get("top_candidates") or []) if isinstance(x, dict) and x.get("alpha_id")}
    blocked = blocked_reasons()
    clear_ids = submit_clear_ids()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = {
        str(r["alpha_id"]): r
        for r in conn.execute(f"select alpha_id, sharpe, fitness, turnover, fail_reasons from backtest_results where {exact_prefix_clause('alpha_id')}", (exact_prefix_param("multi_"),)).fetchall()
    }
    conn.close()

    by_family: dict[str, list[str]] = defaultdict(list)
    by_dataset_pair: dict[str, list[str]] = defaultdict(list)
    by_function_family: dict[str, list[str]] = defaultdict(list)
    composite: dict[str, list[str]] = defaultdict(list)

    for alpha_id, meta in alpha_meta.items():
        interaction = str(meta.get("interaction_type") or "unknown")
        datasets = meta.get("datasets") or []
        pair = "|".join(sorted(str(x) for x in datasets if x)) if datasets else "unknown"
        by_family["multi_"] .append(alpha_id)
        by_dataset_pair[pair].append(alpha_id)
        by_function_family[interaction].append(alpha_id)
        composite[f"{interaction}::{pair}"].append(alpha_id)

    payload = {
        "updated_at": now_iso(),
        "family": {k: summarize_bucket(v, score_index, blocked, clear_ids, rows) for k, v in by_family.items()},
        "function_family": {k: summarize_bucket(v, score_index, blocked, clear_ids, rows) for k, v in by_function_family.items()},
        "dataset_pair": {k: summarize_bucket(v, score_index, blocked, clear_ids, rows) for k, v in by_dataset_pair.items()},
        "function_family_x_dataset_pair": {k: summarize_bucket(v, score_index, blocked, clear_ids, rows) for k, v in composite.items()},
        "summary": {
            "tracked_multi_alphas": len(alpha_meta),
            "tested_multi_alphas": len(rows),
            "function_families": len(by_function_family),
            "dataset_pairs": len(by_dataset_pair),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"updated_at": payload["updated_at"], "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
