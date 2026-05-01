#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "super_repair_state.json"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
SUPER_STATE_PATH = ROOT / "state" / "superalpha_state.json"

DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "INDUSTRY",
    "truncation": 0.04,
}

SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LOW_VOL_GATE = "ts_rank(ts_std_dev(returns, 10), 252) < 0.8"
LIQUID_GATE = "rank(adv20) > 0.2"
LIQUID_EXIT = "rank(adv20) < 0.05"
OVERNIGHT_VOL_HELPER = "rank(-ts_corr(abs((open - ts_delay(close, 1)) / ts_delay(close, 1)), ts_delay(volume / adv20, 1), 20))"
VWAP_OPEN_HELPER = "rank(ts_corr(vwap, open, 252))"
MARKET_RESID = "group_mean(returns, 1, market)"
SUBIND_RESID = "group_mean(returns, 1, subindustry)"



PRICE_EQUIV_REPLACEMENTS = [
    (r"\bclose\b", "open", "field_close_to_open"),
    (r"\bclose\b", "high", "field_close_to_high"),
    (r"\bclose\b", "low", "field_close_to_low"),
    (r"\bclose\b", "vwap", "field_close_to_vwap"),
    (r"\bopen\b", "vwap", "field_open_to_vwap"),
    (r"\breturns\b", "((open - ts_delay(close, 1)) / ts_delay(close, 1))", "field_returns_to_overnight"),
    (r"\breturns\b", "((close - open) / open)", "field_returns_to_intraday"),
]


def equivalent_field_variants(expr: str) -> list[tuple[str, str]]:
    core = expr.strip()
    out: list[tuple[str, str]] = []
    for pattern, repl, label in PRICE_EQUIV_REPLACEMENTS:
        new_expr, n = re.subn(pattern, repl, core)
        if n and new_expr != core:
            out.append((label, new_expr))
    overnight = "((open - ts_delay(close, 1)) / ts_delay(close, 1))"
    intraday = "((close - open) / open)"
    out.extend([
        ("field_overnight_overlay", f"group_neutralize(rank(ts_zscore(({core}), 120) + 0.16 * rank(ts_zscore({overnight}, 120))), subindustry)"),
        ("field_intraday_overlay", f"group_neutralize(rank(ts_zscore(({core}), 120) - 0.16 * rank(ts_zscore({intraday}, 120))), {SIZE_BUCKET})"),
        ("field_vwap_open_residual", f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_rank(({core}), 80) - ts_rank((vwap - open) / open, 120)), {SIZE_BUCKET}), {LIQUID_EXIT})"),
    ])
    return out


def operator_similarity_variants(expr: str) -> list[tuple[str, str]]:
    core = expr.strip()
    out: list[tuple[str, str]] = [
        ("op_rank_to_zscore", f"rank(ts_zscore(({core}), 120))"),
        ("op_rank_to_tsrank", f"rank(ts_rank(({core}), 120))"),
        ("op_decay_delta", f"group_neutralize(rank(ts_delta(ts_decay_linear(({core}), 12), 5)), subindustry)"),
        ("op_median_smooth", f"group_neutralize(rank(ts_median(({core}), 20)), {SIZE_BUCKET})"),
        ("op_corr_residual", f"group_neutralize(rank(ts_zscore(({core}), 120) - ts_rank(returns, 252)), subindustry)"),
    ]
    for old, new, label in [
        ("ts_mean(", "ts_median(", "op_mean_to_median"),
        ("ts_mean(", "ts_decay_linear(", "op_mean_to_decay"),
        ("ts_rank(", "ts_zscore(", "op_tsrank_to_zscore"),
        ("ts_zscore(", "ts_rank(", "op_zscore_to_tsrank"),
    ]:
        if old in core:
            out.append((label, core.replace(old, new, 1)))
    return out

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_state() -> dict[str, Any]:
    state = load_json(STATE_PATH)
    if not state:
        return {"version": 1, "created_at": now_iso(), "seeded_from": {}, "history": []}
    state.setdefault("seeded_from", {})
    state.setdefault("history", [])
    return state


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def existing_csv() -> tuple[set[str], set[str]]:
    exprs: set[str] = set()
    ids: set[str] = set()
    if not ALPHAS_CSV.exists():
        return exprs, ids
    with ALPHAS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("expression"):
                exprs.add(row["expression"].strip())
            if row.get("id"):
                ids.add(row["id"].strip())
    return exprs, ids


def self_corr_sources_from_log(limit: int = 20) -> list[str]:
    if not AUTO_SUBMIT_LOG.exists():
        return []
    counts: Counter[str] = Counter()
    for line in AUTO_SUBMIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-800:]:
        line = line.strip()
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
            if not alpha_id.startswith("super_"):
                continue
            if "SELF_CORRELATION" in reason or "self-correlation cooldown active" in reason:
                counts[alpha_id] += 1
    return [alpha_id for alpha_id, _ in counts.most_common(limit)]


def fetch_result(conn: sqlite3.Connection, alpha_id: str) -> sqlite3.Row | None:
    return conn.execute(
        '''
        select alpha_id, expression, status, sharpe, fitness, turnover, fail_reasons, created_at
        from backtest_results
        where alpha_id=?
        order by datetime(created_at) desc
        limit 1
        ''',
        (alpha_id,),
    ).fetchone()


def strong_super_sources(conn: sqlite3.Connection, limit: int = 8) -> list[sqlite3.Row]:
    blocked = self_corr_sources_from_log(limit=limit * 4)
    rows: list[sqlite3.Row] = []
    seen: set[str] = set()
    for alpha_id in blocked:
        row = fetch_result(conn, alpha_id)
        if row and row["expression"] and alpha_id not in seen:
            rows.append(row)
            seen.add(alpha_id)
    fresh = conn.execute(
        f'''
        select alpha_id, expression, status, sharpe, fitness, turnover, fail_reasons, created_at
        from backtest_results
        where {exact_prefix_clause('alpha_id')}
          and sharpe >= 1.6
          and fitness >= 1.0
          and turnover <= 0.45
          and expression is not null
        order by datetime(created_at) desc, sharpe desc, fitness desc
        limit ?
        ''',
        (exact_prefix_param("super_"), limit * 3),
    ).fetchall()
    for row in fresh:
        aid = str(row["alpha_id"])
        if aid not in seen:
            rows.append(row)
            seen.add(aid)
    rows.sort(key=lambda r: ((str(r["alpha_id"]) in set(blocked)), to_float(r["sharpe"]), to_float(r["fitness"])), reverse=True)
    return rows[:limit]


def parent_metadata(alpha_id: str) -> dict[str, Any]:
    super_state = load_json(SUPER_STATE_PATH)
    return (super_state.get("built_from") or {}).get(alpha_id, {})


def supersc_variants(expr: str) -> list[tuple[str, str]]:
    core = expr.strip()
    variants = [
        ("bucket_residual", f"group_neutralize(rank(ts_zscore(({core}), 120) - ts_zscore(cap, 120)), {SIZE_BUCKET})"),
        ("subindustry_residual", f"group_neutralize(rank(ts_rank(({core}), 120) - ts_rank({SUBIND_RESID}, 120)), subindustry)"),
        ("market_residual_gate", f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_zscore(({core}) - ts_mean(returns - {MARKET_RESID}, 20), 120)), subindustry), -1)"),
        ("liquidity_bucket_gate", f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_rank(({core}), 80) - ts_rank(volume / adv20, 120)), {SIZE_BUCKET}), {LIQUID_EXIT})"),
        ("overnight_helper", f"group_neutralize(rank(ts_rank(({core}), 80) + 0.22 * {OVERNIGHT_VOL_HELPER}), {SIZE_BUCKET})"),
        ("vwap_open_helper", f"group_neutralize(rank(ts_zscore(({core}), 80) - 0.18 * {VWAP_OPEN_HELPER}), subindustry)"),
        ("differenced_neutralizers", f"rank(ts_zscore(group_neutralize(ts_delta(({core}), 5), subindustry) - ts_mean(group_neutralize(({core}), {SIZE_BUCKET}), 15), 80))"),
        ("stability_smoothing", f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_mean(({core}), 5)), {SIZE_BUCKET}), -1)"),
        ("orthogonalized_helper", f"group_neutralize(rank(group_neutralize(rank(({core})), subindustry) + 0.30 * group_neutralize(rank(ts_zscore(({core}), 120) - ts_rank(returns, 252)), {SIZE_BUCKET})), subindustry)"),
        ("regime_split_residual", f"trade_when(rank(ts_std_dev(returns, 20)) > 0.65, group_neutralize(rank(ts_delta(({core}), 5)), subindustry), group_neutralize(rank(ts_decay_linear(({core}), 15) - 0.20 * ts_rank(volume/adv20, 120)), {SIZE_BUCKET}))"),
    ]
    variants.extend(equivalent_field_variants(core))
    variants.extend(operator_similarity_variants(core))
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for label, variant in variants:
        if variant and variant not in seen and variant.strip() != core:
            seen.add(variant)
            deduped.append((label, variant))
    return deduped


def make_id(source_id: str, label: str, expression: str) -> str:
    digest = hashlib.sha1(f"supersc|{source_id}|{label}|{expression}".encode("utf-8")).hexdigest()[:10]
    return f"supersc_{digest}"


def append_rows(rows: list[dict[str, Any]], max_add: int) -> int:
    if not rows:
        return 0
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added = 0
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for row in rows:
            expr = row["expression"].strip()
            alpha_id = row["id"].strip()
            if not expr or expr in existing_exprs or alpha_id in existing_ids:
                continue
            writer.writerow({"id": alpha_id, "expression": expr, **DEFAULTS})
            existing_exprs.add(expr)
            existing_ids.add(alpha_id)
            added += 1
            if added >= max_add:
                break
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate supersc_ de-correlation repairs for strong SuperAlpha candidates")
    parser.add_argument("--max-add", type=int, default=3)
    parser.add_argument("--source-limit", type=int, default=4)
    args = parser.parse_args()

    state = load_state()
    seeded_from: dict[str, Any] = state.setdefault("seeded_from", {})
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sources = strong_super_sources(conn, limit=args.source_limit)
    candidates: list[dict[str, Any]] = []

    for row in sources:
        source_id = str(row["alpha_id"])
        meta = parent_metadata(source_id)
        for label, expr in supersc_variants(str(row["expression"])):
            repair_id = make_id(source_id, label, expr)
            if repair_id in seeded_from:
                continue
            candidates.append({
                "id": repair_id,
                "expression": expr,
                "source_id": source_id,
                "repair_type": label,
                "parent_alpha_ids": meta.get("parent_alpha_ids", []),
                "source_sharpe": to_float(row["sharpe"]),
                "source_fitness": to_float(row["fitness"]),
                "source_turnover": to_float(row["turnover"]),
            })
            seeded_from[repair_id] = {
                "source_id": source_id,
                "repair_type": label,
                "parent_alpha_ids": meta.get("parent_alpha_ids", []),
                "created_at": now_iso(),
            }

    added = append_rows(candidates, args.max_add)
    event = {
        "timestamp_utc": now_iso(),
        "sources": [str(r["alpha_id"]) for r in sources],
        "generated": len(candidates),
        "added": added,
        "repair_types": sorted({c["repair_type"] for c in candidates}),
    }
    state.setdefault("history", []).append(event)
    state["history"] = state["history"][-100:]
    save_state(state)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
