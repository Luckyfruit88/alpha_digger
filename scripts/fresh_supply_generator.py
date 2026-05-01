#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submitted_similarity import load_submitted_features, score_against_submitted

FIELDS_CSV = ROOT / "data" / "data_fields_scored.csv"
ALPHAS_CSV = ROOT / "alphas.csv"
DB_PATH = ROOT / "data" / "backtests.sqlite3"
STATE_PATH = ROOT / "state" / "fresh_supply_state.json"
REPORT_PATH = ROOT / "reports" / "fresh_supply_latest.md"
SUBMITTED_THRESHOLD = 0.58

DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "INDUSTRY",
    "truncation": 0.04,
}

SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LIQUID_GATE = "rank(adv20) > 0.2"
LIQUID_EXIT = "rank(adv20) < 0.05"
LOW_VOL_GATE = "ts_rank(ts_std_dev(returns, 20), 252) < 0.65"
MARKET_RET = "group_mean(returns, 1, market)"

AVOID_DATASET_TOKENS = {"analyst4"}
AVOID_FIELD_TOKENS = ("anl", "analyst", "eps", "estimate", "revision", "rating", "recommend")
PREFERRED_LINEAGES = {"fundamental_valuation", "liquidity_volatility", "market_size", "price_volume", "model_mixed", "unknown"}


TOKEN_LINEAGES: list[tuple[str, tuple[str, ...]]] = [
    ("analyst_earnings", ("anl", "analyst", "eps", "earning", "estimate", "revision", "rating")),
    ("fundamental_valuation", ("asset", "book", "cash", "debt", "enterprise", "fcf", "fundamental", "income", "liab", "margin", "profit", "revenue", "sales", "value", "roe")),
    ("liquidity_volatility", ("adv", "volume", "liquidity", "turnover", "volatility", "std", "variance", "risk", "spread")),
    ("market_size", ("cap", "marketcap", "size")),
    ("price_volume", ("adjfactor", "close", "open", "price", "return", "vwap", "high", "low")),
    ("model_mixed", ("model", "score", "factor", "composite", "torpedo", "trend", "stability", "distress")),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "created_at": now_iso(), "history": [], "alphas": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "created_at": now_iso(), "history": [], "alphas": {}}


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now_iso()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def classify_lineage(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k) or "").lower() for k in ["dataset_id", "dataset_name", "id", "category", "subcategory", "description"])
    for name, needles in TOKEN_LINEAGES:
        if any(n in text for n in needles):
            return name
    return "unknown"


def load_fields() -> list[dict[str, Any]]:
    if not FIELDS_CSV.exists():
        return []
    out: list[dict[str, Any]] = []
    with FIELDS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("region") and row.get("region") != "USA":
                continue
            if row.get("universe") and row.get("universe") != "TOP3000":
                continue
            if row.get("type") and row.get("type") not in {"MATRIX", "VECTOR"}:
                continue
            coverage = to_float(row.get("coverage"))
            if coverage < 0.55:
                continue
            text = " ".join(str(row.get(k) or "").lower() for k in ["dataset_id", "id", "dataset_name", "category"])
            if any(tok in text for tok in AVOID_DATASET_TOKENS) or any(tok in text for tok in AVOID_FIELD_TOKENS):
                continue
            row = dict(row)
            row["lineage_theme"] = classify_lineage(row)
            if row["lineage_theme"] == "analyst_earnings":
                continue
            out.append(row)
    return out


def field_prior(row: dict[str, Any], submitted_field_counts: Counter[str], tested_counts: Counter[str]) -> float:
    coverage = to_float(row.get("coverage"))
    date_cov = to_float(row.get("dateCoverage"))
    score = to_float(row.get("score"))
    alpha_count = to_int(row.get("alphaCount"))
    user_count = to_int(row.get("userCount"))
    lineage = row.get("lineage_theme") or "unknown"
    crowd_bonus = 8.0 / math.sqrt(1 + max(0, user_count))
    alpha_prior = 2.0 * math.log1p(alpha_count)
    submitted_penalty = 5.0 * submitted_field_counts.get(str(row.get("id", "")).lower(), 0)
    tested_penalty = 0.8 * tested_counts.get(str(row.get("id", "")).lower(), 0)
    lineage_bonus = 9.0 if lineage in PREFERRED_LINEAGES else 0.0
    return score / 6 + 24 * coverage + 8 * date_cov + crowd_bonus + alpha_prior + lineage_bonus - submitted_penalty - tested_penalty


def submitted_field_counts(submitted: list[dict[str, Any]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for item in submitted:
        feats = item.get("features") or {}
        c.update(str(x).lower() for x in feats.get("fields") or [])
    return c


def tested_field_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    if not DB_PATH.exists():
        return counts
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("select expression from alpha_tasks where status in ('COMPLETE','ERROR','TIMEOUT','PAUSED_LOW_QUALITY_AUTO')").fetchall()
    conn.close()
    # Lightweight substring accounting; enough to avoid repeatedly sampling the same obvious field ids.
    fields = [r.get("id", "") for r in load_fields()]
    for row in rows[-1200:]:
        expr = str(row["expression"] or "").lower()
        for field in fields:
            if field and field.lower() in expr:
                counts[field.lower()] += 1
    return counts


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


def low_collision_templates(field: str, lineage: str) -> list[tuple[str, str]]:
    f = field
    base = f"rank(ts_zscore({f}, 160) - ts_zscore({MARKET_RET}, 60))"
    return [
        ("field_market_residual", f"group_neutralize({base}, {SIZE_BUCKET})"),
        ("liquid_lowvol_gate", f"trade_when({LIQUID_GATE} && {LOW_VOL_GATE}, group_neutralize(rank(ts_delta(ts_mean({f}, 20), 80)), subindustry), {LIQUID_EXIT})"),
        ("slow_reversal_helper", f"group_neutralize(rank(ts_rank({f}, 252) - ts_rank(ts_mean(returns, 20), 252)), {SIZE_BUCKET})"),
        ("nonlinear_compression", f"group_neutralize(rank(signed_power(rank(ts_zscore({f}, 120)) - 0.5, 3)), subindustry)"),
        ("liquidity_orthogonal", f"group_neutralize(rank(ts_rank({f}, 160) - ts_rank(volume / adv20, 60)), {SIZE_BUCKET})"),
        ("corr_anti_beta", f"-rank(ts_corr(ts_rank({f}, 120), ts_rank(returns - {MARKET_RET}, 120), 120))"),
    ]


def candidate_id(field_row: dict[str, Any], template: str, expr: str) -> str:
    key = "|".join([str(field_row.get("dataset_id") or ""), str(field_row.get("id") or ""), template, expr])
    return "fresh_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def generate(max_add: int, field_limit: int, threshold: float) -> dict[str, Any]:
    state = load_state()
    submitted = load_submitted_features()
    sub_counts = submitted_field_counts(submitted)
    tested_counts = tested_field_counts()
    fields = load_fields()
    ranked = sorted(fields, key=lambda r: field_prior(r, sub_counts, tested_counts), reverse=True)[:field_limit]
    existing_exprs, existing_ids = existing_csv()
    added: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    checked = 0
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    ALPHAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        seen_dataset: Counter[str] = Counter()
        seen_lineage: Counter[str] = Counter()
        for row in ranked:
            field = str(row.get("id") or "")
            lineage = str(row.get("lineage_theme") or "unknown")
            dataset = str(row.get("dataset_id") or "")
            if seen_dataset[dataset] >= 3 or seen_lineage[lineage] >= 4:
                continue
            for template, expr in low_collision_templates(field, lineage):
                checked += 1
                expr = expr.strip()
                if expr in existing_exprs:
                    skipped["duplicate_expr"] += 1
                    continue
                alpha_id = candidate_id(row, template, expr)
                if alpha_id in existing_ids:
                    skipped["duplicate_id"] += 1
                    continue
                sim = score_against_submitted(expr, DEFAULTS, submitted)
                if float(sim.get("max_similarity", 0.0) or 0.0) >= threshold:
                    skipped[f"submitted_{sim.get('collision_level','unknown')}"] += 1
                    continue
                writer.writerow({"id": alpha_id, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(alpha_id)
                seen_dataset[dataset] += 1
                seen_lineage[lineage] += 1
                meta = {
                    "id": alpha_id,
                    "dataset_id": dataset,
                    "field_id": field,
                    "lineage_theme": lineage,
                    "template": template,
                    "submitted_similarity_max": sim.get("max_similarity"),
                    "nearest_submitted_alpha": sim.get("nearest_submitted_alpha"),
                    "created_at": now_iso(),
                }
                state.setdefault("alphas", {})[alpha_id] = meta
                added.append(meta)
                if len(added) >= max_add:
                    break
            if len(added) >= max_add:
                break
    event = {
        "timestamp_utc": now_iso(),
        "requested_max_add": max_add,
        "field_limit": field_limit,
        "submitted_threshold": threshold,
        "fields_considered": len(ranked),
        "candidates_checked": checked,
        "added": added,
        "skipped": dict(skipped),
        "selected_lineages": dict(Counter(x["lineage_theme"] for x in added)),
        "selected_datasets": dict(Counter(x["dataset_id"] for x in added)),
    }
    state.setdefault("history", []).append(event)
    state["history"] = state["history"][-100:]
    save_state(state)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Fresh Low-Collision Supply — {event['timestamp_utc']}",
        "",
        f"- added: `{len(added)}`",
        f"- candidates_checked: `{checked}`",
        f"- submitted_threshold: `{threshold}`",
        f"- skipped: `{dict(skipped)}`",
        "",
        "## Added",
    ]
    lines.extend(f"- {x['id']} field={x['field_id']} lineage={x['lineage_theme']} sim={x['submitted_similarity_max']} nearest={x['nearest_submitted_alpha']} template={x['template']}" for x in added)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return event


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate fresh low-submitted-similarity WorldQuant alpha supply")
    parser.add_argument("--max-add", type=int, default=8)
    parser.add_argument("--field-limit", type=int, default=80)
    parser.add_argument("--submitted-threshold", type=float, default=SUBMITTED_THRESHOLD)
    args = parser.parse_args()
    event = generate(args.max_add, args.field_limit, args.submitted_threshold)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
