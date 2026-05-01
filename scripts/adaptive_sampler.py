#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIELDS_CSV = ROOT / "data" / "data_fields_scored.csv"
STATE_PATH = ROOT / "state" / "adaptive_sampler_state.json"
ALPHAS_CSV = ROOT / "alphas.csv"
DB_PATH = ROOT / "data" / "backtests.sqlite3"
UNDER_SAMPLED_LINEAGES = {"fundamental_valuation", "market_size", "price_volume", "liquidity_volatility"}

DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 6,
    "neutralization": "INDUSTRY",
    "truncation": 0.04,
}

FUNCTION_FAMILIES = [
    "rank_relative",
    "zscore_residual",
    "delta_shock",
    "corr_anticorr",
    "neutralized_residual",
    "gated_activation",
    "composite_helper",
]

COMPARATORS = ["returns", "cap", "adv20", "volume", "ts_std_dev(returns, 120)"]
WINDOWS = [60, 80, 120, 160, 252]
SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LOW_VOL_GATE = "ts_rank(ts_std_dev(returns, 10), 252) < 0.8"
LIQUID_GATE = "rank(adv20) > 0.2"
LIQUID_EXIT = "rank(adv20) < 0.05"
OVERNIGHT = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"


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
        return {"version": 1, "created_at": now_iso(), "arms": {}, "alphas": {}, "history": []}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_fields(path: Path = FIELDS_CSV) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("region") and row.get("region") != "USA":
                continue
            if row.get("universe") and row.get("universe") != "TOP3000":
                continue
            if row.get("type") and row.get("type") not in {"MATRIX", "VECTOR"}:
                continue
            coverage = to_float(row.get("coverage"))
            if coverage < 0.35:
                continue
            rows.append(row)
    return rows


def field_prior(row: dict[str, Any]) -> float:
    # User requested high coverage / high alpha-count fields as priors.  Keep the
    # alphaCount boost logarithmic so crowded fields can be sampled without
    # letting popularity dominate observed value density.
    base = to_float(row.get("score"))
    coverage = to_float(row.get("coverage"))
    date_coverage = to_float(row.get("dateCoverage"))
    alpha_count = to_int(row.get("alphaCount"))
    user_count = to_int(row.get("userCount"))
    typ = 4 if row.get("type") == "MATRIX" else -2
    crowd_penalty = min(10.0, math.log1p(user_count) * 0.7)
    return round(base + 30 * coverage + 8 * date_coverage + 4 * math.log1p(alpha_count) + typ - crowd_penalty, 4)


def classify_lineage_theme(row: dict[str, Any]) -> str:
    text = " ".join(str(row.get(k) or "").lower() for k in ["dataset_id", "dataset_name", "id", "category"])
    if any(token in text for token in ["analyst", "anl", "eps", "earning", "revision", "estimate", "rating"]):
        return "analyst_earnings"
    if any(token in text for token in ["asset", "book", "cash", "debt", "enterprise", "fundamental", "liabil", "margin", "revenue", "sales", "value"]):
        return "fundamental_valuation"
    if any(token in text for token in ["cap", "marketcap", "size"]):
        return "market_size"
    if any(token in text for token in ["adv", "turnover", "liquidity", "volatility", "spread", "volume"]):
        return "liquidity_volatility"
    if any(token in text for token in ["close", "open", "return", "price", "vwap", "high", "low"]):
        return "price_volume"
    return "unknown"


def arm_key(row: dict[str, Any], family: str) -> str:
    return "|".join([row.get("dataset_id", ""), row.get("id", ""), family])


def expressions_for(field: str, family: str) -> list[str]:
    exprs: list[str] = []
    if family == "rank_relative":
        for comp in COMPARATORS[:4]:
            exprs.append(f"rank(ts_rank({field}, 120) - ts_rank({comp}, 120))")
            exprs.append(f"rank(ts_rank({field}, 252) - ts_rank({comp}, 120))")
    elif family == "zscore_residual":
        for comp in ["cap", "returns", "adv20"]:
            exprs.append(f"rank(ts_zscore({field}, 120) - ts_zscore({comp}, 120))")
            exprs.append(f"group_neutralize(rank(ts_zscore({field}, 120) - ts_zscore({comp}, 120)), {SIZE_BUCKET})")
    elif family == "delta_shock":
        for mean_w, delta_w in [(10, 60), (20, 60), (20, 120)]:
            exprs.append(f"rank(ts_delta(ts_mean({field}, {mean_w}), {delta_w}))")
            exprs.append(f"group_neutralize(rank(ts_delta(ts_mean({field}, {mean_w}), {delta_w})), subindustry)")
    elif family == "corr_anticorr":
        for comp, win in [("returns", 120), ("volume", 120), ("adv20", 120), ("vwap", 252)]:
            exprs.append(f"rank(ts_corr(ts_rank({field}, {win}), ts_rank({comp}, {win}), {win}))")
            exprs.append(f"-rank(ts_corr(ts_rank({field}, {win}), ts_rank({comp}, {win}), {win}))")
    elif family == "neutralized_residual":
        market_ret = "group_mean(returns, 1, market)"
        exprs.append(f"group_neutralize(rank(ts_rank({field}, 120) - ts_rank(returns, 252)), {SIZE_BUCKET})")
        exprs.append(f"group_neutralize(rank(ts_zscore({field}, 120) - ts_zscore(cap, 120)), subindustry)")
        exprs.append(f"group_neutralize(rank({field} - rank(ts_std_dev(returns, 30))), {SIZE_BUCKET})")
        exprs.append(f"group_neutralize(rank(ts_rank({field}, 120) - rank(ts_mean(returns - {market_ret}, 20))), {SIZE_BUCKET})")
    elif family == "gated_activation":
        core = f"rank(ts_rank({field}, 120) - ts_rank(returns, 252))"
        exprs.append(f"trade_when({LIQUID_GATE}, group_neutralize({core}, {SIZE_BUCKET}), {LIQUID_EXIT})")
        exprs.append(f"trade_when({LOW_VOL_GATE}, group_neutralize({core}, subindustry), -1)")
        exprs.append(f"trade_when(rank({field}) > 0.7, group_neutralize({core}, {SIZE_BUCKET}), -1)")
    elif family == "composite_helper":
        core = f"rank(ts_rank({field}, 120) - ts_rank(returns, 252))"
        helper1 = f"rank(-ts_corr(abs({OVERNIGHT}), ts_delay(volume / adv20, 1), 20))"
        helper2 = "rank(ts_corr(vwap, open, 252))"
        helper3 = "rank(ts_std_dev(volume / adv20, 30))"
        exprs.append(f"group_neutralize(rank({core} + 0.2 * {helper1}), {SIZE_BUCKET})")
        exprs.append(f"rank({core} + 0.2 * {helper2})")
        exprs.append(f"group_neutralize(rank({core} - {helper3}), {SIZE_BUCKET})")
    return exprs


def reward_from_stats(stats: dict[str, Any]) -> float:
    tested = max(1, int(stats.get("tested", 0)))
    pass_count = int(stats.get("pass", 0))
    strong = int(stats.get("strong", 0))
    near = int(stats.get("near_miss", 0))
    low_fit = int(stats.get("low_fitness", 0))
    low_sub = int(stats.get("low_sub_universe", 0))
    errors = int(stats.get("errors", 0))
    self_corr = int(stats.get("self_corr", 0))
    avg_f = float(stats.get("avg_fitness", 0) or 0)
    avg_s = float(stats.get("avg_sharpe", 0) or 0)
    return round(
        10 * pass_count + 6 * strong + 2.5 * near + 2 * avg_f + avg_s
        - 2.5 * low_sub / tested - 2.0 * low_fit / tested - 1.5 * errors / tested - 1.0 * self_corr / tested,
        4,
    )


def refresh_arm_stats(state: dict[str, Any]) -> None:
    if not DB_PATH.exists() or not state.get("alphas"):
        return
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    by_arm: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for alpha_id, meta in state.get("alphas", {}).items():
        row = conn.execute("select * from backtest_results where alpha_id=?", (alpha_id,)).fetchone()
        if row:
            by_arm[meta["arm_key"]].append(row)
    conn.close()

    for key, rows in by_arm.items():
        if key not in state["arms"]:
            continue
        tested = len(rows)
        sharpes = [to_float(r["sharpe"]) for r in rows if r["sharpe"] is not None]
        fitnesses = [to_float(r["fitness"]) for r in rows if r["fitness"] is not None]
        fail_text = "\n".join(str(r["fail_reasons"] or "") for r in rows)
        stats = {
            "tested": tested,
            "pass": sum(1 for r in rows if to_float(r["sharpe"]) >= 1.6 and to_float(r["fitness"]) >= 1.0 and to_float(r["turnover"], 99) <= 0.45),
            "strong": sum(1 for r in rows if to_float(r["sharpe"]) >= 1.8 and to_float(r["fitness"]) >= 1.15 and to_float(r["turnover"], 99) <= 0.30),
            "near_miss": sum(1 for r in rows if to_float(r["sharpe"]) >= 1.35 and to_float(r["fitness"]) >= 0.85 and to_float(r["turnover"], 99) <= 0.55),
            "avg_sharpe": round(sum(sharpes) / len(sharpes), 4) if sharpes else 0.0,
            "avg_fitness": round(sum(fitnesses) / len(fitnesses), 4) if fitnesses else 0.0,
            "low_fitness": fail_text.count("LOW_FITNESS"),
            "low_sub_universe": fail_text.count("LOW_SUB_UNIVERSE_SHARPE"),
            "self_corr": fail_text.count("SELF_CORRELATION"),
            "high_turnover": fail_text.count("HIGH_TURNOVER"),
            "errors": sum(1 for r in rows if str(r["status"] or "").upper() == "ERROR" or r["error"]),
        }
        stats["reward"] = reward_from_stats(stats)
        arm = state["arms"][key]
        arm["stats"] = stats
        if tested >= 6 and stats["pass"] == 0 and stats["near_miss"] == 0 and (stats["avg_fitness"] < 0.7 or stats["low_fitness"] >= max(4, tested - 1)):
            arm["status"] = "paused_low_value"
        elif tested >= 6 and stats["pass"] == 0 and stats["near_miss"] > 0 and (stats["self_corr"] >= max(3, tested // 2) or stats["high_turnover"] >= max(3, tested // 2)):
            arm["status"] = "paused_stuck_blocked"
        elif stats["pass"] > 0 or stats["strong"] > 0:
            arm["status"] = "promoted"
        elif stats["near_miss"] > 0:
            arm["status"] = "active_near_miss"
        else:
            arm.setdefault("status", "active")


def ensure_arms(state: dict[str, Any], fields: list[dict[str, Any]], max_new_fields: int = 80) -> None:
    ranked = sorted(fields, key=field_prior, reverse=True)[:max_new_fields]
    for row in ranked:
        for family in FUNCTION_FAMILIES:
            key = arm_key(row, family)
            if key in state["arms"]:
                continue
            state["arms"][key] = {
                "dataset_id": row.get("dataset_id"),
                "dataset_name": row.get("dataset_name"),
                "field_id": row.get("id"),
                "field_type": row.get("type"),
                "category": row.get("category"),
                "coverage": to_float(row.get("coverage")),
                "alpha_count": to_int(row.get("alphaCount")),
                "user_count": to_int(row.get("userCount")),
                "field_score": to_float(row.get("score")),
                "lineage_theme": classify_lineage_theme(row),
                "prior": field_prior(row),
                "function_family": family,
                "status": "active",
                "stats": {},
                "created_at": now_iso(),
            }


def select_arms(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    arms = list(state.get("arms", {}).items())
    candidates = []
    lineage_diversity_bonus = float(os.environ.get("ADAPTIVE_LINEAGE_DIVERSITY_BONUS", "3.0"))
    for key, arm in arms:
        if str(arm.get("status", "active")).startswith("paused"):
            continue
        stats = arm.get("stats") or {}
        generated = int(arm.get("generated", 0))
        tested = int(stats.get("tested", 0) or 0)
        reward = float(stats.get("reward", 0) or 0)
        prior = float(arm.get("prior", 0) or 0)
        # UCB-ish score: promoted/near-miss arms get exploitation; untested
        # high-coverage/high-alpha fields keep exploration pressure.
        exploration = 6.0 / math.sqrt(1 + generated)
        score = prior / 20 + reward + exploration
        lineage_theme = str(arm.get("lineage_theme") or "unknown")
        if lineage_theme in UNDER_SAMPLED_LINEAGES or lineage_theme == "unknown":
            score += lineage_diversity_bonus
            arm["lineage_diversity_bonus"] = lineage_diversity_bonus
        if arm.get("status") == "promoted":
            score += 8
        elif arm.get("status") == "active_near_miss":
            score += 4
        if tested >= 6 and reward < 0:
            score -= 6
        candidates.append((score, key, arm))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    seen_dataset = defaultdict(int)
    seen_family = defaultdict(int)
    for _, key, arm in candidates:
        # Keep each small batch diversified across datasets and function families.
        if seen_dataset[arm.get("dataset_id")] >= 2:
            continue
        if seen_family[arm.get("function_family")] >= 2:
            continue
        chosen.append({"key": key, **arm})
        seen_dataset[arm.get("dataset_id")] += 1
        seen_family[arm.get("function_family")] += 1
        if len(chosen) >= limit:
            break
    return chosen


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


def append_candidates(state: dict[str, Any], arms: list[dict[str, Any]], max_add: int) -> list[dict[str, Any]]:
    existing_exprs, existing_ids = existing_csv()
    ALPHAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for arm in arms:
            field = arm["field_id"]
            family = arm["function_family"]
            for expr in expressions_for(field, family):
                expr = expr.strip()
                if not expr or expr in existing_exprs:
                    continue
                digest = hashlib.sha1((arm["key"] + "|" + expr).encode("utf-8")).hexdigest()[:10]
                alpha_id = f"arm_{digest}"
                if alpha_id in existing_ids:
                    continue
                writer.writerow({"id": alpha_id, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(alpha_id)
                state["alphas"][alpha_id] = {
                    "arm_key": arm["key"],
                    "dataset_id": arm.get("dataset_id"),
                    "dataset_name": arm.get("dataset_name"),
                    "field_id": field,
                    "function_family": family,
                    "expression": expr,
                    "created_at": now_iso(),
                }
                state["arms"][arm["key"]]["generated"] = int(state["arms"][arm["key"]].get("generated", 0)) + 1
                added.append({"id": alpha_id, "arm_key": arm["key"], "field": field, "function_family": family})
                if len(added) >= max_add:
                    return added
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive dataset/field/function-family sampler for WorldQuant alphas")
    parser.add_argument("--max-add", type=int, default=6)
    parser.add_argument("--arm-limit", type=int, default=8)
    args = parser.parse_args()

    state = load_state()
    fields = load_fields()
    ensure_arms(state, fields)
    refresh_arm_stats(state)
    selected = select_arms(state, args.arm_limit)
    added = append_candidates(state, selected, args.max_add)
    event = {
        "timestamp_utc": now_iso(),
        "selected_arms": [a["key"] for a in selected],
        "added": added,
        "active_arms": sum(1 for a in state["arms"].values() if not str(a.get("status", "")).startswith("paused")),
        "paused_arms": sum(1 for a in state["arms"].values() if str(a.get("status", "")).startswith("paused")),
    }
    state.setdefault("history", []).append(event)
    state["history"] = state["history"][-100:]
    save_state(state)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
