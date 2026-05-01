#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIELDS_CSV = ROOT / "data" / "data_fields_scored.csv"
DB_PATH = ROOT / "data" / "backtests.sqlite3"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "multi_dataset_state.json"
TRUTH_TABLE_PATH = ROOT / "state" / "self_corr_truth_table.json"

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

INTERACTION_TYPES = [
    "rank_spread",
    "zscore_spread",
    "conditional_gate",
    "residual_helper",
    "regime_interaction",
    "neutralized_product",
    # Broader function-family interactions: the first multi_ smoke test was too
    # narrow (mostly rank/zscore spreads).  Treat transform family as a first
    # class arm so weak early multi_ results do not over-generalize from one
    # operator style.
    "correlation_residual",
    "delta_decay_combo",
    "volatility_scaled",
    "group_rank_contrast",
    "nonlinear_squash",
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
        return {"version": 1, "created_at": now_iso(), "arms": {}, "alphas": {}, "history": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "created_at": now_iso(), "arms": {}, "alphas": {}, "history": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_fields() -> list[dict[str, Any]]:
    if not FIELDS_CSV.exists():
        return []
    rows: list[dict[str, Any]] = []
    with FIELDS_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("region") and row.get("region") != "USA":
                continue
            if row.get("universe") and row.get("universe") != "TOP3000":
                continue
            if row.get("type") and row.get("type") not in {"MATRIX", "VECTOR"}:
                continue
            if to_float(row.get("coverage")) < 0.35:
                continue
            rows.append(row)
    return rows


def field_prior(row: dict[str, Any]) -> float:
    score = to_float(row.get("score"))
    coverage = to_float(row.get("coverage"))
    date_cov = to_float(row.get("dateCoverage"))
    alpha_count = to_int(row.get("alphaCount"))
    user_count = to_int(row.get("userCount"))
    matrix_bonus = 4 if row.get("type") == "MATRIX" else -1
    crowd_penalty = min(9.0, math.log1p(user_count) * 0.65)
    return score + 28 * coverage + 7 * date_cov + 3.5 * math.log1p(alpha_count) + matrix_bonus - crowd_penalty


def category_bucket(row: dict[str, Any]) -> str:
    return str(row.get("category") or row.get("subcategory") or row.get("dataset_id") or "unknown")


def arm_theme(arm: dict[str, Any]) -> str:
    text = " ".join(
        str(arm.get(k) or "").lower()
        for k in ("primary_dataset", "secondary_dataset", "primary_field", "secondary_field", "primary_category", "secondary_category")
    )
    if any(x in text for x in ("analyst", "anl", "eps", "earnings", "revision", "surprise", "estimate")):
        return "analyst_earnings"
    if any(x in text for x in ("adv20", "volume", "turnover", "liquidity", "volatility", "std")):
        return "liquidity_volatility"
    if any(x in text for x in ("assets", "debt", "sales", "revenue", "book", "enterprise", "fundamental", "value")):
        return "fundamental_valuation"
    if any(x in text for x in ("cap", "marketcap", "size")):
        return "market_size"
    if any(x in text for x in ("pv1", "close", "open", "vwap", "returns", "price")):
        return "price_volume"
    return "unknown"


def under_sampled_lineage_themes() -> set[str]:
    summary = load_json(TRUTH_TABLE_PATH).get("summary") or {}
    counts = summary.get("lineage_theme_counts") or {}
    pass_total = sum(int(v.get("pass_quality", 0) or 0) for v in counts.values() if isinstance(v, dict))
    if pass_total <= 0:
        return {"price_volume", "liquidity_volatility", "fundamental_valuation", "market_size", "mixed", "unknown"}
    out: set[str] = set()
    for theme, stats in counts.items():
        if not isinstance(stats, dict):
            continue
        pass_count = int(stats.get("pass_quality", 0) or 0)
        if pass_count <= 2 or pass_count / pass_total <= 0.12:
            out.add(str(theme))
    out.update({"fundamental_valuation", "liquidity_volatility", "market_size"})
    return out


def recent_multi_quality() -> dict[str, int]:
    if not DB_PATH.exists():
        return {"tested": 0, "pass_quality": 0}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select sharpe, fitness, turnover
        from backtest_results
        where alpha_id like 'multi_%'
        order by datetime(created_at) desc
        limit 20
        """
    ).fetchall()
    conn.close()
    return {
        "tested": len(rows),
        "pass_quality": sum(1 for r in rows if to_float(r["sharpe"]) >= 1.6 and to_float(r["fitness"]) >= 1.0 and to_float(r["turnover"], 99) <= 0.45),
    }


def arm_key(a: dict[str, Any], b: dict[str, Any], interaction: str) -> str:
    return "|".join([a.get("dataset_id", ""), a.get("id", ""), b.get("dataset_id", ""), b.get("id", ""), interaction])


def pair_prior(a: dict[str, Any], b: dict[str, Any]) -> float:
    diversity = 14.0 if a.get("dataset_id") != b.get("dataset_id") else -8.0
    category_diversity = 6.0 if category_bucket(a) != category_bucket(b) else 0.0
    coverage = 12.0 * min(to_float(a.get("coverage")), to_float(b.get("coverage")))
    popularity = 2.0 * math.log1p(to_int(a.get("alphaCount")) + to_int(b.get("alphaCount")))
    return field_prior(a) / 25 + field_prior(b) / 25 + diversity + category_diversity + coverage + popularity


def expressions_for(field_a: str, field_b: str, interaction: str) -> list[str]:
    a = field_a
    b = field_b
    if interaction == "rank_spread":
        return [
            f"group_neutralize(rank(ts_rank({a}, 120) - ts_rank({b}, 120)), subindustry)",
            f"rank(ts_rank({a}, 252) - ts_rank({b}, 120))",
            f"group_neutralize(rank(ts_rank({a}, 80) - ts_rank({b}, 252)), {SIZE_BUCKET})",
        ]
    if interaction == "zscore_spread":
        return [
            f"group_neutralize(rank(ts_zscore({a}, 120) - ts_zscore({b}, 120)), subindustry)",
            f"rank(ts_zscore({a}, 80) - ts_zscore({b}, 160))",
            f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_zscore({a}, 120) - ts_zscore({b}, 120)), {SIZE_BUCKET}), {LIQUID_EXIT})",
        ]
    if interaction == "conditional_gate":
        signal = f"group_neutralize(rank(ts_rank({a}, 120) - ts_rank(returns, 252)), subindustry)"
        return [
            f"trade_when(rank({b}) > 0.55, {signal}, -1)",
            f"trade_when(ts_rank({b}, 120) > 0.6, {signal}, -1)",
            f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_rank({a}, 120) - ts_rank({b}, 120)), subindustry), -1)",
        ]
    if interaction == "residual_helper":
        core = f"rank(ts_rank({a}, 120) - ts_rank(returns, 252))"
        helper = f"rank(ts_rank({b}, 120) - ts_rank(cap, 120))"
        return [
            f"group_neutralize(rank({core} + 0.25 * {helper}), subindustry)",
            f"group_neutralize(rank({core} - 0.25 * {helper}), {SIZE_BUCKET})",
            f"rank(ts_zscore({a}, 120) - 0.35 * ts_zscore({b}, 120))",
        ]
    if interaction == "regime_interaction":
        return [
            f"group_neutralize(rank(ts_rank({a}, 120)) * rank(ts_rank({b}, 120)), subindustry)",
            f"rank(ts_rank({a}, 120) * rank(ts_std_dev({b}, 60)))",
            f"trade_when(rank(ts_std_dev({b}, 60)) < 0.8, group_neutralize(rank(ts_rank({a}, 120)), {SIZE_BUCKET}), -1)",
        ]
    if interaction == "neutralized_product":
        return [
            f"group_neutralize(rank({a}) * rank({b}), subindustry)",
            f"group_neutralize(rank(ts_zscore({a}, 120) * ts_zscore({b}, 120)), {SIZE_BUCKET})",
            f"rank(group_neutralize(rank({a}) * rank({b}), subindustry) - rank(ts_std_dev(returns, 30)))",
        ]
    if interaction == "correlation_residual":
        corr = f"ts_corr(rank({a}), rank({b}), 120)"
        return [
            f"group_neutralize(rank(ts_zscore({a}, 120) - 0.30 * rank({corr})), subindustry)",
            f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_zscore({a}, 80) - 0.25 * ts_zscore({b}, 160)), {SIZE_BUCKET}), {LIQUID_EXIT})",
            f"rank(ts_corr(rank({a}), rank(returns), 120) - ts_corr(rank({b}), rank(returns), 120))",
        ]
    if interaction == "delta_decay_combo":
        return [
            f"group_neutralize(rank(ts_decay_linear(ts_delta({a}, 5), 20) - ts_decay_linear(ts_delta({b}, 5), 20)), subindustry)",
            f"rank(ts_delta(ts_mean({a}, 20), 10) - ts_delta(ts_mean({b}, 40), 10))",
            f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_decay_linear(ts_zscore({a}, 60), 12) - ts_decay_linear(ts_zscore({b}, 120), 24)), {SIZE_BUCKET}), -1)",
        ]
    if interaction == "volatility_scaled":
        return [
            f"group_neutralize(rank((ts_zscore({a}, 120) - ts_zscore({b}, 120)) / (1 + ts_std_dev(returns, 20))), subindustry)",
            f"trade_when(rank(ts_std_dev(returns, 20)) < 0.75, group_neutralize(rank(ts_rank({a}, 120) - ts_rank({b}, 120)), {SIZE_BUCKET}), -1)",
            f"rank((ts_rank({a}, 80) - ts_rank({b}, 160)) * (1 - rank(ts_std_dev(returns, 30))))",
        ]
    if interaction == "group_rank_contrast":
        return [
            f"rank(group_neutralize(rank({a}), subindustry) - group_neutralize(rank({b}), subindustry))",
            f"group_neutralize(rank(group_neutralize(rank({a}), {SIZE_BUCKET}) - group_neutralize(rank({b}), subindustry)), subindustry)",
            f"trade_when({LIQUID_GATE}, rank(group_neutralize(ts_rank({a}, 120), subindustry) - group_neutralize(ts_rank({b}, 120), {SIZE_BUCKET})), {LIQUID_EXIT})",
        ]
    if interaction == "nonlinear_squash":
        return [
            f"group_neutralize(rank(rank(ts_zscore({a}, 120)) * (2 * rank(ts_zscore({b}, 120)) - 1)), subindustry)",
            f"rank((rank(ts_rank({a}, 120)) - 0.5) * (rank(ts_rank({b}, 120)) - 0.5))",
            f"trade_when(rank({b}) > 0.6, group_neutralize(rank(ts_zscore({a}, 120) * rank(ts_delta({b}, 10))), {SIZE_BUCKET}), -1)",
        ]
    return []


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


def refresh_stats(state: dict[str, Any]) -> None:
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
        if key not in state.get("arms", {}):
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
            "self_corr": fail_text.count("SELF_CORRELATION"),
            "low_fitness": fail_text.count("LOW_FITNESS"),
            "low_sub_universe": fail_text.count("LOW_SUB_UNIVERSE_SHARPE"),
            "errors": sum(1 for r in rows if str(r["status"] or "").upper() == "ERROR" or r["error"]),
        }
        reward = 12 * stats["pass"] + 7 * stats["strong"] + 3 * stats["near_miss"] + 2 * stats["avg_fitness"] + stats["avg_sharpe"]
        reward -= 2.0 * stats["low_fitness"] / max(1, tested) + 1.5 * stats["self_corr"] / max(1, tested)
        stats["reward"] = round(reward, 4)
        arm = state["arms"][key]
        arm["stats"] = stats
        if tested >= 8 and stats["pass"] == 0 and stats["near_miss"] == 0 and stats["avg_fitness"] < 0.6:
            arm["status"] = "paused_low_value"
        elif tested >= 10 and stats["self_corr"] >= max(6, tested // 2):
            arm["status"] = "paused_self_corr_heavy"
        elif stats["pass"] or stats["strong"]:
            arm["status"] = "promoted"
        elif stats["near_miss"]:
            arm["status"] = "active_near_miss"
        else:
            arm.setdefault("status", "active")


def ensure_arms(state: dict[str, Any], fields: list[dict[str, Any]], max_fields: int = 80) -> None:
    ranked = sorted(fields, key=field_prior, reverse=True)[:max_fields]
    # Pair high-prior fields across different datasets/categories. Limit pair explosion.
    pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 : min(len(ranked), i + 26)]:
            if a.get("id") == b.get("id"):
                continue
            if a.get("dataset_id") == b.get("dataset_id") and category_bucket(a) == category_bucket(b):
                continue
            pairs.append((pair_prior(a, b), a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    for prior, a, b in pairs[:220]:
        for interaction in INTERACTION_TYPES:
            key = arm_key(a, b, interaction)
            if key in state["arms"]:
                continue
            state["arms"][key] = {
                "primary_dataset": a.get("dataset_id"),
                "primary_field": a.get("id"),
                "primary_category": category_bucket(a),
                "secondary_dataset": b.get("dataset_id"),
                "secondary_field": b.get("id"),
                "secondary_category": category_bucket(b),
                "interaction_type": interaction,
                "prior": round(prior, 4),
                "status": "active",
                "stats": {},
                "created_at": now_iso(),
            }


def select_arms(state: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    candidates = []
    for key, arm in state.get("arms", {}).items():
        if str(arm.get("status", "active")).startswith("paused"):
            continue
        generated = int(arm.get("generated", 0) or 0)
        stats = arm.get("stats") or {}
        reward = float(stats.get("reward", 0) or 0)
        prior = float(arm.get("prior", 0) or 0)
        score = prior / 10 + reward + 5 / math.sqrt(1 + generated)
        if arm.get("status") == "promoted":
            score += 8
        elif arm.get("status") == "active_near_miss":
            score += 4
        candidates.append((score, key, arm))
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    seen_dataset = Counter()
    seen_interaction = Counter()
    for _, key, arm in candidates:
        dkey = tuple(sorted([arm.get("primary_dataset"), arm.get("secondary_dataset")]))
        if seen_dataset[dkey] >= 2:
            continue
        if seen_interaction[arm.get("interaction_type")] >= 2:
            continue
        chosen.append({"key": key, **arm})
        seen_dataset[dkey] += 1
        seen_interaction[arm.get("interaction_type")] += 1
        if len(chosen) >= limit:
            break
    return chosen


def append_candidates(state: dict[str, Any], arms: list[dict[str, Any]], max_add: int) -> list[dict[str, Any]]:
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for arm in arms:
            for expr in expressions_for(arm["primary_field"], arm["secondary_field"], arm["interaction_type"]):
                expr = expr.strip()
                if not expr or expr in existing_exprs:
                    continue
                digest = hashlib.sha1((arm["key"] + "|" + expr).encode("utf-8")).hexdigest()[:10]
                alpha_id = f"multi_{digest}"
                if alpha_id in existing_ids:
                    continue
                writer.writerow({"id": alpha_id, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(alpha_id)
                state["alphas"][alpha_id] = {
                    "track": "multi",
                    "arm_key": arm["key"],
                    "datasets": [arm.get("primary_dataset"), arm.get("secondary_dataset")],
                    "fields": [arm.get("primary_field"), arm.get("secondary_field")],
                    "interaction_type": arm.get("interaction_type"),
                    "expression": expr,
                    "created_at": now_iso(),
                }
                state["arms"][arm["key"]]["generated"] = int(state["arms"][arm["key"]].get("generated", 0)) + 1
                added.append({"id": alpha_id, "arm_key": arm["key"], "interaction_type": arm.get("interaction_type")})
                if len(added) >= max_add:
                    return added
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate multi-dataset WorldQuant alpha candidates")
    parser.add_argument("--max-add", type=int, default=4)
    parser.add_argument("--arm-limit", type=int, default=6)
    args = parser.parse_args()

    state = load_state()
    fields = load_fields()
    ensure_arms(state, fields)
    refresh_stats(state)
    selected = select_arms(state, args.arm_limit)
    quality = recent_multi_quality()
    under_sampled = under_sampled_lineage_themes()
    selected_under_sampled = any(arm_theme(a) in under_sampled for a in selected)
    effective_max_add = args.max_add
    throttle_reason = ""
    if quality["tested"] >= 8 and quality["pass_quality"] == 0 and not selected_under_sampled:
        effective_max_add = min(effective_max_add, 1)
        throttle_reason = "recent multi_ has zero pass-quality and selected arms are not under-sampled lineage"
    added = append_candidates(state, selected, effective_max_add)
    event = {
        "timestamp_utc": now_iso(),
        "selected_arms": [a["key"] for a in selected],
        "added": added,
        "requested_max_add": args.max_add,
        "effective_max_add": effective_max_add,
        "recent_multi_quality": quality,
        "selected_under_sampled_lineage": selected_under_sampled,
        "throttle_reason": throttle_reason,
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
