#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from scripts.submitted_similarity import load_submitted_features, score_against_submitted  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "superalpha_state.json"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
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
UNDER_SAMPLED_LINEAGES = {"fundamental_valuation", "market_size", "price_volume", "liquidity_volatility"}
MAX_SUBMITTED_SIMILARITY_GENERATE = 0.65


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "created_at": now_iso(), "built_from": {}, "history": [], "lineage_pair_blacklist": {}}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        state.setdefault("lineage_pair_blacklist", {})
        return state
    except Exception:
        return {"version": 1, "created_at": now_iso(), "built_from": {}, "history": [], "lineage_pair_blacklist": {}}


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


def blocked_ids_from_log(limit: int = 80) -> set[str]:
    if not AUTO_SUBMIT_LOG.exists():
        return set()
    counts: Counter[str] = Counter()
    for line in AUTO_SUBMIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-800:]:
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
            if "SELF_CORRELATION" in reason or "self-correlation cooldown active" in reason:
                counts[alpha_id] += 1
    blocked = {alpha_id for alpha_id, _ in counts.most_common(limit)}
    truth = load_json(TRUTH_TABLE_PATH)
    truth_rows = truth.get("alphas") or []
    truth_ranked: list[tuple[int, str]] = []
    for row in truth_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("self_corr_status") or "") not in {"blocked", "pending", "cooldown"}:
            continue
        skip_counts = row.get("skip_counts") or {}
        total = sum(int(v) for v in skip_counts.values()) if isinstance(skip_counts, dict) else 0
        truth_ranked.append((total, str(row.get("alpha_id") or "")))
    truth_ranked.sort(reverse=True)
    blocked.update(alpha_id for _, alpha_id in truth_ranked[:limit] if alpha_id)
    return blocked


def truth_lineage_index() -> dict[str, str]:
    truth = load_json(TRUTH_TABLE_PATH)
    out: dict[str, str] = {}
    for row in truth.get("alphas") or []:
        if not isinstance(row, dict):
            continue
        alpha_id = str(row.get("alpha_id") or "")
        lineage = row.get("lineage") or {}
        theme = str(lineage.get("theme") or "unknown")
        if alpha_id:
            out[alpha_id] = theme
    return out


def lineage_pair_key(a_theme: str, b_theme: str) -> str:
    return "+".join(sorted([a_theme or "unknown", b_theme or "unknown"]))


def lineage_pair_priority(a_theme: str, b_theme: str) -> int:
    a_under = a_theme in UNDER_SAMPLED_LINEAGES
    b_under = b_theme in UNDER_SAMPLED_LINEAGES
    if a_under and b_under:
        return 2
    if a_under or b_under:
        return 1
    return 0


def refresh_lineage_pair_blacklist(state: dict[str, Any], lineage: dict[str, str]) -> None:
    truth = load_json(TRUTH_TABLE_PATH)
    blocked_super = {
        str(row.get("alpha_id"))
        for row in truth.get("alphas") or []
        if isinstance(row, dict)
        and str(row.get("alpha_id") or "").startswith("super_")
        and str(row.get("self_corr_status") or "") == "blocked"
    }
    blacklist: Counter[str] = Counter()
    for alpha_id in blocked_super:
        meta = (state.get("built_from") or {}).get(alpha_id) or {}
        parent_ids = [str(x) for x in meta.get("parent_alpha_ids") or [] if x]
        if len(parent_ids) < 2:
            continue
        themes = [lineage.get(parent_id, "unknown") for parent_id in parent_ids[:2]]
        blacklist[lineage_pair_key(themes[0], themes[1])] += 1
    state["lineage_pair_blacklist"] = dict(blacklist)


def family(alpha_id: str) -> str:
    return alpha_id.split("_", 1)[0] + "_" if "_" in alpha_id else alpha_id


def source_candidates(conn: sqlite3.Connection, limit: int = 80) -> list[sqlite3.Row]:
    blocked = blocked_ids_from_log()
    rows = conn.execute(
        f'''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons, created_at
        from backtest_results
        where sharpe >= 1.2
          and fitness >= 0.7
          and turnover <= 0.55
          and expression is not null
          and not ({exact_prefix_clause('alpha_id')})
        order by datetime(created_at) desc
        limit 300
        ''',
        (exact_prefix_param("super_"),),
    ).fetchall()
    def score(row: sqlite3.Row) -> float:
        aid = str(row["alpha_id"])
        s = 3 * to_float(row["sharpe"]) + 4 * to_float(row["fitness"]) - 1.5 * to_float(row["turnover"])
        if aid in blocked:
            s += 5.0
        fam = family(aid)
        if fam in {"multi_", "arm_"}:
            s += 2.0
        if fam in {"repairsc2_", "repairsc3_", "decor_"}:
            s += 1.0
        return s
    rows = sorted(rows, key=score, reverse=True)
    return rows[:limit]


def pair_score(a: sqlite3.Row, b: sqlite3.Row, blocked: set[str], lineage: dict[str, str] | None = None) -> float:
    af, bf = family(str(a["alpha_id"])), family(str(b["alpha_id"]))
    diversity = 5.0 if af != bf else -5.0
    lineage = lineage or {}
    at, bt = lineage.get(str(a["alpha_id"]), "unknown"), lineage.get(str(b["alpha_id"]), "unknown")
    lineage_penalty = 4.0 if at == bt and at not in {"", "unknown"} else 0.0
    lineage_priority_bonus = 45.0 * lineage_pair_priority(at, bt)
    blocked_bonus = 4.0 if str(a["alpha_id"]) in blocked or str(b["alpha_id"]) in blocked else 0.0
    quality = to_float(a["sharpe"]) + to_float(b["sharpe"]) + 1.5 * (to_float(a["fitness"]) + to_float(b["fitness"]))
    turnover_penalty = to_float(a["turnover"]) + to_float(b["turnover"])
    return quality + diversity + blocked_bonus + lineage_priority_bonus - turnover_penalty - lineage_penalty


def submitted_collision_meta(expr: str, submitted_features: list[dict[str, Any]], threshold: float = MAX_SUBMITTED_SIMILARITY_GENERATE) -> tuple[bool, dict[str, Any]]:
    if not submitted_features:
        return False, {"max_similarity": 0.0, "collision_level": "low"}
    meta = score_against_submitted(expr or "", DEFAULTS, submitted_features)
    sim = float(meta.get("max_similarity", 0.0) or 0.0)
    return sim >= threshold, meta


def build_expressions(a_expr: str, b_expr: str) -> list[str]:
    a = a_expr.strip()
    b = b_expr.strip()
    a_rank = f"group_neutralize(rank(({a})), subindustry)"
    b_rank = f"group_neutralize(rank(({b})), {SIZE_BUCKET})"
    a_delta = f"group_neutralize(rank(ts_delta(({a}), 5)), subindustry)"
    b_decay = f"group_neutralize(rank(ts_decay_linear(({b}), 12)), {SIZE_BUCKET})"
    b_resid = f"group_neutralize(rank(ts_zscore(({b}), 120) - 0.35 * ts_zscore(({a}), 120)), subindustry)"
    b_market_resid = f"group_neutralize(rank(ts_zscore(({b}) - ts_mean(returns, 20), 120)), {SIZE_BUCKET})"
    # Use several structurally distinct blend families. Pure weighted sums keep
    # parent lineage too visible and can inherit the same self-correlation block
    # that stopped the parent alpha.
    return [
        # Baseline blends.
        f"group_neutralize(rank(0.65 * ({a}) + 0.35 * ({b})), subindustry)",
        f"group_neutralize(rank(0.50 * ({a}) + 0.50 * ({b})), {SIZE_BUCKET})",
        # Regime/gated blends.
        f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(0.70 * ({a}) + 0.30 * ({b})), subindustry), -1)",
        f"trade_when({LIQUID_GATE}, group_neutralize(rank(({a}) - 0.35 * ({b})), {SIZE_BUCKET}), {LIQUID_EXIT})",
        # Function-family rotations beyond simple linear blend.
        f"rank(ts_zscore(({a}), 80) + 0.25 * ts_zscore(({b}), 120))",
        f"group_neutralize(rank(ts_delta(({a}), 5) - 0.30 * rank(ts_delta(({b}), 10))), subindustry)",
        f"group_neutralize(rank(ts_decay_linear(({a}), 12) - 0.25 * ts_decay_linear(({b}), 24)), {SIZE_BUCKET})",
        f"trade_when(rank(ts_std_dev(returns, 20)) < 0.75, group_neutralize(rank(ts_zscore(({a}), 120) - 0.35 * ts_zscore(({b}), 120)), subindustry), -1)",
        f"rank(group_neutralize(rank(({a})), subindustry) - group_neutralize(rank(({b})), {SIZE_BUCKET}))",
        f"group_neutralize(rank((rank(({a})) - 0.5) * (rank(({b})) - 0.5)), subindustry)",
        # Structural decorrelation layer: explicitly residualize / separate time structure.
        f"group_neutralize(rank(({a_rank}) + 0.45 * ({b_resid})), subindustry)",
        f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(({a_delta}) + 0.40 * ({b_market_resid})), {SIZE_BUCKET}), -1)",
        f"trade_when(rank(ts_std_dev(returns, 20)) > 0.65, {a_rank}, {b_rank})",
        f"trade_when(rank(adv20) > 0.55, {a_rank}, {b_decay})",
        f"group_neutralize(rank(({a_delta}) - 0.30 * ({b_decay})), subindustry)",
    ]


def choose_pairs(rows: list[sqlite3.Row], limit: int, state: dict[str, Any] | None = None) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
    blocked = blocked_ids_from_log()
    lineage = truth_lineage_index()
    blacklist = (state or {}).get("lineage_pair_blacklist") or {}
    pairs: list[tuple[float, sqlite3.Row, sqlite3.Row]] = []
    for i, a in enumerate(rows):
        for b in rows[i + 1 : min(len(rows), i + 40)]:
            if a["alpha_id"] == b["alpha_id"]:
                continue
            at, bt = lineage.get(str(a["alpha_id"]), "unknown"), lineage.get(str(b["alpha_id"]), "unknown")
            if at == bt:
                print(f"SKIP reason=same_lineage_parents parents={a['alpha_id']},{b['alpha_id']} lineage={at}")
                continue
            pair_key = lineage_pair_key(at, bt)
            if int(blacklist.get(pair_key, 0) or 0) > 0:
                continue
            if family(str(a["alpha_id"])) == family(str(b["alpha_id"])) and not ({str(a["alpha_id"]), str(b["alpha_id"])} & blocked):
                continue
            pairs.append((pair_score(a, b, blocked, lineage), a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    chosen = []
    used = Counter()
    for _, a, b in pairs:
        if used[str(a["alpha_id"])] >= 2 or used[str(b["alpha_id"])] >= 2:
            continue
        chosen.append((a, b))
        used[str(a["alpha_id"])] += 1
        used[str(b["alpha_id"])] += 1
        if len(chosen) >= limit:
            break
    return chosen


def make_id(parent_ids: list[str], expr: str) -> str:
    digest = hashlib.sha1(("super|" + "|".join(parent_ids) + "|" + expr).encode("utf-8")).hexdigest()[:10]
    return f"super_{digest}"


def append_candidates(state: dict[str, Any], pairs: list[tuple[sqlite3.Row, sqlite3.Row]], max_add: int, submitted_features: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    built_from = state.setdefault("built_from", {})
    lineage = truth_lineage_index()
    with ALPHAS_CSV.open("a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for a, b in pairs:
            parent_ids = [str(a["alpha_id"]), str(b["alpha_id"])]
            for expr in build_expressions(str(a["expression"]), str(b["expression"])):
                expr = expr.strip()
                alpha_id = make_id(parent_ids, expr)
                if expr in existing_exprs or alpha_id in existing_ids or alpha_id in built_from:
                    continue
                blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                if blocked:
                    state.setdefault("skipped", {})[alpha_id] = {
                        "track": "super",
                        "reason": "SKIP_SUBMITTED_COLLISION",
                        "parent_alpha_ids": parent_ids,
                        "submitted_similarity_max": submitted_meta.get("max_similarity", 0.0),
                        "submitted_collision_level": submitted_meta.get("collision_level"),
                        "submitted_nearest_alpha_id": submitted_meta.get("nearest_submitted_alpha"),
                        "submitted_top_matches": submitted_meta.get("top_matches", []),
                        "created_at": now_iso(),
                    }
                    continue
                writer.writerow({"id": alpha_id, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(alpha_id)
                built_from[alpha_id] = {
                    "track": "super",
                    "parent_alpha_ids": parent_ids,
                    "parent_families": [family(parent_ids[0]), family(parent_ids[1])],
                    "parent_lineages": [lineage.get(parent_ids[0], "unknown"), lineage.get(parent_ids[1], "unknown")],
                    "lineage_pair": lineage_pair_key(lineage.get(parent_ids[0], "unknown"), lineage.get(parent_ids[1], "unknown")),
                    "expression": expr,
                    "created_at": now_iso(),
                }
                added.append({"id": alpha_id, "parents": parent_ids, "parent_lineages": built_from[alpha_id]["parent_lineages"]})
                if len(added) >= max_add:
                    return added
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SuperAlpha candidates from strong low-correlation-ish components")
    parser.add_argument("--max-add", type=int, default=3)
    parser.add_argument("--pair-limit", type=int, default=8)
    args = parser.parse_args()

    state = load_state()
    lineage = truth_lineage_index()
    refresh_lineage_pair_blacklist(state, lineage)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = source_candidates(conn)
    pairs = choose_pairs(rows, args.pair_limit, state)
    submitted_features = load_submitted_features()
    added = append_candidates(state, pairs, args.max_add, submitted_features=submitted_features)
    event = {
        "timestamp_utc": now_iso(),
        "candidate_components": len(rows),
        "selected_pairs": [[str(a["alpha_id"]), str(b["alpha_id"])] for a, b in pairs],
        "selected_pair_lineages": [[lineage.get(str(a["alpha_id"]), "unknown"), lineage.get(str(b["alpha_id"]), "unknown")] for a, b in pairs],
        "lineage_pair_blacklist": state.get("lineage_pair_blacklist", {}),
        "submitted_filter": {"threshold": MAX_SUBMITTED_SIMILARITY_GENERATE, "submitted_library_rows": len(submitted_features)},
        "added": added,
    }
    state.setdefault("history", []).append(event)
    state["history"] = state["history"][-100:]
    save_state(state)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
