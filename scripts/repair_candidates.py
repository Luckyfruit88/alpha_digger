#!/usr/bin/env python3
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import re
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
from scripts.self_corr_truth_table import classify_lineage  # noqa: E402
from scripts.submitted_similarity import extract_operators, load_submitted_features, score_against_submitted  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "repair_candidates_state.json"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
TRUTH_TABLE_PATH = ROOT / "state" / "self_corr_truth_table.json"
REPORT_PATH = ROOT / "reports" / "repair_candidates_latest.md"

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
LINEAGE_QUOTA_MAX = 0.25
ANALYST_QUOTA_FRACTION = 0.30
MAX_SUBMITTED_SIMILARITY_GENERATE = 0.65
ESCAPE_FIELD_POOLS = {
    "fundamental_quality": ["fnd6_ebitda", "fnd6_netincome", "fnd6_assets", "fnd6_debt", "fnd6_loxdr"],
    "accrual_quality": ["cashflow_per_share_min_guidance", "anl4_adjusted_netincome_ft", "anl4_ebit_value"],
    "cashflow_stability": ["cashflow_per_share_min_guidance", "anl4_ebit_value", "anl4_tbve_ft"],
}
ESCAPE_SOURCE_FIELD_PATTERNS = [
    r"\bearnings_momentum_analyst_score\b",
    r"\banalyst_revision_rank_derivative\b",
    r"\breturns\b",
    r"\bclose\b",
    r"\bopen\b",
    r"\bvwap\b",
    r"\bvolume\b",
    r"\badv20\b",
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


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"version": 1, "created_at": now_iso(), "seeded_from": {}, "history": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "created_at": now_iso(), "seeded_from": {}, "history": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def truth_index() -> dict[str, dict[str, Any]]:
    payload = load_json(TRUTH_TABLE_PATH) or {}
    rows = payload.get("alphas") or []
    return {str(row.get("alpha_id")): row for row in rows if isinstance(row, dict) and row.get("alpha_id")}


def lineage_share_summary(truth_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in truth_rows.values():
        theme = str((row.get("lineage") or {}).get("theme") or "unknown")
        counts[theme] += 1
    total = sum(counts.values())
    shares = {theme: round(count / max(1, total), 4) for theme, count in counts.items()}
    return {"total": total, "counts": dict(counts), "shares": shares, "analyst_share": shares.get("analyst_earnings", 0.0)}


def row_lineage_theme(row: sqlite3.Row, truth_rows: dict[str, dict[str, Any]]) -> str:
    truth_row = truth_rows.get(str(row["alpha_id"])) or {}
    theme = str((truth_row.get("lineage") or {}).get("theme") or "")
    if theme:
        return theme
    return str(classify_lineage(str(row["expression"] or "")).get("theme") or "unknown")


def should_retire_parent(alpha_id: str, truth_rows: dict[str, dict[str, Any]]) -> tuple[bool, int, str]:
    row = truth_rows.get(alpha_id) or {}
    depth = int(to_float(row.get("repair_depth"), 0))
    status = str(row.get("self_corr_status") or "unknown")
    if depth >= 3 and status in {"blocked", "predicted_blocked"}:
        return True, depth, status
    return False, depth, status


def record_retirement(state: dict[str, Any], alpha_id: str, depth: int, status: str) -> None:
    item = {
        "timestamp_utc": now_iso(),
        "alpha_id": alpha_id,
        "repair_depth": depth,
        "self_corr_status": status,
        "reason": "self_corr_blocked",
    }
    retired = state.setdefault("retired", [])
    if not any(isinstance(x, dict) and x.get("alpha_id") == alpha_id for x in retired):
        retired.append(item)
    print(f"RETIRED {alpha_id} repair_depth={depth} reason=self_corr_blocked")


def parent_allowed(state: dict[str, Any], source_id: str, truth_rows: dict[str, dict[str, Any]]) -> bool:
    retire, depth, status = should_retire_parent(source_id, truth_rows)
    if retire:
        record_retirement(state, source_id, depth, status)
        return False
    return True


def apply_lineage_quota(candidates: list[dict[str, Any]], quota_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    analyst_share = float(quota_summary.get("analyst_share", 0.0) or 0.0)
    status = {
        "lineage_quota_max": LINEAGE_QUOTA_MAX,
        "analyst_share": round(analyst_share, 4),
        "enabled": analyst_share > LINEAGE_QUOTA_MAX,
        "before": dict(Counter(str(c.get("lineage_theme") or "unknown") for c in candidates)),
        "dropped": 0,
    }
    if analyst_share <= LINEAGE_QUOTA_MAX:
        status["after"] = status["before"]
        return candidates, status
    analyst_cap = max(1, int(len(candidates) * ANALYST_QUOTA_FRACTION))
    kept: list[dict[str, Any]] = []
    analyst_kept = 0
    for candidate in candidates:
        if candidate.get("lineage_theme") == "analyst_earnings":
            if analyst_kept >= analyst_cap:
                status["dropped"] += 1
                continue
            analyst_kept += 1
        kept.append(candidate)
    status["analyst_cap"] = analyst_cap
    status["after"] = dict(Counter(str(c.get("lineage_theme") or "unknown") for c in kept))
    return kept, status


def apply_submitted_similarity_filter(
    candidates: list[dict[str, Any]],
    submitted_features: list[dict[str, Any]],
    threshold: float = MAX_SUBMITTED_SIMILARITY_GENERATE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    status: dict[str, Any] = {
        "threshold": threshold,
        "enabled": bool(submitted_features),
        "submitted_library_rows": len(submitted_features or []),
        "before": len(candidates),
        "dropped": 0,
        "dropped_by_level": {},
        "nearest": {},
    }
    if not submitted_features:
        status["after"] = len(candidates)
        return candidates, status
    kept: list[dict[str, Any]] = []
    dropped_levels: Counter[str] = Counter()
    nearest: Counter[str] = Counter()
    for candidate in candidates:
        meta = score_against_submitted(str(candidate.get("expression") or ""), DEFAULTS, submitted_features)
        sim = float(meta.get("max_similarity", 0.0) or 0.0)
        level = str(meta.get("collision_level") or "low")
        candidate["submitted_similarity_max"] = round(sim, 4)
        candidate["submitted_collision_level"] = level
        candidate["submitted_nearest_alpha_id"] = meta.get("nearest_submitted_alpha")
        if sim >= threshold:
            status["dropped"] += 1
            dropped_levels[level] += 1
            if meta.get("nearest_submitted_alpha"):
                nearest[str(meta.get("nearest_submitted_alpha"))] += 1
            continue
        kept.append(candidate)
    status["after"] = len(kept)
    status["dropped_by_level"] = dict(dropped_levels)
    status["nearest"] = dict(nearest.most_common(10))
    return kept, status


def write_report(event: dict[str, Any]) -> None:
    lines = [
        f"# Repair Candidates Latest - {event.get('timestamp_utc')}",
        "",
        f"- generated: `{event.get('generated', 0)}`",
        f"- added: `{event.get('added', 0)}`",
        f"- sources: `{event.get('sources', [])}`",
        f"- repair_types: `{event.get('repair_types', [])}`",
        "",
        "## Lineage Quota Status",
        f"`{json.dumps(event.get('lineage_quota_status', {}), sort_keys=True)}`",
        "",
        "## Submitted Reference Filter",
        f"`{json.dumps(event.get('submitted_filter_status', {}), sort_keys=True)}`",
        "",
        "## Retired Parents",
    ]
    for item in event.get("retired_this_run", []) or []:
        lines.append(f"- {item.get('alpha_id')} depth={item.get('repair_depth')} status={item.get('self_corr_status')} reason={item.get('reason')}")
    lines.extend(["", "## Abandoned Submitted-Collision Parents"])
    for item in event.get("abandoned_parents_this_run", []) or []:
        lines.append(f"- {item.get('repair_type')} similarity={item.get('submitted_similarity_max')} nearest={item.get('submitted_nearest_alpha_id')} reason={item.get('reason')}")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
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


def high_turnover_candidates(conn: sqlite3.Connection, limit: int = 6) -> list[sqlite3.Row]:
    return conn.execute(
        '''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons
        from backtest_results
        where turnover >= 0.55
          and sharpe >= 1.45
          and fitness >= 0.80
        order by sharpe desc, fitness desc
        limit ?
        ''',
        (limit,),
    ).fetchall()


def self_corr_candidates(conn: sqlite3.Connection, limit: int = 6) -> list[sqlite3.Row]:
    truth_ids = truth_self_corr_sources(limit=limit)
    if truth_ids:
        rows = [fetch_result_by_id(conn, alpha_id) for alpha_id in truth_ids]
        return [row for row in rows if row is not None][:limit]
    return conn.execute(
        '''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons
        from backtest_results
        where fail_reasons like '%SELF_CORRELATION%'
        order by sharpe desc, fitness desc
        limit ?
        ''',
        (limit,),
    ).fetchall()


def truth_self_corr_sources(
    limit: int = 10,
    statuses: set[str] | None = None,
    prefix: str | None = None,
    require_pass_quality: bool = False,
) -> list[str]:
    statuses = statuses or {"blocked", "pending", "cooldown"}
    payload = load_json(TRUTH_TABLE_PATH)
    rows = payload.get("alphas") or []
    ranked: list[tuple[int, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        alpha_id = str(row.get("alpha_id") or "")
        if not alpha_id:
            continue
        if prefix and not alpha_id.startswith(prefix):
            continue
        if str(row.get("self_corr_status") or "") not in statuses:
            continue
        if require_pass_quality and not bool(row.get("pass_quality")):
            continue
        skip_counts = row.get("skip_counts") or {}
        total_skips = sum(int(v) for v in skip_counts.values()) if isinstance(skip_counts, dict) else 0
        ranked.append((total_skips, str(row.get("last_seen_at") or ""), alpha_id))
    ranked.sort(reverse=True)
    return [alpha_id for _, _, alpha_id in ranked[:limit]]


def self_corr_sources_from_log(limit: int = 10) -> list[str]:
    if not AUTO_SUBMIT_LOG.exists():
        return []
    counts: Counter[str] = Counter()
    for line in AUTO_SUBMIT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()[-400:]:
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
            alpha_id, reason = item[0], str(item[1])
            if "SELF_CORRELATION" in reason or "self-correlation cooldown active" in reason:
                counts[str(alpha_id)] += 1
    return [alpha_id for alpha_id, _ in counts.most_common(limit)]


def fetch_result_by_id(conn: sqlite3.Connection, alpha_id: str) -> sqlite3.Row | None:
    return conn.execute(
        '''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons
        from backtest_results
        where alpha_id=?
        order by datetime(created_at) desc
        limit 1
        ''',
        (alpha_id,),
    ).fetchone()


def submit_gated_repairsc_sources(conn: sqlite3.Connection, limit: int = 8) -> list[sqlite3.Row]:
    rows = conn.execute(
        f'''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons
        from backtest_results
        where {exact_prefix_clause("alpha_id")}
          and sharpe >= 1.6
          and fitness >= 1.0
          and turnover <= 0.45
        order by datetime(created_at) desc
        limit ?
        ''',
        (exact_prefix_param("repairsc_"), limit * 3),
    ).fetchall()
    blocked_ids = set(truth_self_corr_sources(limit=limit * 6, prefix="repairsc_", require_pass_quality=True))
    if not blocked_ids:
        blocked_ids = set(self_corr_sources_from_log(limit=limit * 6))
    prioritized = [row for row in rows if str(row["alpha_id"]) in blocked_ids]
    return prioritized[:limit]


def submit_gated_repairsc2_sources(conn: sqlite3.Connection, limit: int = 6) -> list[sqlite3.Row]:
    rows = conn.execute(
        f'''
        select alpha_id, expression, sharpe, fitness, turnover, fail_reasons
        from backtest_results
        where {exact_prefix_clause("alpha_id")}
          and sharpe >= 1.6
          and fitness >= 1.0
          and turnover <= 0.45
        order by datetime(created_at) desc
        limit ?
        ''',
        (exact_prefix_param("repairsc2_"), limit * 3),
    ).fetchall()
    blocked_ids = set(truth_self_corr_sources(limit=limit * 8, prefix="repairsc2_", require_pass_quality=True))
    if not blocked_ids:
        blocked_ids = set(self_corr_sources_from_log(limit=limit * 8))
    prioritized = [row for row in rows if str(row["alpha_id"]) in blocked_ids]
    return prioritized[:limit]



PRICE_EQUIV_REPLACEMENTS = [
    (r"\bclose\b", "open", "close_to_open"),
    (r"\bclose\b", "high", "close_to_high"),
    (r"\bclose\b", "low", "close_to_low"),
    (r"\bclose\b", "vwap", "close_to_vwap"),
    (r"\bopen\b", "vwap", "open_to_vwap"),
    (r"\bopen\b", "close", "open_to_close"),
]
RETURN_EQUIV_REPLACEMENTS = [
    (r"\breturns\b", "((open - ts_delay(close, 1)) / ts_delay(close, 1))", "returns_to_overnight"),
    (r"\breturns\b", "((close - open) / open)", "returns_to_intraday"),
    (r"\breturns\b", "((vwap - open) / open)", "returns_to_vwap_open"),
]


def equivalent_field_repairs(expr: str) -> list[tuple[str, str]]:
    """Generate economically similar field substitutions for self-corr repair.

    Keep this conservative: use price/liquidity/return equivalents rather than
    arbitrary fields, so correlation reduction still has an interpretable reason.
    """
    core = expr.strip()
    variants: list[tuple[str, str]] = []
    for pattern, repl, label in PRICE_EQUIV_REPLACEMENTS + RETURN_EQUIV_REPLACEMENTS:
        new_expr, n = re.subn(pattern, repl, core)
        if n and new_expr != core:
            variants.append((label, new_expr))
    # Add helper-style substitutions without rewriting the whole expression.
    overnight = "((open - ts_delay(close, 1)) / ts_delay(close, 1))"
    intraday = "((close - open) / open)"
    variants.extend([
        ("field_helper_overnight", f"group_neutralize(rank(ts_zscore(({core}), 120) + 0.18 * rank(ts_zscore({overnight}, 120))), subindustry)"),
        ("field_helper_intraday", f"group_neutralize(rank(ts_zscore(({core}), 120) - 0.18 * rank(ts_zscore({intraday}, 120))), {SIZE_BUCKET})"),
        ("field_helper_vwap_open", f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_rank(({core}), 80) - ts_rank((vwap - open) / open, 120)), {SIZE_BUCKET}), {LIQUID_EXIT})"),
    ])
    return variants


def operator_similarity_repairs(expr: str) -> list[tuple[str, str]]:
    """Generate similar-meaning operator substitutions to reduce max correlation."""
    core = expr.strip()
    variants: list[tuple[str, str]] = [
        ("op_rank_to_zscore", f"rank(ts_zscore(({core}), 120))"),
        ("op_rank_to_ts_rank", f"rank(ts_rank(({core}), 120))"),
        ("op_decay_delta", f"group_neutralize(rank(ts_delta(ts_decay_linear(({core}), 12), 5)), subindustry)"),
        ("op_median_smooth", f"group_neutralize(rank(ts_median(({core}), 20)), {SIZE_BUCKET})"),
        ("op_residualized_zscore", f"group_neutralize(rank(ts_zscore(({core}), 120) - ts_rank(returns, 252)), subindustry)"),
    ]
    replacements = [
        ("ts_mean(", "ts_median(", "op_mean_to_median"),
        ("ts_mean(", "ts_decay_linear(", "op_mean_to_decay"),
        ("ts_rank(", "ts_zscore(", "op_tsrank_to_zscore"),
        ("ts_zscore(", "ts_rank(", "op_zscore_to_tsrank"),
    ]
    for old, new, label in replacements:
        if old in core:
            variants.append((label, core.replace(old, new, 1)))
    return variants


def structural_self_corr_repairs(expr: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, variant in equivalent_field_repairs(expr) + operator_similarity_repairs(expr):
        if variant and variant not in seen and variant.strip() != expr.strip():
            seen.add(variant)
            out.append((label, variant))
    return out


def escape_repair(expr: str) -> list[tuple[str, str]]:
    """Move weak-collision parents to low-density economic lineages.

    This keeps the parent operator/window skeleton where possible but forces
    analyst/price/liquidity source fields toward fundamental/accrual/cashflow
    pools, so weak submitted collisions do not just become superficial window
    mutations.
    """
    core = expr.strip()
    operators = extract_operators(core)
    pool_fields = [field for fields in ESCAPE_FIELD_POOLS.values() for field in fields]
    variants: list[tuple[str, str]] = []
    for idx, field in enumerate(pool_fields[:10]):
        candidate = core
        replaced = False
        for pattern in ESCAPE_SOURCE_FIELD_PATTERNS:
            candidate, n = re.subn(pattern, field, candidate, count=1)
            if n:
                replaced = True
                break
        if not replaced:
            if "ts_rank" in operators:
                candidate = f"group_neutralize(rank(ts_rank({field}, 120) - ts_rank(cap, 120)), subindustry)"
            elif "ts_zscore" in operators:
                candidate = f"group_neutralize(rank(ts_zscore({field}, 120) - ts_zscore(cap, 120)), {SIZE_BUCKET})"
            else:
                candidate = f"group_neutralize(rank(ts_rank({field}, 120)), subindustry)"
        else:
            candidate = f"group_neutralize(rank({candidate}), subindustry)"
        variants.append((f"escape_repair_{idx}_{field}", candidate))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, candidate in variants:
        if candidate and candidate not in seen and candidate != core:
            seen.add(candidate)
            out.append((label, candidate))
    return out


def parent_similarity_policy(expr: str, submitted_features: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not submitted_features:
        return "normal", {"max_similarity": 0.0, "collision_level": "low"}
    meta = score_against_submitted(expr or "", DEFAULTS, submitted_features)
    sim = float(meta.get("max_similarity", 0.0) or 0.0)
    if sim >= 0.65:
        return "abandon", meta
    if sim >= 0.45:
        return "escape", meta
    return "normal", meta


def record_parent_abandon(state: dict[str, Any], source_id: str, meta: dict[str, Any], repair_type: str) -> None:
    state.setdefault("abandoned_parents", {})[source_id] = {
        "reason": "ABANDON_SUBMITTED_COLLISION_PARENT",
        "repair_type": repair_type,
        "submitted_similarity_max": meta.get("max_similarity", 0.0),
        "submitted_collision_level": meta.get("collision_level"),
        "submitted_nearest_alpha_id": meta.get("nearest_submitted_alpha"),
        "submitted_top_matches": meta.get("top_matches", []),
        "created_at": now_iso(),
    }


def repair_variants_for_parent(expr: str, base_variants: list[str], submitted_features: list[dict[str, Any]], state: dict[str, Any], source_id: str, repair_type: str) -> tuple[list[tuple[str, str]], str, dict[str, Any]]:
    policy, meta = parent_similarity_policy(expr, submitted_features)
    if policy == "abandon":
        record_parent_abandon(state, source_id, meta, repair_type)
        return [], policy, meta
    if policy == "escape":
        return escape_repair(expr), policy, meta
    return [(repair_type, variant) for variant in base_variants], policy, meta

def turnover_repairs(expr: str) -> list[str]:
    core = expr.strip()
    variants = [
        f"ts_mean({core}, 5)",
        f"ts_mean({core}, 10)",
        f"trade_when({LOW_VOL_GATE}, {core}, -1)",
        f"trade_when({LIQUID_GATE}, {core}, {LIQUID_EXIT})",
        f"group_neutralize(ts_mean({core}, 5), {SIZE_BUCKET})",
        f"group_neutralize(trade_when({LOW_VOL_GATE}, {core}, -1), {SIZE_BUCKET})",
    ]
    return variants


def self_corr_repairs(expr: str) -> list[str]:
    core = expr.strip()
    variants = [
        f"group_neutralize({core}, subindustry)",
        f"group_neutralize({core}, {SIZE_BUCKET})",
        f"rank(ts_zscore({core}, 120))",
        f"rank({core} - ts_zscore(returns, 120))",
        f"trade_when({LOW_VOL_GATE}, group_neutralize({core}, subindustry), -1)",
        f"rank({core} + 0.15 * rank(-ts_corr(abs((open - ts_delay(close, 1)) / ts_delay(close, 1)), ts_delay(volume / adv20, 1), 20)))",
    ]
    return variants


def submit_gated_self_corr_repairs(expr: str) -> list[str]:
    core = expr.strip()
    overnight_corr = "rank(-ts_corr(abs((open - ts_delay(close, 1)) / ts_delay(close, 1)), ts_delay(volume / adv20, 1), 20))"
    market_resid = "group_mean(returns, 1, market)"
    variants = [
        f"group_neutralize(rank(ts_zscore({core}, 120) - ts_zscore(cap, 120)), subindustry)",
        f"group_neutralize(rank(ts_rank({core}, 120) - ts_rank(returns, 252)), bucket(rank(cap), range='0.1,1,0.1'))",
        f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_rank({core}, 120) - ts_rank(returns, 252)), subindustry), -1)",
        f"group_neutralize(rank({core} + 0.2 * {overnight_corr}), subindustry)",
        f"rank(ts_zscore({core} - ts_mean(returns - {market_resid}, 20), 120))",
        f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_zscore({core}, 120) - ts_zscore(adv20, 120)), bucket(rank(cap), range='0.1,1,0.1')), {LIQUID_EXIT})",
    ]
    return variants


def submit_gated_self_corr_repairs_v2(expr: str) -> list[str]:
    core = expr.strip()
    overnight_corr = "rank(-ts_corr(abs((open - ts_delay(close, 1)) / ts_delay(close, 1)), ts_delay(volume / adv20, 1), 20))"
    vwap_open = "rank(ts_corr(vwap, open, 252))"
    market_resid = "group_mean(returns, 1, market)"
    size_bucket = "bucket(rank(cap), range='0.1,1,0.1')"
    variants = [
        f"group_neutralize(rank(ts_zscore({core} - ts_mean(returns - {market_resid}, 20), 120) - ts_zscore(cap, 120)), {size_bucket})",
        f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_rank({core}, 80) - ts_rank(volume / adv20, 120)), subindustry), -1)",
        f"group_neutralize(rank(ts_rank({core}, 80) - ts_rank(vwap_open, 120) + 0.2 * {overnight_corr}), {size_bucket})",
        f"trade_when({LIQUID_GATE}, group_neutralize(rank(ts_zscore({core}, 80) - ts_zscore(ts_std_dev(returns, 30), 80)), subindustry), {LIQUID_EXIT})",
        f"group_neutralize(rank(ts_rank({core}, 120) - ts_rank(group_mean(returns, 1, subindustry), 120) - ts_rank(cap, 120)), subindustry)",
        f"rank(ts_zscore(group_neutralize({core}, subindustry) - ts_mean(group_neutralize({core}, {size_bucket}), 15), 80))",
    ]
    return variants


def append_rows(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    existing_exprs, existing_ids = existing_csv()
    ALPHAS_CSV.parent.mkdir(parents=True, exist_ok=True)
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
    return added


def make_id(prefix: str, source_id: str, expression: str) -> str:
    digest = hashlib.sha1(f"{prefix}|{source_id}|{expression}".encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate local repair candidates from prior backtest results")
    parser.add_argument("--preview", action="store_true", help="Evaluate repair sources/quotas and write report without appending alphas.csv or state")
    args = parser.parse_args()

    state = load_state()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    candidates = []
    seeded_from: dict[str, str] = state.setdefault("seeded_from", {})
    truth_rows = truth_index()
    submitted_features = load_submitted_features()
    quota_summary = lineage_share_summary(truth_rows)
    retired_before = len(state.get("retired", []) or [])
    abandoned_before = set((state.get("abandoned_parents") or {}).keys())

    for row in high_turnover_candidates(conn):
        source_id = row["alpha_id"]
        if not parent_allowed(state, source_id, truth_rows):
            continue
        lineage_theme = row_lineage_theme(row, truth_rows)
        for expr in turnover_repairs(row["expression"]):
            alpha_id = make_id("repairto", source_id, expr)
            if alpha_id in seeded_from:
                continue
            candidates.append({"id": alpha_id, "expression": expr, "source_id": source_id, "repair_type": "turnover", "lineage_theme": lineage_theme})
            seeded_from[alpha_id] = source_id

    seen_self_corr_sources = set()
    for row in self_corr_candidates(conn):
        source_id = row["alpha_id"]
        if not parent_allowed(state, source_id, truth_rows):
            continue
        lineage_theme = row_lineage_theme(row, truth_rows)
        seen_self_corr_sources.add(source_id)
        variants, parent_policy, parent_meta = repair_variants_for_parent(row["expression"], self_corr_repairs(row["expression"]), submitted_features, state, source_id, "self_corr")
        for label, expr in variants:
            alpha_id = make_id("repairsc", source_id, expr)
            if alpha_id in seeded_from:
                continue
            candidates.append({"id": alpha_id, "expression": expr, "source_id": source_id, "repair_type": label if parent_policy == "escape" else "self_corr", "parent_similarity_policy": parent_policy, "parent_submitted_similarity_max": parent_meta.get("max_similarity", 0.0), "lineage_theme": lineage_theme})
            seeded_from[alpha_id] = source_id

    for alpha_id in self_corr_sources_from_log():
        if alpha_id in seen_self_corr_sources:
            continue
        row = fetch_result_by_id(conn, alpha_id)
        if not row or not row["expression"]:
            continue
        if not parent_allowed(state, alpha_id, truth_rows):
            continue
        lineage_theme = row_lineage_theme(row, truth_rows)
        variants, parent_policy, parent_meta = repair_variants_for_parent(row["expression"], self_corr_repairs(row["expression"]), submitted_features, state, alpha_id, "self_corr_log")
        for label, expr in variants:
            repair_id = make_id("repairsc", alpha_id, expr)
            if repair_id in seeded_from:
                continue
            candidates.append({"id": repair_id, "expression": expr, "source_id": alpha_id, "repair_type": label if parent_policy == "escape" else "self_corr_log", "parent_similarity_policy": parent_policy, "parent_submitted_similarity_max": parent_meta.get("max_similarity", 0.0), "lineage_theme": lineage_theme})
            seeded_from[repair_id] = alpha_id

    for row in submit_gated_repairsc_sources(conn):
        source_id = row["alpha_id"]
        if not parent_allowed(state, source_id, truth_rows):
            continue
        lineage_theme = row_lineage_theme(row, truth_rows)
        variants, parent_policy, parent_meta = repair_variants_for_parent(row["expression"], submit_gated_self_corr_repairs(row["expression"]), submitted_features, state, source_id, "submit_gated_self_corr")
        for label, expr in variants:
            repair_id = make_id("repairsc2", source_id, expr)
            if repair_id in seeded_from:
                continue
            candidates.append({"id": repair_id, "expression": expr, "source_id": source_id, "repair_type": label if parent_policy == "escape" else "submit_gated_self_corr", "parent_similarity_policy": parent_policy, "parent_submitted_similarity_max": parent_meta.get("max_similarity", 0.0), "lineage_theme": lineage_theme})
            seeded_from[repair_id] = source_id

    for row in submit_gated_repairsc2_sources(conn):
        source_id = row["alpha_id"]
        if not parent_allowed(state, source_id, truth_rows):
            continue
        lineage_theme = row_lineage_theme(row, truth_rows)
        variants, parent_policy, parent_meta = repair_variants_for_parent(row["expression"], submit_gated_self_corr_repairs_v2(row["expression"]), submitted_features, state, source_id, "submit_gated_self_corr_v2")
        for label, expr in variants:
            repair_id = make_id("repairsc3", source_id, expr)
            if repair_id in seeded_from:
                continue
            candidates.append({"id": repair_id, "expression": expr, "source_id": source_id, "repair_type": label if parent_policy == "escape" else "submit_gated_self_corr_v2", "parent_similarity_policy": parent_policy, "parent_submitted_similarity_max": parent_meta.get("max_similarity", 0.0), "lineage_theme": lineage_theme})
            seeded_from[repair_id] = source_id
        # Phase 3: use equivalent field and operator-similarity mutation as a
        # first-class route for high-quality candidates that keep failing submit
        # self-correlation checks.  This targets max-correlation reduction without
        # arbitrary grouping.
        for label, expr in structural_self_corr_repairs(row["expression"]):
            repair_id = make_id("repairsc3", source_id, label + "|" + expr)
            if repair_id in seeded_from:
                continue
            candidates.append({"id": repair_id, "expression": expr, "source_id": source_id, "repair_type": label, "lineage_theme": lineage_theme})
            seeded_from[repair_id] = source_id

    generated_ids = {str(c["id"]) for c in candidates}
    candidates, lineage_quota_status = apply_lineage_quota(candidates, quota_summary)
    candidates, submitted_filter_status = apply_submitted_similarity_filter(candidates, submitted_features)
    kept_ids = {str(c["id"]) for c in candidates}
    for dropped_id in generated_ids - kept_ids:
        seeded_from.pop(dropped_id, None)
    added = 0 if args.preview else append_rows(candidates)
    retired_this_run = (state.get("retired", []) or [])[retired_before:]
    event = {
        "timestamp_utc": now_iso(),
        "preview": args.preview,
        "added": added,
        "generated": len(candidates),
        "sources": sorted({c["source_id"] for c in candidates}),
        "repair_types": sorted({c["repair_type"] for c in candidates}),
        "lineage_quota_status": lineage_quota_status,
        "submitted_filter_status": submitted_filter_status,
        "retired_this_run": retired_this_run,
        "abandoned_parents_this_run": [v for k, v in (state.get("abandoned_parents") or {}).items() if k not in abandoned_before],
    }
    if not args.preview:
        state.setdefault("history", []).append(event)
        state["history"] = state["history"][-100:]
        save_state(state)
    write_report(event)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
