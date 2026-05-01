#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.expr_parser import expression_has_token, replace_ts_corr_second_args, ts_corr_calls
from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param
from scripts.lineage_occupancy import build_lineage_hints, load_lineage_hints
from scripts.submitted_similarity import load_submitted_features, score_against_submitted

DB_PATH = ROOT / "data" / "backtests.sqlite3"
FIELDS_CSV = ROOT / "data" / "data_fields_scored.csv"
ALPHAS_CSV = ROOT / "alphas.csv"
STATE_PATH = ROOT / "state" / "d1_generator_state.json"
PANEL_PATH = ROOT / "state" / "multi_d1_panel.json"
D1_TRUTH_TABLE_PATH = ROOT / "state" / "d1_truth_table.json"
REPORT_PATH = ROOT / "reports" / "d1_generator_latest.md"

DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 8,
    "neutralization": "INDUSTRY",
    "truncation": 0.03,
}

SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LOW_VOL_GATE = "rank(ts_std_dev(returns, 20)) < 0.72"

D1_V1_STRUCTURES = {
    "delta_decay_split": lambda a, b: f"group_neutralize(rank(ts_delta(ts_zscore({a}, 80), 5) - 0.30 * ts_decay_linear(ts_zscore({b}, 160), 20)), subindustry)",
    "residualized_helper": lambda a, b: f"group_neutralize(rank(ts_zscore({a}, 120) + 0.22 * (ts_zscore({b}, 120) - ts_rank(returns, 252))), {SIZE_BUCKET})",
    "regime_split": lambda a, b: f"trade_when({LOW_VOL_GATE}, group_neutralize(rank(ts_rank({a}, 120)), subindustry), group_neutralize(rank(ts_decay_linear({b}, 18)), {SIZE_BUCKET}))",
}

MAX_OPERATORS = 58
# Clean D1 v2 passes observed at 511 and 597 chars; cap below the timeout-prone
# expansion zone while preserving viable anchor expressions.
MAX_EXPR_LEN = 620
MAX_SUBMITTED_SIMILARITY_GENERATE = 0.65

D1_V2_TEMPLATES = {
    # v2.1 priority: simple, anchor-preserving, and lower operator pressure.
    "conservative_rank_spread": lambda a, h: f"group_neutralize(0.75 * rank({a}) - 0.25 * rank(ts_decay_linear({h}, 5)), subindustry)",
    "regime_anchor_helper": lambda a, h: f"trade_when(rank(ts_std_dev(returns, 20)) > 0.45, group_neutralize(rank({a}), subindustry), group_neutralize(rank(ts_zscore({h}, 20)), {SIZE_BUCKET}))",
    "vol_scaled_residual": lambda a, h: f"group_neutralize(rank({a}), subindustry) - 0.12 * group_neutralize(ts_zscore({h}, 10), subindustry) / (ts_std_dev(returns, 20) + 0.001)",
    # Reduced priority: useful, but can bloat expressions when anchors are already complex.
    "anchor_decay_helper_delta": lambda a, h: f"group_neutralize(ts_decay_linear(rank({a}), 10) - 0.22 * ts_delta(rank({h}), 3), subindustry)",
    # Paused by default: high ts_corr/operator pressure and produced a concentrated-weight failure.
    "anchor_preserve_residual_helper": lambda a, h: f"group_neutralize(rank({a}) + 0.18 * group_neutralize(rank({h}) - ts_corr(rank({h}), rank({a}), 20), subindustry), subindustry)",
}

D1_V2_PAUSED_TEMPLATES = {"anchor_preserve_residual_helper"}

MAX_V22_OPERATORS = 45
D1_V22_TEMPLATES = {
    "v22_conservative_rank_spread": lambda a, h: f"rank({a}) - rank(ts_decay_linear({h}, 10))",
    "v22_regime_sign_anchor": lambda a, h: f"sign(ts_mean({a}, 5)) * group_neutralize(rank({h}), subindustry)",
}

PRICE_CORR_TARGETS = {"close", "open", "returns", "ret", "daily_return", "price", "vwap"}
V23_SAFE_TARGETS = ["adv20", "volume", "adv60"]
MAX_V23_OPERATORS = 45
MAX_TS_CORR_COUNT = 2
FIELD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
IGNORED_FIELD_TOKENS = {
    "abs", "bucket", "group_mean", "group_neutralize", "rank", "range", "signed_power",
    "trade_when", "ts_corr", "ts_decay_linear", "ts_delta", "ts_mean", "ts_median",
    "ts_rank", "ts_std_dev", "ts_zscore", "if_else", "log", "sign", "industry",
    "market", "sector", "subindustry",
}


def is_price_corr_target(expr: str) -> bool:
    return expression_has_token(expr or "", PRICE_CORR_TARGETS)


def find_corr_calls(expr: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for call in ts_corr_calls(expr or ""):
        field_b = str(call.get("field_b") or "").strip()
        item = dict(call)
        item["is_price_target"] = is_price_corr_target(field_b)
        matches.append(item)
    return matches


def replace_corr_targets(expr: str, new_target: str = "adv20") -> str:
    return replace_ts_corr_second_args(expr or "", is_price_corr_target, new_target)


def replace_all_corr_targets(expr: str, new_target: str) -> str:
    return replace_ts_corr_second_args(expr or "", lambda target: bool(target.strip()), new_target)


def replace_safe_corr_targets(expr: str, new_target: str) -> str:
    return replace_ts_corr_second_args(expr or "", lambda target: not is_price_corr_target(target), new_target)


def ts_corr_count(expr: str) -> int:
    return len(find_corr_calls(expr or ""))


def count_operators(expr: str) -> int:
    return len(re.findall(r"\b[a-z_]+\s*\(", expr or ""))


def operator_combo(expr: str) -> list[str]:
    return sorted(set(re.findall(r"\b[a-z_]+\s*\(", expr or "")))[:12]


def field_combo(expr: str) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for token in FIELD_RE.findall(expr or ""):
        t = token.lower()
        if t in IGNORED_FIELD_TOKENS or t.isdigit() or t in seen:
            continue
        seen.add(t)
        fields.append(token)
    return fields[:12]


def neutralization_positions(expr: str) -> list[int]:
    return [m.start() for m in re.finditer(r"\b(?:group_neutralize|neutralize)\s*\(", expr or "")][:8]


def validate_expression(expr: str) -> tuple[bool, str]:
    op_count = count_operators(expr)
    if op_count > MAX_OPERATORS:
        return False, f"operator_count={op_count} > {MAX_OPERATORS}"
    if len(expr or "") > MAX_EXPR_LEN:
        return False, f"expression_length={len(expr or '')} > {MAX_EXPR_LEN}"
    return True, "ok"


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
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_expression_index() -> dict[str, str]:
    if not DB_PATH.exists():
        return {}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("select alpha_id, expression from backtest_results where expression is not null").fetchall()
    conn.close()
    return {str(alpha_id): str(expr) for alpha_id, expr in rows}


def load_template_bias(state: dict[str, Any]) -> dict[str, Any]:
    truth = load_json(D1_TRUTH_TABLE_PATH) or {}
    clear_ids = [str(entry.get("alpha_id")) for entry in truth.get("entries", []) if isinstance(entry, dict) and entry.get("detail_check_result") == "clear"]
    expressions = load_expression_index()
    candidates = state.get("candidates") or {}
    templates = [str((candidates.get(alpha_id) or {}).get("template") or (candidates.get(alpha_id) or {}).get("template_name") or "") for alpha_id in clear_ids]
    templates = [t for t in templates if t]
    clear_exprs = [expressions[alpha_id] for alpha_id in clear_ids if alpha_id in expressions]
    return {
        "clear_count": len(clear_ids),
        "clear_alpha_ids": clear_ids[:20],
        "templates": templates[:20],
        "operator_combos": [operator_combo(expr) for expr in clear_exprs[:10]],
        "field_combos": [field_combo(expr) for expr in clear_exprs[:10]],
        "neutralization_positions": [neutralization_positions(expr) for expr in clear_exprs[:10]],
    }


def preferred_template_order(template_bias: dict[str, Any], available: list[str]) -> list[str]:
    preferred = [t for t in template_bias.get("templates", []) if t in available]
    return list(dict.fromkeys(preferred + available))


def write_report(event: dict[str, Any]) -> None:
    lines = [
        f"# D1 Generator Latest - {event.get('timestamp_utc')}",
        "",
        f"- mode: `{event.get('mode')}`",
        f"- preview: `{event.get('preview')}`",
        f"- added: `{len(event.get('added', []) or [])}`",
        "",
        "## Template Bias Source",
        f"`{json.dumps(event.get('template_bias_source', {}), sort_keys=True)}`",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def submitted_collision_meta(expr: str, submitted_features: list[dict[str, Any]], threshold: float = MAX_SUBMITTED_SIMILARITY_GENERATE) -> tuple[bool, dict[str, Any]]:
    if not submitted_features:
        return False, {"max_similarity": 0.0, "collision_level": "low"}
    meta = score_against_submitted(expr or "", DEFAULTS, submitted_features)
    sim = float(meta.get("max_similarity", 0.0) or 0.0)
    return sim >= threshold, meta


def record_submitted_skip(state: dict[str, Any], aid: str, track: str, meta: dict[str, Any], extra: dict[str, Any] | None = None, write: bool = True) -> None:
    if not write:
        return
    payload = {
        "track": track,
        "reason": "SKIP_SUBMITTED_COLLISION",
        "submitted_similarity_max": meta.get("max_similarity", 0.0),
        "submitted_collision_level": meta.get("collision_level"),
        "submitted_nearest_alpha_id": meta.get("nearest_submitted_alpha"),
        "submitted_top_matches": meta.get("top_matches", []),
        "created_at": now_iso(),
    }
    if extra:
        payload.update(extra)
    state.setdefault("skipped", {})[aid] = payload


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
            if to_float(row.get("coverage")) < 0.40:
                continue
            rows.append(row)
    return rows


def field_lineage_score(row: dict[str, Any], lineage_hints: dict[str, Any] | None = None) -> tuple[float, list[str]]:
    if not lineage_hints:
        return 0.0, []
    weights = lineage_hints.get("explore_weight") or {}
    keyword_hints = lineage_hints.get("field_keyword_hints") or {}
    haystack = " ".join(str(row.get(k) or "") for k in ("id", "description", "dataset_id", "category", "subcategory")).lower()
    matched: list[str] = []
    bonus = 0.0
    for lineage, keywords in keyword_hints.items():
        if any(str(keyword).lower() in haystack for keyword in keywords):
            matched.append(str(lineage))
            bonus += float(weights.get(lineage, 0.0) or 0.0)
    return bonus, matched


def field_prior(row: dict[str, Any], lineage_hints: dict[str, Any] | None = None) -> float:
    crowded = to_float(row.get("userCount")) + to_float(row.get("alphaCount"))
    type_bonus = 3.0 if row.get("type") == "MATRIX" else 1.0 if row.get("type") == "VECTOR" else 0.0
    lineage_bonus, _ = field_lineage_score(row, lineage_hints)
    return (
        to_float(row.get("score"))
        + 24 * to_float(row.get("coverage"))
        + 3.5 * to_float(row.get("dateCoverage"))
        + type_bonus
        + 18.0 * lineage_bonus
        - 0.015 * crowded
    )


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


def anchor_dataset_hint(alpha_id: str, expression: str, multi_state: dict[str, Any] | None) -> set[str]:
    multi_state = multi_state or {}
    meta = (multi_state.get("alphas") or {}).get(alpha_id) or {}
    datasets = set(str(x) for x in (meta.get("datasets") or []) if x)
    field_to_dataset = {str(v.get("field_id") or v.get("id") or ""): str(v.get("dataset_id") or "") for v in multi_state.get("arms", {}).values() if isinstance(v, dict)}
    for field, dataset in field_to_dataset.items():
        if field and dataset and field in expression:
            datasets.add(dataset)
    return datasets


def load_anchors(limit: int) -> list[dict[str, Any]]:
    multi_state = load_json(ROOT / "state" / "multi_dataset_state.json")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    multi_clause = exact_prefix_clause("r.alpha_id")
    repairsc3_clause = exact_prefix_clause("r.alpha_id")
    rows = conn.execute(f"""
        select r.alpha_id, r.sharpe, r.fitness, r.turnover, coalesce(r.fail_reasons, '') as fail_reasons, t.expression
        from backtest_results r join alpha_tasks t on r.alpha_id = t.id
        where r.sharpe >= 1.2
          and r.fitness >= 0.65
          and r.turnover <= 0.50
          and (r.fail_reasons is null or trim(r.fail_reasons) = '')
          and t.expression is not null
          and ({multi_clause} or {repairsc3_clause})
        order by r.sharpe * 0.4 + r.fitness * 0.6 desc
        limit ?
    """, (exact_prefix_param("multi_"), exact_prefix_param("repairsc3_"), limit)).fetchall()
    conn.close()
    anchors: list[dict[str, Any]] = []
    for r in rows:
        anchors.append({
            "alpha_id": r["alpha_id"],
            "expression": r["expression"],
            "sharpe": r["sharpe"],
            "fitness": r["fitness"],
            "turnover": r["turnover"],
            "datasets": sorted(anchor_dataset_hint(r["alpha_id"], r["expression"], multi_state)),
        })
    return anchors


def choose_helpers(anchor: dict[str, Any], fields: list[dict[str, Any]], limit: int, require_diff_category: bool = False, lineage_hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    anchor_datasets = set(anchor.get("datasets") or [])
    anchor_categories = set(str(x) for x in (anchor.get("categories") or []) if x)
    candidates = []
    for row in fields:
        dataset = str(row.get("dataset_id") or "")
        field_id = str(row.get("id") or "")
        category = str(row.get("category") or "")
        if not dataset or not field_id:
            continue
        if dataset in anchor_datasets:
            continue
        if require_diff_category and category in anchor_categories:
            continue
        if field_id in anchor["expression"]:
            continue
        if to_float(row.get("alphaCount")) > 250:
            continue
        candidates.append((field_prior(row, lineage_hints), row))
    candidates.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen_dataset: set[str] = set()
    seen_category: set[str] = set()
    for _, row in candidates:
        dataset = str(row.get("dataset_id"))
        category = str(row.get("category") or "")
        if dataset in seen_dataset and len(out) < limit - 1:
            continue
        if require_diff_category and category in seen_category and len(out) < limit - 1:
            continue
        out.append(row)
        seen_dataset.add(dataset)
        seen_category.add(category)
        if len(out) >= limit:
            break
    return out


def choose_v1_pairs(fields: list[dict[str, Any]], panel: dict[str, Any], limit: int, lineage_hints: dict[str, Any] | None = None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ranked = sorted(fields, key=lambda row: field_prior(row, lineage_hints), reverse=True)[:60]
    pair_stats = panel.get("dataset_pair") or {}
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for i, a in enumerate(ranked):
        for b in ranked[i + 1 : min(len(ranked), i + 20)]:
            da, db = str(a.get("dataset_id") or ""), str(b.get("dataset_id") or "")
            if not da or not db or da == db:
                continue
            pair_key = "|".join(sorted([da, db]))
            ps = pair_stats.get(pair_key, {})
            self_corr_rate = float(ps.get("self_corr_block_rate", 0.0) or 0.0)
            d1_rate = float(ps.get("d1_ready_rate", 0.0) or 0.0)
            score = field_prior(a, lineage_hints) + field_prior(b, lineage_hints) + 12 * (1 - self_corr_rate) + 20 * d1_rate
            candidates.append((score, a, b))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(a, b) for _, a, b in candidates[:limit]]


def append_v2_candidates(state: dict[str, Any], anchors: list[dict[str, Any]], fields: list[dict[str, Any]], max_add: int, max_per_anchor: int, include_paused: bool = False, write: bool = True, submitted_features: list[dict[str, Any]] | None = None, lineage_hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with (ALPHAS_CSV.open("a", newline="", encoding="utf-8") if write else nullcontext(None)) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]) if write and f else None
        if writer and is_new:
            writer.writeheader()
        for anchor in anchors:
            per_anchor = 0
            helpers = choose_helpers(anchor, fields, limit=5, lineage_hints=lineage_hints)
            for helper in helpers:
                helper_expr = str(helper.get("id"))
                helper_dataset = str(helper.get("dataset_id"))
                for template_name, builder in D1_V2_TEMPLATES.items():
                    if template_name in D1_V2_PAUSED_TEMPLATES and not include_paused:
                        continue
                    expr = builder(anchor["expression"], helper_expr)
                    valid, reason = validate_expression(expr)
                    aid = "d1v2_" + hashlib.sha1(f"{anchor['alpha_id']}|{helper_dataset}|{helper_expr}|{template_name}|{expr}".encode("utf-8")).hexdigest()[:10]
                    if not valid:
                        state.setdefault("skipped", {})[aid] = {
                            "track": "d1_v2",
                            "anchor_id": anchor["alpha_id"],
                            "helper_dataset": helper_dataset,
                            "helper_field": helper_expr,
                            "template_name": template_name,
                            "reason": f"SKIP_COMPLEX: {reason}",
                            "operator_count": count_operators(expr),
                            "expression_length": len(expr or ""),
                            "created_at": now_iso(),
                        }
                        continue
                    if expr in existing_exprs or aid in existing_ids:
                        continue
                    blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                    if blocked:
                        record_submitted_skip(state, aid, "d1_v2", submitted_meta, {"anchor_id": anchor["alpha_id"], "helper_dataset": helper_dataset, "helper_field": helper_expr, "template_name": template_name}, write=write)
                        continue
                    if writer:
                        writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                    existing_exprs.add(expr)
                    existing_ids.add(aid)
                    meta = {
                        "track": "d1_v2",
                        "anchor_id": anchor["alpha_id"],
                        "helper_dataset": helper_dataset,
                        "helper_field": helper_expr,
                        "helper_lineage_hints": field_lineage_score(helper, lineage_hints)[1],
                        "template_name": template_name,
                        "template": template_name,
                        "ts_corr_count": ts_corr_count(expr),
                        "cross_dataset": 1,
                        "cross_family": 1,
                        "target_turnover_max": 0.35,
                        "target_p_self_corr_max": 0.15,
                        "anchor_sharpe": anchor["sharpe"],
                        "anchor_fitness": anchor["fitness"],
                        "anchor_turnover": anchor["turnover"],
                        "operator_count": count_operators(expr),
                        "expression_length": len(expr or ""),
                        "created_at": now_iso(),
                    }
                    if write:
                        state.setdefault("candidates", {})[aid] = meta
                    item = {"id": aid, **meta}
                    if not write:
                        item["expression"] = expr
                    added.append(item)
                    per_anchor += 1
                    if len(added) >= max_add:
                        return added
                    if per_anchor >= max_per_anchor:
                        break
                if per_anchor >= max_per_anchor:
                    break
    return added


def load_v22_anchors(limit: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        select r.alpha_id, r.sharpe, r.fitness, r.turnover, coalesce(r.fail_reasons, '') as fail_reasons, t.expression
        from backtest_results r join alpha_tasks t on r.alpha_id = t.id
        where r.alpha_id in ('d1v2_a1abb364db', 'd1v2_81bbb2ad52', 'd1v2_802708cd94')
          and r.sharpe >= 1.6
          and r.fitness >= 1.0
          and r.turnover <= 0.45
          and (r.fail_reasons is null or trim(r.fail_reasons) = '')
        order by r.fitness desc, r.sharpe desc
        limit ?
    """, (limit,)).fetchall()
    conn.close()
    # v2.2 intentionally does not embed full anchor expressions because the clean
    # v2 anchors still contain ts_corr-heavy lineage. Preserve only coarse signal
    # direction/rank through a small anchor proxy so new candidates can satisfy
    # the no-ts_corr and low-complexity constraints.
    transforms = {
        "d1v2_a1abb364db": lambda e: "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))",
        "d1v2_81bbb2ad52": lambda e: "sign(ts_mean(rank(earnings_momentum_analyst_score) - rank(returns), 3))",
        "d1v2_802708cd94": lambda e: "rank(earnings_momentum_analyst_score)",
    }
    anchors: list[dict[str, Any]] = []
    for r in rows:
        aid = str(r["alpha_id"])
        expr = str(r["expression"] or "")
        anchors.append({
            "alpha_id": aid,
            "expression": transforms.get(aid, lambda x: f"rank({x})")(expr),
            "raw_expression": expr,
            "sharpe": r["sharpe"],
            "fitness": r["fitness"],
            "turnover": r["turnover"],
            "datasets": [],
            "categories": ["Model"],
        })
    return anchors


def validate_v22_expression(expr: str) -> tuple[bool, str]:
    if "ts_corr" in (expr or ""):
        return False, "v2.2 forbids ts_corr"
    op_count = count_operators(expr)
    if op_count > MAX_V22_OPERATORS:
        return False, f"operator_count={op_count} > {MAX_V22_OPERATORS}"
    if len(expr or "") > MAX_EXPR_LEN:
        return False, f"expression_length={len(expr or '')} > {MAX_EXPR_LEN}"
    return True, "ok"


def append_v22_candidates(state: dict[str, Any], anchors: list[dict[str, Any]], fields: list[dict[str, Any]], max_add: int, max_per_anchor: int, write: bool = True, submitted_features: list[dict[str, Any]] | None = None, lineage_hints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with (ALPHAS_CSV.open("a", newline="", encoding="utf-8") if write else nullcontext(None)) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]) if write and f else None
        if writer and is_new:
            writer.writeheader()
        for anchor in anchors:
            per_anchor = 0
            helpers = choose_helpers(anchor, fields, limit=4, require_diff_category=True, lineage_hints=lineage_hints)
            for helper in helpers:
                helper_expr = str(helper.get("id"))
                helper_dataset = str(helper.get("dataset_id"))
                helper_category = str(helper.get("category") or "")
                for template_name, builder in D1_V22_TEMPLATES.items():
                    expr = builder(anchor["expression"], helper_expr)
                    valid, reason = validate_v22_expression(expr)
                    aid = "d1v22_" + hashlib.sha1(f"{anchor['alpha_id']}|{helper_dataset}|{helper_expr}|{template_name}|{expr}".encode("utf-8")).hexdigest()[:10]
                    if not valid:
                        state.setdefault("skipped", {})[aid] = {"track": "d1_v2_2", "anchor_id": anchor["alpha_id"], "helper_dataset": helper_dataset, "helper_field": helper_expr, "template_name": template_name, "reason": f"SKIP_COMPLEX: {reason}", "operator_count": count_operators(expr), "expression_length": len(expr or ""), "created_at": now_iso()}
                        continue
                    if expr in existing_exprs or aid in existing_ids:
                        continue
                    blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                    if blocked:
                        record_submitted_skip(state, aid, "d1_v2_2", submitted_meta, {"anchor_id": anchor["alpha_id"], "helper_dataset": helper_dataset, "helper_field": helper_expr, "template_name": template_name}, write=write)
                        continue
                    if writer:
                        writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                    existing_exprs.add(expr)
                    existing_ids.add(aid)
                    meta = {
                        "track": "d1_v2_2",
                        "anchor_id": anchor["alpha_id"],
                        "helper_dataset": helper_dataset,
                        "helper_field": helper_expr,
                        "helper_category": helper_category,
                        "helper_lineage_hints": field_lineage_score(helper, lineage_hints)[1],
                        "template_name": template_name,
                        "template": template_name,
                        "ts_corr_count": ts_corr_count(expr),
                        "cross_dataset": 1,
                        "cross_family": 0,
                        "has_differenced_neutralizer": 1 if " - " in expr else 0,
                        "regime_conditioned": 1 if "sign(ts_mean" in expr else 0,
                        "target_turnover_max": 0.35,
                        "target_p_self_corr_max": 0.15,
                        "anchor_sharpe": anchor["sharpe"],
                        "anchor_fitness": anchor["fitness"],
                        "anchor_turnover": anchor["turnover"],
                        "operator_count": count_operators(expr),
                        "expression_length": len(expr or ""),
                        "created_at": now_iso(),
                    }
                    if write:
                        state.setdefault("candidates", {})[aid] = meta
                    item = {"id": aid, **meta}
                    if not write:
                        item["expression"] = expr
                    added.append(item)
                    per_anchor += 1
                    if len(added) >= max_add:
                        return added
                    if per_anchor >= max_per_anchor:
                        break
                if per_anchor >= max_per_anchor:
                    break
    return added


def load_v23_anchors(limit: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        select r.alpha_id, r.sharpe, r.fitness, r.turnover, coalesce(r.fail_reasons, '') as fail_reasons, t.expression
        from backtest_results r join alpha_tasks t on r.alpha_id = t.id
        where r.alpha_id in ('d1v2_a1abb364db', 'd1v2_81bbb2ad52', 'd1v2_802708cd94')
          and r.sharpe >= 1.6
          and r.fitness >= 1.0
          and r.turnover <= 0.45
          and (r.fail_reasons is null or trim(r.fail_reasons) = '')
          and t.expression is not null
        order by r.fitness desc, r.sharpe desc
        limit ?
    """, (limit,)).fetchall()
    conn.close()
    anchors: list[dict[str, Any]] = []
    for r in rows:
        expr = str(r["expression"] or "")
        corr_calls = find_corr_calls(expr)
        if not any(c.get("is_price_target") for c in corr_calls):
            continue
        anchors.append({
            "alpha_id": str(r["alpha_id"]),
            "expression": expr,
            "sharpe": r["sharpe"],
            "fitness": r["fitness"],
            "turnover": r["turnover"],
            "corr_calls": corr_calls,
            "datasets": [],
            "categories": ["Model"],
        })
    return anchors


def validate_v23_expression(expr: str) -> tuple[bool, str]:
    corr_calls = find_corr_calls(expr)
    if any(c.get("is_price_target") for c in corr_calls):
        return False, "v2.3 still has price/returns ts_corr target"
    if len(corr_calls) > MAX_TS_CORR_COUNT:
        return False, f"ts_corr_count={len(corr_calls)} > {MAX_TS_CORR_COUNT}"
    op_count = count_operators(expr)
    if op_count > MAX_V23_OPERATORS:
        return False, f"operator_count={op_count} > {MAX_V23_OPERATORS}"
    if len(expr or "") > MAX_EXPR_LEN:
        return False, f"expression_length={len(expr or '')} > {MAX_EXPR_LEN}"
    return True, "ok"


def append_v23_candidates(state: dict[str, Any], anchors: list[dict[str, Any]], max_add: int, targets: list[str] | None = None, write: bool = True, submitted_features: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    targets = targets or V23_SAFE_TARGETS[:2]
    with (ALPHAS_CSV.open("a", newline="", encoding="utf-8") if write else nullcontext(None)) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]) if write and f else None
        if writer and is_new:
            writer.writeheader()
        for anchor in anchors:
            original_expr = str(anchor["expression"] or "")
            original_corr_calls = find_corr_calls(original_expr)
            for target in targets:
                expr = replace_corr_targets(original_expr, target)
                valid, reason = validate_v23_expression(expr)
                aid = "d1v23_" + hashlib.sha1(f"{anchor['alpha_id']}|{target}|v23_surgical_target_replace|{expr}".encode("utf-8")).hexdigest()[:10]
                if not valid:
                    state.setdefault("skipped", {})[aid] = {"track": "d1_v2_3", "anchor_id": anchor["alpha_id"], "corr_target_replaced": target, "template_name": "v23_surgical_target_replace", "reason": f"SKIP_COMPLEX: {reason}", "operator_count": count_operators(expr), "expression_length": len(expr or ""), "created_at": now_iso()}
                    continue
                if expr in existing_exprs or aid in existing_ids:
                    continue
                blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                if blocked:
                    record_submitted_skip(state, aid, "d1_v2_3", submitted_meta, {"anchor_id": anchor["alpha_id"], "corr_target_replaced": target, "template_name": "v23_surgical_target_replace"}, write=write)
                    continue
                if writer:
                    writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(aid)
                meta = {
                    "track": "d1_v2_3",
                    "anchor_id": anchor["alpha_id"],
                    "template_name": "v23_surgical_target_replace",
                    "template": "v23_surgical_target_replace",
                    "corr_target_replaced": target,
                    "original_corr_targets": [c.get("field_b") for c in original_corr_calls],
                    "ts_corr_count": ts_corr_count(expr),
                    "is_price_corr": 0,
                    "cross_dataset": 1,
                    "cross_family": 1,
                    "has_differenced_neutralizer": 1 if " - " in expr else 0,
                    "regime_conditioned": 1 if "trade_when" in expr or "sign(ts_mean" in expr else 0,
                    "target_turnover_max": 0.35,
                    "target_p_self_corr_max": 0.15,
                    "anchor_sharpe": anchor["sharpe"],
                    "anchor_fitness": anchor["fitness"],
                    "anchor_turnover": anchor["turnover"],
                    "operator_count": count_operators(expr),
                    "expression_length": len(expr or ""),
                    "created_at": now_iso(),
                }
                if write:
                    state.setdefault("candidates", {})[aid] = meta
                item = {"id": aid, **meta}
                if not write:
                    item["expression"] = expr
                added.append(item)
                if len(added) >= max_add:
                    return added
    return added


def load_v24_anchors(limit: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    d1v23_clause = exact_prefix_clause("r.alpha_id")
    rows = conn.execute(f"""
        select r.alpha_id, r.sharpe, r.fitness, r.turnover, coalesce(r.fail_reasons, '') as fail_reasons, t.expression
        from backtest_results r join alpha_tasks t on r.alpha_id = t.id
        where (
                {d1v23_clause}
                or r.alpha_id in ('d1v2_a1abb364db', 'd1v2_81bbb2ad52', 'd1v2_802708cd94')
              )
          and r.sharpe >= 1.2
          and r.fitness >= 0.65
          and r.turnover <= 0.50
          and (r.fail_reasons is null or trim(r.fail_reasons) = '')
          and t.expression is not null
        order by r.fitness desc, r.sharpe desc
        limit ?
    """, (exact_prefix_param("d1v23_"), limit * 3)).fetchall()
    conn.close()

    anchors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        expr = replace_corr_targets(str(r["expression"] or ""), "adv20")
        valid, _ = validate_v24_expression(expr)
        if not valid:
            continue
        aid = str(r["alpha_id"])
        if expr in seen:
            continue
        seen.add(expr)
        anchors.append({
            "alpha_id": aid,
            "expression": expr,
            "raw_expression": str(r["expression"] or ""),
            "sharpe": r["sharpe"],
            "fitness": r["fitness"],
            "turnover": r["turnover"],
            "corr_calls": find_corr_calls(expr),
            "datasets": [],
            "categories": ["Model"],
        })
        if len(anchors) >= limit:
            break
    if anchors:
        return anchors
    return load_v23_anchors(limit)


def shift_helper_weights(expr: str) -> str:
    replacements = [
        ("0.35", "0.24"),
        ("0.30", "0.22"),
        ("0.25", "0.18"),
        ("0.22", "0.16"),
        ("0.18", "0.12"),
        ("0.16", "0.11"),
    ]
    out = expr
    for old, new in replacements:
        if old in out:
            out = out.replace(old, new, 1)
            break
    return out


def flatten_neutralization(expr: str) -> str:
    out = expr
    out = re.sub(r"group_neutralize\(\s*rank\(([^()]+)\)\s*,\s*subindustry\s*\)", r"rank(group_neutralize(\1, subindustry))", out, count=1)
    if out != expr:
        return out
    if ", subindustry)" in out:
        return out.replace(", subindustry)", f", {SIZE_BUCKET})", 1)
    return out


def return_proxy_light(expr: str) -> str:
    proxy = "((vwap - open) / open)"
    return re.sub(r"\breturns\b", proxy, expr, count=1)


def v24_variants(expr: str, mode: str) -> list[tuple[str, str]]:
    clean = replace_corr_targets(expr, "adv20")
    variants = [
        ("corr_arg_rotate", replace_all_corr_targets(clean, "volume")),
        ("return_proxy_light", return_proxy_light(clean)),
        ("neutralization_flatten", flatten_neutralization(clean)),
        ("helper_weight_shift", shift_helper_weights(clean)),
        ("safe_target_expand_adv60", replace_safe_corr_targets(clean, "adv60")),
    ]
    if mode == "v2.4":
        variants.extend([
            ("safe_target_expand_volume", replace_safe_corr_targets(clean, "volume")),
            ("safe_target_expand_adv20", replace_safe_corr_targets(clean, "adv20")),
        ])
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, candidate in variants:
        candidate = candidate.strip()
        if not candidate or candidate == expr.strip() or candidate in seen:
            continue
        seen.add(candidate)
        out.append((name, candidate))
    return out


def validate_v24_expression(expr: str) -> tuple[bool, str]:
    corr_calls = find_corr_calls(expr)
    if any(c.get("is_price_target") for c in corr_calls):
        return False, "v2.3b/v2.4 forbids price/returns as ts_corr second arg"
    if len(corr_calls) > MAX_TS_CORR_COUNT:
        return False, f"ts_corr_count={len(corr_calls)} > {MAX_TS_CORR_COUNT}"
    op_count = count_operators(expr)
    if op_count > MAX_V23_OPERATORS:
        return False, f"operator_count={op_count} > {MAX_V23_OPERATORS}"
    if len(expr or "") > MAX_EXPR_LEN:
        return False, f"expression_length={len(expr or '')} > {MAX_EXPR_LEN}"
    return True, "ok"


def append_v24_candidates(state: dict[str, Any], anchors: list[dict[str, Any]], max_add: int, mode: str, write: bool = True, submitted_features: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    prefix = "d1v24_" if mode == "v2.4" else "d1v23b_"
    track = "d1_v2_4" if mode == "v2.4" else "d1_v2_3b"
    with (ALPHAS_CSV.open("a", newline="", encoding="utf-8") if write else nullcontext(None)) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]) if write and f else None
        if writer and is_new:
            writer.writeheader()
        for anchor in anchors:
            original_expr = str(anchor["expression"] or "")
            original_corr_calls = find_corr_calls(original_expr)
            for transform_name, expr in v24_variants(original_expr, mode):
                valid, reason = validate_v24_expression(expr)
                aid = prefix + hashlib.sha1(f"{anchor['alpha_id']}|{transform_name}|{expr}".encode("utf-8")).hexdigest()[:10]
                if not valid:
                    if write:
                        state.setdefault("skipped", {})[aid] = {"track": track, "anchor_id": anchor["alpha_id"], "transform_name": transform_name, "reason": f"SKIP_COMPLEX: {reason}", "operator_count": count_operators(expr), "expression_length": len(expr or ""), "ts_corr_count": ts_corr_count(expr), "created_at": now_iso()}
                    continue
                if expr in existing_exprs or aid in existing_ids:
                    continue
                blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                if blocked:
                    record_submitted_skip(state, aid, track, submitted_meta, {"anchor_id": anchor["alpha_id"], "transform_name": transform_name}, write=write)
                    continue
                if writer:
                    writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(aid)
                new_corr_calls = find_corr_calls(expr)
                meta = {
                    "track": track,
                    "anchor_id": anchor["alpha_id"],
                    "template_name": transform_name,
                    "template": transform_name,
                    "original_corr_targets": [c.get("field_b") for c in original_corr_calls],
                    "new_corr_targets": [c.get("field_b") for c in new_corr_calls],
                    "ts_corr_count": len(new_corr_calls),
                    "is_price_corr": 0,
                    "cross_dataset": 1,
                    "cross_family": 1,
                    "lineage_reduction_transform": transform_name,
                    "has_differenced_neutralizer": 1 if " - " in expr else 0,
                    "regime_conditioned": 1 if "trade_when" in expr or "sign(ts_mean" in expr else 0,
                    "target_turnover_max": 0.35,
                    "target_p_self_corr_max": 0.15,
                    "anchor_sharpe": anchor["sharpe"],
                    "anchor_fitness": anchor["fitness"],
                    "anchor_turnover": anchor["turnover"],
                    "operator_count": count_operators(expr),
                    "expression_length": len(expr or ""),
                    "created_at": now_iso(),
                }
                if write:
                    state.setdefault("candidates", {})[aid] = meta
                item = {"id": aid, **meta}
                if not write:
                    item["expression"] = expr
                added.append(item)
                if len(added) >= max_add:
                    return added
    return added


def append_v1_candidates(state: dict[str, Any], pairs: list[tuple[dict[str, Any], dict[str, Any]]], max_add: int, write: bool = True, submitted_features: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    submitted_features = submitted_features or []
    existing_exprs, existing_ids = existing_csv()
    is_new = not ALPHAS_CSV.exists() or ALPHAS_CSV.stat().st_size == 0
    added: list[dict[str, Any]] = []
    with (ALPHAS_CSV.open("a", newline="", encoding="utf-8") if write else nullcontext(None)) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]) if write and f else None
        if writer and is_new:
            writer.writeheader()
        for a, b in pairs:
            fa, fb = str(a.get("id")), str(b.get("id"))
            da, db = str(a.get("dataset_id")), str(b.get("dataset_id"))
            for structure_name, builder in D1_V1_STRUCTURES.items():
                expr = builder(fa, fb)
                aid = "d1_" + hashlib.sha1(f"v1|{da}|{fa}|{db}|{fb}|{structure_name}|{expr}".encode("utf-8")).hexdigest()[:10]
                if expr in existing_exprs or aid in existing_ids:
                    continue
                blocked, submitted_meta = submitted_collision_meta(expr, submitted_features)
                if blocked:
                    record_submitted_skip(state, aid, "d1_v1", submitted_meta, {"primary_field": fa, "secondary_field": fb, "structure_name": structure_name}, write=write)
                    continue
                if writer:
                    writer.writerow({"id": aid, "expression": expr, **DEFAULTS})
                existing_exprs.add(expr)
                existing_ids.add(aid)
                meta = {"track": "d1_v1", "primary_dataset": da, "secondary_dataset": db, "primary_field": fa, "secondary_field": fb, "structure_name": structure_name, "target_turnover_max": 0.35, "target_p_self_corr_max": 0.15, "operator_count": count_operators(expr), "expression_length": len(expr or ""), "ts_corr_count": ts_corr_count(expr), "created_at": now_iso()}
                if write:
                    state.setdefault("candidates", {})[aid] = meta
                item = {"id": aid, **meta}
                if not write:
                    item["expression"] = expr
                added.append(item)
                if len(added) >= max_add:
                    return added
    return added


def main() -> int:
    global D1_V2_TEMPLATES, D1_V22_TEMPLATES
    parser = argparse.ArgumentParser(description="Generate D1-focused low-self-correlation candidates")
    parser.add_argument("--max-add", type=int, default=6)
    parser.add_argument("--pair-limit", type=int, default=4)
    parser.add_argument("--anchor-limit", type=int, default=3)
    parser.add_argument("--max-per-anchor", type=int, default=3)
    parser.add_argument("--mode", choices=["v2", "v1", "v2.2", "v2.3", "v2.3b", "v2.4"], default="v2")
    parser.add_argument("--preview", action="store_true", help="Print candidate expressions/metadata without writing alphas.csv or state")
    args = parser.parse_args()

    state = load_json(STATE_PATH) or {"version": 2, "created_at": now_iso(), "candidates": {}, "history": []}
    state["version"] = max(int(state.get("version", 1)), 2)
    state.setdefault("_schema_version", "2.4")
    template_bias = load_template_bias(state)
    D1_V2_TEMPLATES = {name: D1_V2_TEMPLATES[name] for name in preferred_template_order(template_bias, list(D1_V2_TEMPLATES))}
    D1_V22_TEMPLATES = {name: D1_V22_TEMPLATES[name] for name in preferred_template_order(template_bias, list(D1_V22_TEMPLATES))}
    fields = load_fields()
    submitted_features = load_submitted_features()
    lineage_hints = build_lineage_hints(submitted_features) if submitted_features else load_lineage_hints()
    if args.mode == "v1":
        panel = load_json(PANEL_PATH)
        pairs = choose_v1_pairs(fields, panel, args.pair_limit, lineage_hints=lineage_hints)
        added = append_v1_candidates(state, pairs, args.max_add, write=not args.preview, submitted_features=submitted_features)
        event: dict[str, Any] = {"timestamp_utc": now_iso(), "mode": "v1", "selected_pairs": [[str(a.get("dataset_id")), str(a.get("id")), str(b.get("dataset_id")), str(b.get("id"))] for a, b in pairs], "added": added}
    elif args.mode == "v2.2":
        anchors = load_v22_anchors(args.anchor_limit)
        added = append_v22_candidates(state, anchors, fields, args.max_add, args.max_per_anchor, write=not args.preview, submitted_features=submitted_features, lineage_hints=lineage_hints)
        event = {"timestamp_utc": now_iso(), "mode": "v2.2", "selected_anchors": [{k: a[k] for k in ["alpha_id", "sharpe", "fitness", "turnover"]} for a in anchors], "added": added}
    elif args.mode == "v2.3":
        anchors = load_v23_anchors(args.anchor_limit)
        added = append_v23_candidates(state, anchors, args.max_add, V23_SAFE_TARGETS[:2], write=not args.preview, submitted_features=submitted_features)
        event = {"timestamp_utc": now_iso(), "mode": "v2.3", "selected_anchors": [{k: a[k] for k in ["alpha_id", "sharpe", "fitness", "turnover"]} for a in anchors], "added": added}
    elif args.mode in {"v2.3b", "v2.4"}:
        anchors = load_v24_anchors(args.anchor_limit)
        added = append_v24_candidates(state, anchors, args.max_add, args.mode, write=not args.preview, submitted_features=submitted_features)
        event = {"timestamp_utc": now_iso(), "mode": args.mode, "preview": args.preview, "selected_anchors": [{k: a[k] for k in ["alpha_id", "sharpe", "fitness", "turnover"]} for a in anchors], "added": added}
    else:
        anchors = load_anchors(args.anchor_limit)
        added = append_v2_candidates(state, anchors, fields, args.max_add, args.max_per_anchor, write=not args.preview, submitted_features=submitted_features, lineage_hints=lineage_hints)
        event = {"timestamp_utc": now_iso(), "mode": "v2", "selected_anchors": [{k: a[k] for k in ["alpha_id", "sharpe", "fitness", "turnover", "datasets"]} for a in anchors], "added": added}
    event["preview"] = args.preview
    event["template_bias_source"] = template_bias
    event["submitted_filter"] = {"threshold": MAX_SUBMITTED_SIMILARITY_GENERATE, "submitted_library_rows": len(submitted_features)}
    event["lineage_hints"] = {"top": (lineage_hints.get("ranked_lineages") or [])[:8]}
    if not args.preview:
        state.setdefault("history", []).append(event)
        state["history"] = state["history"][-100:]
        save_json(STATE_PATH, state)
    write_report(event)
    print(json.dumps(event, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
