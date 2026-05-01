#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

from alpha_factory.expr_parser import expression_has_token, ts_corr_calls
from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param, exact_prefix_where
from scripts.submitted_similarity import load_submitted_features, score_against_submitted

DB_PATH = ROOT / "data" / "backtests.sqlite3"
STATE_PATH = ROOT / "state" / "ml_candidate_scorer_state.json"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
SUBMISSIONS_PATH = ROOT / "data" / "auto_submissions.jsonl"
SUPER_STATE_PATH = ROOT / "state" / "superalpha_state.json"
MULTI_STATE_PATH = ROOT / "state" / "multi_dataset_state.json"
D1_STATE_PATH = ROOT / "state" / "d1_generator_state.json"
TRUTH_TABLE_PATH = ROOT / "state" / "self_corr_truth_table.json"
REPORT_PATH = ROOT / "reports" / "ml_candidate_scorer_report.md"

PREFIXES = ["d1v23_", "d1v22_", "d1v2_", "d1_", "repairfit_", "multi_", "super_", "arm_", "repairsc2_", "repairsc3_", "decor_", "refit_", "supersc_"]
SELF_CORR_PATTERNS = ("SELF_CORRELATION", "self-correlation cooldown active")
PRICE_CORR_TARGETS = {"close", "open", "returns", "ret", "daily_return", "price", "vwap", "high", "low"}
PASS_THRESHOLD = {"sharpe": 1.6, "fitness": 1.0, "turnover": 0.45}
UNDER_SAMPLED_LINEAGES = {"fundamental_valuation", "market_size", "price_volume", "liquidity_volatility"}
LINEAGE_PRIOR_DISCOUNT = {
    "fundamental_valuation": 0.65,
    "market_size": 0.70,
    "liquidity_volatility": 0.75,
    "price_volume": 0.80,
    "analyst_earnings": 1.00,
    "mixed": 1.00,
}
OPERATOR_PATTERNS = {
    "rank": re.compile(r"\brank\s*\("),
    "zscore": re.compile(r"\b(?:zscore|ts_zscore)\s*\("),
    "ts_rank": re.compile(r"\bts_rank\s*\("),
    "decay": re.compile(r"\b(?:decay|ts_decay_linear)\s*\("),
    "delta": re.compile(r"\b(?:delta|ts_delta)\s*\("),
    "corr": re.compile(r"\b(?:corr|ts_corr)\s*\("),
    "residual": re.compile(r"\bresidual\w*\s*\("),
    "neutralize": re.compile(r"\b(?:neutralize|group_neutralize)\s*\("),
    "median": re.compile(r"\b(?:median|ts_median)\s*\("),
    "log": re.compile(r"\blog\s*\("),
    "sign": re.compile(r"\bsign\s*\("),
    "abs": re.compile(r"\babs\s*\("),
    "power": re.compile(r"\b(?:power|signed_power)\s*\("),
    "min": re.compile(r"\b(?:min|ts_min)\s*\("),
    "max": re.compile(r"\b(?:max|ts_max)\s*\("),
    "if_else": re.compile(r"\bif_else\s*\("),
}
FIELD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
IGNORED_FIELD_TOKENS = {
    "abs", "bucket", "group_mean", "group_neutralize", "rank", "range", "signed_power",
    "trade_when", "ts_corr", "ts_decay_linear", "ts_delta", "ts_mean", "ts_median",
    "ts_rank", "ts_std_dev", "ts_zscore", "if_else", "log", "sign", "nan", "inf",
    "true", "false", "industry", "market", "sector", "subindustry",
}
REPAIR_PREFIXES = ("repairsc_", "repairsc2_", "repairsc3_")


def split_top_level_args(arg_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(arg_text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(arg_text[start:i].strip())
            start = i + 1
    args.append(arg_text[start:].strip())
    return args


def ts_corr_targets(expr: str) -> list[str]:
    return [str(call.get("field_b") or "").strip() for call in ts_corr_calls(expr or "")]


def corr_target_flags(expr: str) -> tuple[float, float]:
    targets = [t.strip() for t in ts_corr_targets(expr)]
    returns = 1.0 if any(expression_has_token(t, {"returns", "ret", "daily_return"}) for t in targets) else 0.0
    price = 1.0 if any(expression_has_token(t, PRICE_CORR_TARGETS) for t in targets) else 0.0
    return returns, price


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def family(alpha_id: str) -> str:
    return alpha_id.split("_", 1)[0] + "_" if "_" in alpha_id else alpha_id


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_truth_table(path: Path = TRUTH_TABLE_PATH) -> dict[str, Any]:
    payload = load_json(path)
    rows = payload.get("alphas") or []
    if not isinstance(rows, list):
        rows = []
    index = {str(row.get("alpha_id")): row for row in rows if isinstance(row, dict) and row.get("alpha_id")}
    return {"summary": payload.get("summary") or {}, "index": index}


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


def blocked_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for alpha_id, reasons in blocked_reasons().items():
        counts[alpha_id] = sum(
            1
            for reason in reasons
            if (
                ("unsafe checks:" in reason and "SELF_CORRELATION:" in reason)
                or "self-correlation cooldown active" in reason
            )
        )
    return counts


def truth_blocked_counts(truth_index: dict[str, dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for alpha_id, row in truth_index.items():
        skip_counts = row.get("skip_counts") or {}
        if isinstance(skip_counts, dict):
            counts[alpha_id] = sum(
                int(v)
                for k, v in skip_counts.items()
                if k in {"self_corr_pending", "self_corr_blocked", "self_corr_cooldown", "pre_submit_self_corr_gate"}
            )
    return counts


def self_corr_labels() -> dict[str, int]:
    labels: dict[str, int] = {}
    for alpha_id, reasons in blocked_reasons().items():
        if any(
            ("unsafe checks:" in reason and "SELF_CORRELATION:" in reason)
            or "self-correlation cooldown active" in reason
            for reason in reasons
        ):
            labels[alpha_id] = 1
    if SUBMISSIONS_PATH.exists():
        for line in SUBMISSIONS_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-2000:]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            local_id = str(payload.get("local_id") or "")
            status = str(payload.get("status") or "").lower()
            if local_id and status in {"submitted", "already_submitted"}:
                labels.setdefault(local_id, 0)
    return labels


def truth_self_corr_labels(truth_index: dict[str, dict[str, Any]]) -> dict[str, int]:
    labels: dict[str, int] = {}
    for alpha_id, row in truth_index.items():
        status = str(row.get("self_corr_status") or "unknown")
        if status in {"blocked", "pending", "cooldown"}:
            labels[alpha_id] = 1
        elif status == "clear":
            labels[alpha_id] = 0
    return labels


def expr_features(expr: str) -> dict[str, float]:
    e = expr or ""
    corr_returns, corr_price = corr_target_flags(e)
    op_counts = [len(pattern.findall(e)) for pattern in OPERATOR_PATTERNS.values()]
    op_total = sum(op_counts)
    operator_entropy = 0.0
    if op_total:
        operator_entropy = -sum((count / op_total) * math.log2(count / op_total) for count in op_counts if count)
    return {
        "has_group_neutralize": 1.0 if "group_neutralize" in e else 0.0,
        "has_trade_when": 1.0 if "trade_when" in e else 0.0,
        "has_ts_zscore": 1.0 if "ts_zscore" in e else 0.0,
        "has_ts_rank": 1.0 if "ts_rank" in e else 0.0,
        "has_corr": 1.0 if "ts_corr" in e else 0.0,
        "has_decay": 1.0 if "ts_decay_linear" in e else 0.0,
        "has_delta": 1.0 if "ts_delta" in e else 0.0,
        "has_bucket": 1.0 if "bucket(" in e else 0.0,
        "has_subindustry": 1.0 if "subindustry" in e else 0.0,
        "has_market": 1.0 if "market" in e else 0.0,
        "neutralization_depth": min(3.0, float(e.count("group_neutralize"))),
        "trade_when_depth": min(2.0, float(e.count("trade_when"))),
        "corr_target_returns": corr_returns,
        "corr_target_price": corr_price,
        "operator_entropy": operator_entropy,
        "uses_residual_or_neutralize": 1.0 if re.search(r"\b(?:residual\w*|neutralize|group_neutralize)\s*\(", e) else 0.0,
        "length_penalty": min(1.0, len(e) / 900.0),
        "complexity": min(1.0, (e.count("(") + e.count("+") + e.count("-") + e.count("*")) / 80.0),
    }


def field_diversity_index(expr: str, datasets_or_fields: list[Any] | None = None) -> float:
    fields = {str(x).strip().lower() for x in (datasets_or_fields or []) if str(x).strip()}
    for token in FIELD_RE.findall(expr or ""):
        t = token.lower()
        if t in IGNORED_FIELD_TOKENS or t.isdigit():
            continue
        fields.add(t)
    return float(len(fields))


def repair_depth_from_parent_ids(parent_ids: list[Any]) -> int:
    return sum(1 for parent_id in parent_ids if str(parent_id).startswith(REPAIR_PREFIXES))


def lineage_features(alpha_id: str) -> dict[str, float]:
    aid = str(alpha_id)
    meta = (load_json(SUPER_STATE_PATH).get("built_from") or {}).get(aid, {})
    multi_state = load_json(MULTI_STATE_PATH)
    multi_meta = (multi_state.get("alphas") or {}).get(aid, {})
    d1_meta = (load_json(D1_STATE_PATH).get("candidates") or {}).get(aid, {})

    parent_families = list(meta.get("parent_families") or [])
    parent_ids = list(meta.get("parent_alpha_ids") or [])
    datasets = list(multi_meta.get("datasets") or [])

    if d1_meta:
        anchor_id = str(d1_meta.get("anchor_id") or "")
        if anchor_id:
            parent_ids.append(anchor_id)
            parent_families.append(family(anchor_id))
        helper_dataset = str(d1_meta.get("helper_dataset") or "")
        anchor_datasets = d1_meta.get("anchor_datasets") or []
        if helper_dataset:
            datasets.append(helper_dataset)
        datasets.extend(str(x) for x in anchor_datasets if x)

    cross_family = 1.0 if len(set(parent_families)) >= 2 or to_float(d1_meta.get("cross_family")) > 0 else 0.0
    cross_dataset = 1.0 if len(set(datasets)) >= 2 or to_float(d1_meta.get("cross_dataset")) > 0 else 0.0
    return {
        "has_parent_metadata": 1.0 if parent_ids or d1_meta else 0.0,
        "cross_family_parents": cross_family,
        "cross_dataset_parents": cross_dataset,
        "parent_count": float(len(set(parent_ids))),
        "has_differenced_neutralizer": 1.0 if to_float(d1_meta.get("has_differenced_neutralizer")) > 0 else 0.0,
        "regime_conditioned": 1.0 if to_float(d1_meta.get("regime_conditioned")) > 0 else 0.0,
        "is_d1_v22": 1.0 if str(d1_meta.get("track") or "") == "d1_v2_2" else 0.0,
        "is_d1_v23": 1.0 if str(d1_meta.get("track") or "") == "d1_v2_3" else 0.0,
    }


def lineage_theme_shares(truth_summary: dict[str, Any]) -> dict[str, float]:
    counts = truth_summary.get("lineage_theme_counts") or {}
    pass_total = sum(int(v.get("pass_quality", 0) or 0) for v in counts.values() if isinstance(v, dict))
    if pass_total <= 0:
        return {}
    return {str(k): int(v.get("pass_quality", 0) or 0) / pass_total for k, v in counts.items() if isinstance(v, dict)}


def beta_mean(successes: int, total: int, prior_mean: float, prior_strength: float = 16.0) -> float:
    if total <= 0:
        return prior_mean
    prior_mean = min(1.0, max(0.0, prior_mean))
    return (successes + prior_mean * prior_strength) / (total + prior_strength)


def score_row(
    row: sqlite3.Row,
    block: Counter[str],
    labels: dict[str, int],
    family_stats: dict[str, dict[str, float]],
    truth_index: dict[str, dict[str, Any]] | None = None,
    theme_shares: dict[str, float] | None = None,
    submitted_features: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    aid = str(row["alpha_id"])
    fam = family(aid)
    sharpe = to_float(row["sharpe"])
    fitness = to_float(row["fitness"])
    turnover = to_float(row["turnover"], 99)
    fail = str(row["fail_reasons"] or "")
    expression = str(row["expression"] or "")
    x = expr_features(expression)
    x.update(lineage_features(aid))
    truth_row = (truth_index or {}).get(aid, {})
    lineage = truth_row.get("lineage") or {}
    lineage_theme = str(lineage.get("theme") or "unknown")
    datasets_or_fields = list(lineage.get("datasets_or_fields") or [])
    x["field_diversity_index"] = field_diversity_index(expression, datasets_or_fields)
    x["diversity_high"] = 1.0 if x["field_diversity_index"] > 3 else 0.0
    status = str(truth_row.get("self_corr_status") or "unknown")
    repair_depth = int(to_float(truth_row.get("repair_depth"), repair_depth_from_parent_ids(list(lineage.get("parent_ids") or []))))
    theme_share = float((theme_shares or {}).get(lineage_theme, 0.0))
    fs = family_stats.get(fam, {})
    family_pass_rate = fs.get("pass_rate", 0.0)
    family_self_corr_rate = fs.get("self_corr_rate", family_stats.get("__global__", {}).get("self_corr_rate", 0.25))
    quality = 0.42 * sharpe + 0.52 * fitness - 0.45 * max(0, turnover - 0.25)
    novelty = 0.12 * x["has_group_neutralize"] + 0.08 * x["has_trade_when"] + 0.08 * x["has_ts_zscore"] + 0.10 * x["has_corr"]
    novelty += 0.10 * x["has_decay"] + 0.08 * x["has_delta"] + 0.06 * x["cross_family_parents"] + 0.10 * x["cross_dataset_parents"]
    if fam in {"multi_", "super_", "arm_"}:
        novelty += 0.35
    if fam in {"repairsc2_", "repairsc3_", "supersc_"}:
        novelty += 0.25

    risk_components = {
        "wq_self_corr_failure": 0.58 if labels.get(aid) == 1 or "SELF_CORRELATION" in fail else 0.0,
        "cooldown_history": min(0.45, block[aid] * 0.11),
        "family_self_corr_history": min(0.34, family_self_corr_rate * 0.58),
        "truth_status_blocked": 0.82 if status == "blocked" else 0.0,
        "truth_status_pending": 0.66 if status == "pending" else 0.0,
        "truth_status_cooldown": 0.58 if status == "cooldown" else 0.0,
        "truth_status_predicted_blocked": 0.28 if status == "predicted_blocked" else 0.0,
        "lineage_concentration": max(0.0, theme_share - 0.42) * 0.52,
        "price_or_returns_corr_target": 0.13 * x["corr_target_returns"] + 0.10 * x["corr_target_price"],
        "neutralization_depth": 0.05 * max(0.0, x["neutralization_depth"] - 1.0),
        "trade_when_depth": 0.04 * max(0.0, x["trade_when_depth"] - 1.0),
        "lineage_parent_count": 0.04 * max(0.0, x["parent_count"] - 1.0),
        "complexity": 0.06 * x["complexity"],
    }
    risk_discounts = {
        "cross_dataset_parents": -0.11 * x["cross_dataset_parents"],
        "cross_family_parents": -0.07 * x["cross_family_parents"],
        "has_delta": -0.05 * x["has_delta"],
        "operator_entropy": -0.035 * min(3.0, x["operator_entropy"]),
        "field_diversity_high": -0.05 * x["diversity_high"],
        "residual_or_neutralize": -0.06 * x["uses_residual_or_neutralize"],
        "differenced_neutralizer": -0.06 * x.get("has_differenced_neutralizer", 0.0),
        "regime_conditioned": -0.04 * x.get("regime_conditioned", 0.0),
        "truth_status_clear": -0.20 if status == "clear" else 0.0,
        "clean_v22_low_complexity": -0.08 if x.get("is_d1_v22", 0.0) and not x["has_corr"] and x["complexity"] <= 0.45 else 0.0,
        "clean_v23_safe_corr": -0.09 if x.get("is_d1_v23", 0.0) and x["has_corr"] and not x["corr_target_returns"] and not x["corr_target_price"] and x["complexity"] <= 0.65 else 0.0,
    }
    self_corr_risk = max(0.0, sum(risk_components.values()) + sum(risk_discounts.values()))
    submit_score = quality + novelty + 0.8 * family_pass_rate - self_corr_risk - 0.25 * x["length_penalty"]
    label = labels.get(aid)
    empirical_offset = 0.12 if label == 1 else -0.08 if label == 0 and block[aid] == 0 else 0.0
    base_p_self_corr = 1 / (1 + math.exp(-((self_corr_risk + empirical_offset) * 3.6 - 1.6)))
    lineage_prior_discount = LINEAGE_PRIOR_DISCOUNT.get(lineage_theme, 1.0)
    p_self_corr = min(1.0, max(0.0, base_p_self_corr * lineage_prior_discount))
    submitted_meta = score_against_submitted(expression, {}, submitted_features or []) if submitted_features else {
        "max_similarity": 0.0,
        "mean_top5_similarity": 0.0,
        "nearest_submitted_alpha": None,
        "lineage_overlap": 0.0,
        "collision_level": "missing",
        "top_matches": [],
    }
    submitted_similarity = float(submitted_meta.get("max_similarity", 0.0) or 0.0)
    submitted_collision_level = str(submitted_meta.get("collision_level") or "missing")
    if submitted_similarity >= 0.80:
        p_self_corr = max(p_self_corr, 0.90)
        submit_score -= 1.50
    elif submitted_similarity >= 0.65:
        p_self_corr = max(p_self_corr, 0.60)
        submit_score -= 0.75
    elif submitted_similarity >= 0.45:
        p_self_corr = min(1.0, p_self_corr + 0.10)
        submit_score -= 0.30
    p_pass = 1 / (1 + math.exp(-(submit_score - 1.35)))
    quality_pass = sharpe >= PASS_THRESHOLD["sharpe"] and fitness >= PASS_THRESHOLD["fitness"] and turnover <= PASS_THRESHOLD["turnover"]
    d1_ready = int(
        quality_pass
        and p_self_corr < 0.20
    )
    structural_novel = bool(
        x["cross_dataset_parents"]
        or x["cross_family_parents"]
        or x["has_delta"]
        or x["has_decay"]
        or x.get("has_differenced_neutralizer", 0.0)
        or x.get("regime_conditioned", 0.0)
        or lineage_theme == "mixed"
    )
    under_sampled = theme_share <= 0.12 or lineage_theme in {"fundamental_valuation", "liquidity_volatility", "market_size", "unknown"}
    fde_candidate = bool(
        lineage_theme in UNDER_SAMPLED_LINEAGES
        and quality_pass
        and repair_depth <= 1
    )
    exploration_candidate = bool(
        quality_pass
        and status not in {"blocked", "pending", "cooldown", "clear"}
        and (
            under_sampled
            or (p_self_corr >= 0.20 and p_self_corr <= 0.75 and structural_novel)
        )
    )
    if not quality_pass:
        exploration_reason = ""
    elif status in {"blocked", "pending", "cooldown", "clear"}:
        exploration_reason = f"authoritative_status={status}"
    elif under_sampled:
        exploration_reason = f"quality_pass under-sampled lineage={lineage_theme} share={theme_share:.3f}"
    elif structural_novel and p_self_corr >= 0.20:
        exploration_reason = f"quality_pass structurally novel high predicted risk p={p_self_corr:.3f}"
    else:
        exploration_reason = ""
    sorted_risk = sorted(
        {**risk_components, **risk_discounts}.items(),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    return {
        "alpha_id": aid,
        "family": fam,
        "sharpe": sharpe,
        "fitness": fitness,
        "turnover": turnover,
        "pass_quality": int(quality_pass),
        "quality_score": round(quality, 4),
        "submit_score": round(submit_score, 4),
        "p_pass": round(p_pass, 4),
        "self_corr_risk": round(self_corr_risk, 4),
        "p_self_corr_block": round(p_self_corr, 4),
        "base_p_self_corr_block": round(base_p_self_corr, 4),
        "lineage_prior_discount": round(lineage_prior_discount, 4),
        "blocked_count": int(block[aid]),
        "self_corr_label": label,
        "truth_self_corr_status": status,
        "lineage_theme": lineage_theme,
        "submitted_similarity_max": round(submitted_similarity, 4),
        "submitted_similarity_top5_mean": submitted_meta.get("mean_top5_similarity", 0.0),
        "submitted_nearest_alpha_id": submitted_meta.get("nearest_submitted_alpha"),
        "submitted_lineage_overlap": submitted_meta.get("lineage_overlap", 0.0),
        "submitted_collision_level": submitted_collision_level,
        "submitted_similarity_penalty": 1.0 if submitted_similarity >= 0.80 else 0.5 if submitted_similarity >= 0.65 else 0.2 if submitted_similarity >= 0.45 else 0.0,
        "submitted_top_matches": submitted_meta.get("top_matches", [])[:3],
        "lineage_theme_pass_share": round(theme_share, 4),
        "repair_depth": repair_depth,
        "d1_ready": d1_ready,
        "exploration_candidate": exploration_candidate,
        "exploration_reason": exploration_reason,
        "fde_candidate": fde_candidate,
        "features": {k: round(v, 4) for k, v in x.items()},
        "risk_components": {k: round(v, 4) for k, v in {**risk_components, **risk_discounts}.items() if abs(v) >= 0.0001},
        "risk_explanation": [f"{k}={v:.3f}" for k, v in sorted_risk if abs(v) >= 0.03][:6],
    }


def family_stats(conn: sqlite3.Connection, labels: dict[str, int]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    global_rows = conn.execute("select alpha_id, fail_reasons from backtest_results order by datetime(created_at) desc limit 600").fetchall()
    global_self_corr = sum(1 for r in global_rows if labels.get(str(r["alpha_id"])) == 1 or "SELF_CORRELATION" in str(r["fail_reasons"] or ""))
    global_pass_rows = conn.execute(
        """
        select sharpe, fitness, turnover
        from backtest_results
        where sharpe is not null and fitness is not null and turnover is not null
        order by datetime(created_at) desc
        limit 600
        """
    ).fetchall()
    global_pass = sum(1 for r in global_pass_rows if to_float(r["sharpe"]) >= 1.6 and to_float(r["fitness"]) >= 1.0 and to_float(r["turnover"], 99) <= 0.45)
    global_pass_rate = global_pass / max(1, len(global_pass_rows))
    global_self_corr_rate = global_self_corr / max(1, len(global_rows))
    stats["__global__"] = {"tested": len(global_rows), "pass_rate": global_pass_rate, "self_corr_rate": global_self_corr_rate}
    for prefix in PREFIXES:
        rows = conn.execute(
            f"select alpha_id, sharpe, fitness, turnover, fail_reasons from backtest_results where {exact_prefix_clause('alpha_id')} order by datetime(created_at) desc limit 120",
            (exact_prefix_param(prefix),),
        ).fetchall()
        if not rows:
            continue
        tested = len(rows)
        pass_count = sum(1 for r in rows if to_float(r["sharpe"]) >= 1.6 and to_float(r["fitness"]) >= 1.0 and to_float(r["turnover"], 99) <= 0.45)
        self_corr = sum(1 for r in rows if labels.get(str(r["alpha_id"])) == 1 or "SELF_CORRELATION" in str(r["fail_reasons"] or ""))
        stats[prefix] = {
            "tested": tested,
            "pass_rate": beta_mean(pass_count, tested, global_pass_rate, prior_strength=10.0),
            "raw_pass_rate": pass_count / max(1, tested),
            "self_corr_rate": beta_mean(self_corr, tested, global_self_corr_rate, prior_strength=18.0),
            "raw_self_corr_rate": self_corr / max(1, tested),
        }
    return stats


def p_self_corr_histogram(scored: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for row in scored:
        p = float(row.get("p_self_corr_block", 1.0) or 0.0)
        if p < 0.2:
            buckets["0-0.2"] += 1
        elif p < 0.4:
            buckets["0.2-0.4"] += 1
        elif p < 0.6:
            buckets["0.4-0.6"] += 1
        elif p < 0.8:
            buckets["0.6-0.8"] += 1
        else:
            buckets["0.8-1.0"] += 1
    return buckets


def write_report(payload: dict[str, Any], report_path: Path = REPORT_PATH) -> None:
    summary = payload.get("summary") or {}
    lines = [
        f"# ML Candidate Scorer Report - {payload.get('updated_at')}",
        "",
        f"- rows_scored: `{summary.get('rows_scored', 0)}`",
        f"- high_submit_score: `{summary.get('high_submit_score', 0)}`",
        f"- high_self_corr_risk: `{summary.get('high_self_corr_risk', 0)}`",
        f"- d1_ready: `{summary.get('d1_ready', 0)}`",
        f"- exploration_candidates: `{summary.get('exploration_candidates', 0)}`",
        f"- fde_candidates: `{summary.get('fde_candidates', 0)}`",
        f"- avg_operator_entropy: `{summary.get('avg_operator_entropy', 0.0)}`",
        f"- avg_field_diversity_index: `{summary.get('avg_field_diversity_index', 0.0)}`",
        f"- lineage_prior_discount_applications: `{summary.get('lineage_prior_discount_applications', 0)}`",
        "",
        "## p_self_corr_block Histogram",
    ]
    for bucket, count in (summary.get("p_self_corr_block_histogram") or {}).items():
        lines.append(f"- {bucket}: {count}")
    lines.extend(["", "## FDE by Lineage"])
    for lineage, count in (summary.get("fde_by_lineage") or {}).items():
        lines.append(f"- {lineage}: {count}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score alpha candidates for pass/submit feasibility using lightweight feature model")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    block = blocked_counts()
    labels = self_corr_labels()
    truth = load_truth_table()
    truth_index = truth["index"]
    if truth_index:
        truth_block = truth_blocked_counts(truth_index)
        for alpha_id, count in truth_block.items():
            block[alpha_id] = max(block[alpha_id], count)
        labels.update(truth_self_corr_labels(truth_index))
    theme_shares = lineage_theme_shares(truth.get("summary") or {})
    fs = family_stats(conn, labels)
    where, params = exact_prefix_where("alpha_id", PREFIXES)
    rows = conn.execute(
        f"select alpha_id, expression, sharpe, fitness, turnover, fail_reasons, created_at from backtest_results where {where} order by datetime(created_at) desc limit ?",
        params + (args.limit * 4,),
    ).fetchall()
    submitted_features = load_submitted_features()
    scored = [score_row(r, block, labels, fs, truth_index=truth_index, theme_shares=theme_shares, submitted_features=submitted_features) for r in rows]
    scored.sort(key=lambda x: (x["submit_score"], -x["p_self_corr_block"]), reverse=True)
    avg_operator_entropy = round(sum(float(x.get("features", {}).get("operator_entropy", 0.0) or 0.0) for x in scored) / max(1, len(scored)), 4)
    avg_field_diversity = round(sum(float(x.get("features", {}).get("field_diversity_index", 0.0) or 0.0) for x in scored) / max(1, len(scored)), 4)
    fde_by_lineage = {lineage: sum(1 for x in scored if x.get("fde_candidate") and x.get("lineage_theme") == lineage) for lineage in sorted(UNDER_SAMPLED_LINEAGES)}
    payload = {
        "version": 5,
        "updated_at": now_iso(),
        "model": "lightweight_logistic_heuristic_v5_structural_fde",
        "family_stats": fs,
        "truth_table": {
            "enabled": bool(truth_index),
            "rows": len(truth_index),
            "status_counts": (truth.get("summary") or {}).get("status_counts", {}),
            "lineage_theme_pass_shares": {k: round(v, 4) for k, v in theme_shares.items()},
        },
        "top_candidates": scored[: args.limit],
        "self_corr_labels": {
            "blocked": sum(1 for v in labels.values() if v == 1),
            "cleared_authoritative": sum(1 for v in labels.values() if v == 0),
        },
        "summary": {
            "rows_scored": len(scored),
            "families": dict(Counter(x["family"] for x in scored)),
            "high_submit_score": sum(1 for x in scored if x["submit_score"] >= 1.6),
            "high_self_corr_risk": sum(1 for x in scored if x["p_self_corr_block"] >= 0.55),
            "d1_ready": sum(1 for x in scored if x["d1_ready"]),
            "exploration_candidates": sum(1 for x in scored if x.get("exploration_candidate")),
            "fde_candidates": sum(1 for x in scored if x.get("fde_candidate")),
            "fde_by_lineage": fde_by_lineage,
            "avg_operator_entropy": avg_operator_entropy,
            "avg_field_diversity_index": avg_field_diversity,
            "p_self_corr_block_histogram": p_self_corr_histogram(scored),
            "lineage_prior_discount_applications": sum(1 for x in scored if float(x.get("lineage_prior_discount", 1.0) or 1.0) < 1.0),
            "submitted_library_rows": len(submitted_features),
            "submitted_collision_counts": dict(Counter(str(x.get("submitted_collision_level") or "missing") for x in scored)),
            "submitted_high_collision": sum(1 for x in scored if float(x.get("submitted_similarity_max", 0.0) or 0.0) >= 0.80),
            "submitted_medium_collision": sum(1 for x in scored if 0.65 <= float(x.get("submitted_similarity_max", 0.0) or 0.0) < 0.80),
        },
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload)
    print(json.dumps({"updated_at": payload["updated_at"], "summary": payload["summary"], "labels": payload["self_corr_labels"], "top": payload["top_candidates"][:8]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
