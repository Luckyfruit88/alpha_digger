#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIBRARY_PATH = ROOT / "data" / "submitted_alpha_library.jsonl"
FEATURES_PATH = ROOT / "state" / "submitted_alpha_features.json"
REPORT_PATH = ROOT / "reports" / "submitted_alpha_library.md"

SIMILARITY_WEIGHTS = {
    "field_jaccard": 0.20,
    "operator_jaccard": 0.15,
    "window_overlap": 0.10,
    "lineage_overlap": 0.35,
    "skeleton_sequence_similarity": 0.20,
    "settings_similarity": 0.00,
}

OPERATOR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])\b\d+(?:\.\d+)?\b")
WINDOW_RE = re.compile(r",\s*(\d{1,4})(?=\s*[,\)])")
KNOWN_OPERATORS = {
    "abs", "add", "and", "bucket", "corr", "decay", "delta", "divide", "group_mean", "group_neutralize",
    "if_else", "log", "max", "min", "multiply", "rank", "regression_neut", "residual", "scale", "sign",
    "signed_power", "subtract", "trade_when", "ts_arg_max", "ts_arg_min", "ts_corr", "ts_covariance",
    "ts_decay_linear", "ts_delta", "ts_max", "ts_mean", "ts_median", "ts_min", "ts_rank", "ts_std_dev", "ts_sum", "ts_zscore", "vec_avg",
    "winsorize", "zscore",
}
IGNORED_TOKENS = KNOWN_OPERATORS | {"true", "false", "nan", "inf", "industry", "subindustry", "sector", "market", "densify"}
LINEAGE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("analyst_momentum", ("anl", "analyst", "earnings", "eps", "est", "recommend", "rating")),
    ("valuation", ("value", "valuation", "book", "tbv", "bve", "pe", "pb", "fcf", "cashflow", "sales", "revenue")),
    ("price_return", ("returns", "return", "close", "open", "vwap", "high", "low", "price")),
    ("volume_liquidity", ("volume", "adv", "turnover", "liquidity", "dollar", "shares")),
    ("size", ("cap", "mkt", "market_cap", "size")),
    ("volatility", ("vol", "std", "variance", "beta", "risk")),
    ("fundamental_quality", ("quality", "margin", "profit", "roe", "roa", "debt", "asset", "liab", "income")),
    ("composite_factor", ("composite", "factor", "score", "alpha")),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def normalize_expression(expr: str) -> str:
    s = (expr or "").lower()
    s = re.sub(r"\s+", "", s)
    s = NUMBER_RE.sub("#", s)
    return s


def skeleton_expression(expr: str) -> str:
    s = normalize_expression(expr)
    tokens = set(extract_fields(expr))
    for tok in sorted(tokens, key=len, reverse=True):
        s = re.sub(rf"\b{re.escape(tok.lower())}\b", "FIELD", s)
    return s


def extract_operators(expr: str) -> list[str]:
    return [m.group(1).lower() for m in OPERATOR_RE.finditer(expr or "")]


def extract_fields(expr: str) -> list[str]:
    ops = set(extract_operators(expr))
    fields: list[str] = []
    for tok in TOKEN_RE.findall(expr or ""):
        low = tok.lower()
        if low in IGNORED_TOKENS or low in ops:
            continue
        if len(low) <= 1:
            continue
        fields.append(low)
    return sorted(set(fields))


def extract_windows(expr: str) -> list[int]:
    vals = []
    for raw in WINDOW_RE.findall(expr or ""):
        try:
            n = int(raw)
        except Exception:
            continue
        if 1 <= n <= 2000:
            vals.append(n)
    return sorted(set(vals))


def lineage_tags(fields: list[str], operators: list[str]) -> list[str]:
    joined = " ".join(fields).lower()
    tags = {name for name, needles in LINEAGE_RULES if any(n in joined for n in needles)}
    if "ts_corr" in operators or "corr" in operators:
        tags.add("correlation")
    if not tags:
        tags.add("unknown")
    return sorted(tags)


def expression_features(expr: str) -> dict[str, Any]:
    operators = extract_operators(expr)
    fields = extract_fields(expr)
    windows = extract_windows(expr)
    norm = normalize_expression(expr)
    skel = skeleton_expression(expr)
    return {
        "fields": fields,
        "operators": sorted(set(operators)),
        "operator_counts": dict(Counter(operators)),
        "windows": windows,
        "lineages": lineage_tags(fields, operators),
        "raw_hash": stable_hash(expr or ""),
        "normalized_hash": stable_hash(norm),
        "field_set_hash": stable_hash("|".join(fields)),
        "operator_skeleton_hash": stable_hash(skel),
        "normalized_expression": norm,
        "skeleton": skel,
        "has_corr": int("ts_corr" in operators or "corr" in operators),
        "has_group_neutralize": int("group_neutralize" in operators),
        "has_trade_when": int("trade_when" in operators),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def build_features(input_path: Path = LIBRARY_PATH, output_path: Path = FEATURES_PATH, report_path: Path = REPORT_PATH) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    items = []
    field_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    lineage_counts: Counter[str] = Counter()
    for row in rows:
        expr = str(row.get("expression") or "")
        feats = expression_features(expr)
        field_counts.update(feats["fields"])
        op_counts.update(feats["operators"])
        lineage_counts.update(feats["lineages"])
        items.append({
            "alpha_id": row.get("alpha_id"),
            "status": row.get("status"),
            "stage": row.get("stage"),
            "date_submitted": row.get("date_submitted"),
            "settings": row.get("settings") or {},
            "metrics": row.get("metrics") or {},
            "features": feats,
        })
    payload = {
        "version": 1,
        "updated_at": now_iso(),
        "source": str(input_path),
        "count": len(items),
        "items": items,
        "summary": {
            "top_fields": field_counts.most_common(30),
            "top_operators": op_counts.most_common(30),
            "lineages": dict(lineage_counts),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Submitted Alpha Library", "", f"- updated_at: `{payload['updated_at']}`", f"- count: `{len(items)}`", "", "## Lineages"]
    for k, v in lineage_counts.most_common():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Top Fields"] + [f"- {k}: {v}" for k, v in field_counts.most_common(20)]
    lines += ["", "## Top Operators"] + [f"- {k}: {v}" for k, v in op_counts.most_common(20)]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def settings_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    keys = {"region", "universe", "delay", "neutralization", "pasteurization", "language"}
    present = [k for k in keys if k in a or k in b]
    if not present:
        return 0.0
    return sum(1 for k in present if str(a.get(k)).lower() == str(b.get(k)).lower()) / len(present)


def sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a[:4000], b[:4000]).ratio()


def load_submitted_features(path: Path = FEATURES_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = payload.get("items") or []
    return [x for x in items if isinstance(x, dict)]


def score_against_submitted(expression: str, settings: dict[str, Any] | None, submitted_features: list[dict[str, Any]]) -> dict[str, Any]:
    cand = expression_features(expression or "")
    settings = settings or {}
    scored = []
    for item in submitted_features:
        feats = item.get("features") or {}
        fj = jaccard(set(cand.get("fields", [])), set(feats.get("fields", [])))
        oj = jaccard(set(cand.get("operators", [])), set(feats.get("operators", [])))
        wj = jaccard(set(cand.get("windows", [])), set(feats.get("windows", [])))
        lj = jaccard(set(cand.get("lineages", [])), set(feats.get("lineages", [])))
        ss = settings_similarity(settings, item.get("settings") or {})
        ns = sequence_similarity(cand.get("skeleton", ""), feats.get("skeleton", ""))
        sim = (
            SIMILARITY_WEIGHTS["field_jaccard"] * fj
            + SIMILARITY_WEIGHTS["operator_jaccard"] * oj
            + SIMILARITY_WEIGHTS["window_overlap"] * wj
            + SIMILARITY_WEIGHTS["lineage_overlap"] * lj
            + SIMILARITY_WEIGHTS["skeleton_sequence_similarity"] * ns
            + SIMILARITY_WEIGHTS["settings_similarity"] * ss
        )
        scored.append({"alpha_id": item.get("alpha_id"), "similarity": round(sim, 4), "field_jaccard": fj, "operator_jaccard": oj, "window_overlap": wj, "lineage_overlap": lj, "skeleton_sequence_similarity": ns, "settings_similarity": ss, "lineages": feats.get("lineages", [])})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    top = scored[:5]
    max_sim = float(top[0]["similarity"]) if top else 0.0
    level = "high" if max_sim >= 0.80 else "medium" if max_sim >= 0.65 else "weak" if max_sim >= 0.45 else "low"
    nearest = top[0] if top else {}
    return {
        "max_similarity": round(max_sim, 4),
        "mean_top5_similarity": round(sum(float(x["similarity"]) for x in top) / max(1, len(top)), 4),
        "nearest_submitted_alpha": nearest.get("alpha_id"),
        "field_jaccard": round(float(nearest.get("field_jaccard", 0.0) or 0.0), 4),
        "operator_jaccard": round(float(nearest.get("operator_jaccard", 0.0) or 0.0), 4),
        "window_overlap": round(float(nearest.get("window_overlap", 0.0) or 0.0), 4),
        "lineage_overlap": round(float(nearest.get("lineage_overlap", 0.0) or 0.0), 4),
        "skeleton_sequence_similarity": round(float(nearest.get("skeleton_sequence_similarity", 0.0) or 0.0), 4),
        "settings_similarity": round(float(nearest.get("settings_similarity", 0.0) or 0.0), 4),
        "weights": SIMILARITY_WEIGHTS,
        "collision_level": level,
        "candidate_lineages": cand.get("lineages", []),
        "top_matches": [{"alpha_id": x.get("alpha_id"), "similarity": x.get("similarity"), "lineages": x.get("lineages", [])} for x in top],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build submitted alpha structural features")
    parser.add_argument("--input", type=Path, default=LIBRARY_PATH)
    parser.add_argument("--output", type=Path, default=FEATURES_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    payload = build_features(args.input, args.output, args.report)
    print(json.dumps({"updated_at": payload["updated_at"], "count": payload["count"], "output": str(args.output), "report": str(args.report)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
