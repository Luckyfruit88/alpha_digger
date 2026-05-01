#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.submitted_similarity import FEATURES_PATH, load_submitted_features

STATE_PATH = ROOT / "state" / "lineage_occupancy.json"
REPORT_PATH = ROOT / "reports" / "lineage_occupancy.md"

TARGET_LINEAGES = [
    "fundamental_quality",
    "accrual_quality",
    "cashflow_stability",
    "balance_sheet_stability",
    "fundamental_valuation",
    "liquidity_volatility",
    "market_size",
    "analyst_momentum",
    "price_return",
    "correlation",
    "volume_liquidity",
    "size",
    "volatility",
    "valuation",
    "composite_factor",
]

LINEAGE_KEYWORDS = {
    "fundamental_quality": ["quality", "margin", "profit", "roe", "roa", "debt", "asset", "liab", "income"],
    "accrual_quality": ["accrual", "working_capital", "wc", "receivable", "inventory", "depreciation"],
    "cashflow_stability": ["cashflow", "cash_flow", "fcf", "operating_cash", "cfo", "free_cash", "capex"],
    "balance_sheet_stability": ["asset", "liab", "debt", "equity", "book", "bve", "tbv", "leverage"],
    "fundamental_valuation": ["value", "valuation", "book", "tbv", "bve", "pe", "pb", "sales", "revenue"],
    "liquidity_volatility": ["liquidity", "turnover", "vol", "std", "variance", "risk"],
    "market_size": ["cap", "mkt", "market_cap", "size"],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def lineage_counts(submitted_features: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in submitted_features:
        feats = item.get("features") or {}
        for lineage in feats.get("lineages") or ["unknown"]:
            counts[str(lineage)] += 1
    return counts


def softmax_inverse_counts(counts: Counter[str], targets: list[str] | None = None) -> dict[str, float]:
    names = list(dict.fromkeys(targets or TARGET_LINEAGES))
    raw = {name: 1.0 / (float(counts.get(name, 0)) + 1.0) for name in names}
    # Normalize the inverse-count scores as a softmax so absent/rare lineages
    # get a smooth but bounded preference rather than a hard rule.
    exps = {name: math.exp(score) for name, score in raw.items()}
    denom = sum(exps.values()) or 1.0
    return {name: round(value / denom, 6) for name, value in exps.items()}


def build_lineage_hints(submitted_features: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    submitted_features = submitted_features if submitted_features is not None else load_submitted_features()
    counts = lineage_counts(submitted_features)
    weights = softmax_inverse_counts(counts)
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "version": 1,
        "updated_at": now_iso(),
        "source": str(FEATURES_PATH),
        "submitted_library_rows": len(submitted_features or []),
        "counts": dict(counts),
        "explore_weight": weights,
        "ranked_lineages": [{"lineage": k, "weight": v, "submitted_count": int(counts.get(k, 0))} for k, v in ranked],
        "field_keyword_hints": LINEAGE_KEYWORDS,
    }


def save_outputs(payload: dict[str, Any], state_path: Path = STATE_PATH, report_path: Path = REPORT_PATH) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Lineage Occupancy",
        "",
        f"- updated_at: `{payload.get('updated_at')}`",
        f"- submitted_library_rows: `{payload.get('submitted_library_rows')}`",
        "",
        "## Explore Weights",
    ]
    for item in payload.get("ranked_lineages", []):
        lines.append(f"- {item['lineage']}: weight={item['weight']} submitted_count={item['submitted_count']}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_lineage_hints(path: Path = STATE_PATH) -> dict[str, Any]:
    if not path.exists():
        return build_lineage_hints()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else build_lineage_hints()
    except Exception:
        return build_lineage_hints()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lineage occupancy hints from submitted alpha features")
    parser.add_argument("--output", type=Path, default=STATE_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    payload = build_lineage_hints()
    save_outputs(payload, args.output, args.report)
    print(json.dumps({"updated_at": payload["updated_at"], "submitted_library_rows": payload["submitted_library_rows"], "top": payload["ranked_lineages"][:8], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
