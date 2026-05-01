from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Iterable

# Curated generator based on locally scored WorldQuant fields.
# Emphasis: low-crowding, high-coverage analyst/model fields with explicit
# earnings surprise/revision/guidance semantics.
BASE_FIELDS = [
    "open", "high", "low", "close", "volume", "vwap", "returns",
    "adv20", "cap", "sharesout",
]

MODEL_SIGNAL_FIELDS = [
    "earnings_torpedo_indicator",
    "earnings_revision_magnitude",
    "coefficient_variation_fy2_eps",
    "equity_value_score",
    "change_in_eps_surprise",
    "earnings_momentum_composite_score",
    "earnings_momentum_composite_score_2",
    "earnings_momentum_analyst_score",
    "enterprise_value_weighted_value_score",
    "fifty_to_two_hundred_day_price_ratio",
    "earnings_shortfall_metric",
]

EPS_GUIDANCE_FIELDS = [
    "eps_reported_min_guidance_qtr",
    "eps_adjusted_min_guidance_value",
    "eps_adjusted_min_guidance_qtr",
    "earnings_per_share_min_guidance",
    "earnings_per_share_max_guidance",
    "eps_min_guidance_quarterly",
    "eps_max_guidance_quarterly",
    "earnings_per_share_median_value",
    "eps_previous_estimate_value",
    "eps_estimate_value",
    "earnings_per_share_reported",
    "earnings_per_share_reported_value",
    "earnings_per_share_minimum",
    "earnings_per_share_maximum",
    "anl4_qfv4_eps_low",
    "anl4_qfv4_median_eps",
    "anl4_qfv4_eps_mean",
    "anl4_qfv4_eps_high",
]

CASHFLOW_GUIDANCE_FIELDS = [
    "cashflow_per_share_max_guidance",
    "cashflow_per_share_min_guidance",
    "cashflow_per_share_max_guidance_quarterly",
    "cashflow_per_share_min_guidance_quarterly",
    "cash_flow_operations_min_guidance",
    "cash_flow_financing_max_guidance",
]

DIVIDEND_VALUE_FIELDS = [
    "anl4_afv4_div_median",
    "anl4_afv4_div_high",
    "anl4_afv4_div_low",
    "dividend_min_guidance_quarterly",
    "dividend_min_guidance_value",
]

QUALITY_VALUE_FIELDS = [
    "ebit_reported_value",
    "earnings_per_share_nongaap_value",
    "book_value_per_share_min_guidance_qtr",
    "capital_expenditure_max_guidance_qtr",
]

ALL_SIGNAL_FIELDS = MODEL_SIGNAL_FIELDS + EPS_GUIDANCE_FIELDS + CASHFLOW_GUIDANCE_FIELDS + DIVIDEND_VALUE_FIELDS + QUALITY_VALUE_FIELDS

MID_WINDOWS = [20, 40, 60]
SLOW_WINDOWS = [120, 252]

CURATED_TEMPLATES = [
    # Direct model signals: these already encode revisions/surprises; keep transforms simple.
    "rank(ts_mean({model}, {mid}))",
    "rank(ts_delta({model}, {mid}))",
    "rank(ts_rank({model}, {slow}) - ts_rank(returns, {mid}))",
    "-rank(ts_corr(ts_rank({model}, {mid}), ts_rank(volume, {mid}), {mid}))",

    # EPS guidance/revision spread style signals.
    "rank(ts_delta({eps}, {mid}))",
    "rank(ts_mean({eps}, {mid}) / (ts_std_dev({eps}, {slow}) + 0.001))",
    "rank(ts_rank({eps_hi}, {slow}) - ts_rank({eps_lo}, {slow}))",
    "rank(ts_rank({eps}, {slow}) - ts_rank(cap, {slow}))",
    "rank(ts_rank({eps}, {slow}) - ts_rank(adv20, {slow}))",

    # Cashflow/dividend/quality as valuation and stability proxies.
    "rank(ts_delta({cashflow}, {mid}))",
    "rank(ts_rank({cashflow}, {slow}) - ts_rank(cap, {slow}))",
    "rank(ts_rank({dividend}, {slow}) - ts_rank(cap, {slow}))",
    "rank(ts_rank({quality}, {slow}) - ts_rank(cap, {slow}))",

    # Cross-confirmation with market behavior; medium windows only.
    "rank(ts_corr(ts_rank({signal}, {mid}), ts_rank(returns, {mid}), {mid}))",
    "-rank(ts_corr(ts_rank({signal}, {mid}), ts_rank(volume, {mid}), {mid}))",
    "rank(ts_rank({signal}, {slow}) - ts_rank(ts_std_dev(returns, {slow}), {slow}))",
]


def _format_template(template: str, rng: random.Random) -> str:
    eps_a, eps_b = rng.sample(EPS_GUIDANCE_FIELDS, 2)
    return template.format(
        mid=rng.choice(MID_WINDOWS),
        slow=rng.choice(SLOW_WINDOWS),
        model=rng.choice(MODEL_SIGNAL_FIELDS),
        eps=rng.choice(EPS_GUIDANCE_FIELDS),
        eps_hi=eps_a,
        eps_lo=eps_b,
        cashflow=rng.choice(CASHFLOW_GUIDANCE_FIELDS),
        dividend=rng.choice(DIVIDEND_VALUE_FIELDS),
        quality=rng.choice(QUALITY_VALUE_FIELDS),
        signal=rng.choice(ALL_SIGNAL_FIELDS),
    )


def _passes_static_filters(expr: str) -> bool:
    blocked_fragments = [
        "rank(open)", "rank(high)", "rank(low)", "rank(close)", "rank(volume)",
        "close / low", "high / close", "open / close", "close / volume",
        "ts_delta(returns, 3)", "ts_delta(returns, 5)",
        "ts_rank(low, 3)", "ts_rank(volume, 3)",
        "ts_corr(sharesout", "ts_covariance(",
    ]
    return not any(fragment in expr for fragment in blocked_fragments)


def generate_expressions(count: int, seed: int | None = None, fields: Iterable[str] | None = None) -> list[str]:
    rng = random.Random(seed)
    expressions: set[str] = set()
    attempts = 0
    max_attempts = max(count * 100, 100)
    while len(expressions) < count and attempts < max_attempts:
        attempts += 1
        expr = _format_template(rng.choice(CURATED_TEMPLATES), rng)
        if _passes_static_filters(expr):
            expressions.add(expr)
    return sorted(expressions)


def append_unique_to_csv(csv_path: str | Path, expressions: Iterable[str], defaults: dict, prefix: str = "curated") -> int:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_exprs: set[str] = set()
    existing_ids: set[str] = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("expression"):
                    existing_exprs.add(row["expression"].strip())
                if row.get("id"):
                    existing_ids.add(row["id"].strip())

    is_new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "expression", "region", "universe", "delay", "decay", "neutralization", "truncation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        added = 0
        for expr in expressions:
            expr = expr.strip()
            if not expr or expr in existing_exprs or not _passes_static_filters(expr):
                continue
            digest = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:10]
            alpha_id = f"{prefix}_{digest}"
            if alpha_id in existing_ids:
                continue
            writer.writerow({
                "id": alpha_id,
                "expression": expr,
                "region": defaults.get("region", "USA"),
                "universe": defaults.get("universe", "TOP3000"),
                "delay": defaults.get("delay", 1),
                "decay": defaults.get("decay", 4),
                "neutralization": defaults.get("neutralization", "INDUSTRY"),
                "truncation": defaults.get("truncation", 0.05),
            })
            existing_exprs.add(expr)
            existing_ids.add(alpha_id)
            added += 1
    return added
