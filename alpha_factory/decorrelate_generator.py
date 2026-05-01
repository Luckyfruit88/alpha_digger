from __future__ import annotations

import csv
import hashlib
import itertools
from pathlib import Path
from typing import Iterable

# Decorrelated variants for cases where a seed passes normal checks but is too
# similar to an already-submitted alpha. These change structure/field family,
# not only numeric windows.
FIELDS = [
    "change_in_eps_surprise",
    "earnings_revision_magnitude",
    "earnings_torpedo_indicator",
    "earnings_shortfall_metric",
    "earnings_momentum_composite_score",
    "earnings_momentum_analyst_score",
    "cashflow_per_share_max_guidance_quarterly",
    "actual_sales_value_annual",
    "actual_cashflow_per_share_value_quarterly",
    "actual_eps_value_quarterly",
    "analyst_revision_rank_derivative",
    "eps_estimate_value",
    "eps_previous_estimate_value",
    "earnings_per_share_reported_value",
]
COMPARATORS = [
    "returns",
    "ts_std_dev(returns, 120)",
    "ts_std_dev(returns, 252)",
    "cap",
    "adv20",
    "volume",
]
WINDOWS = [40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 252]
BASE_DECORR = [
    "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 200))",
    "rank(ts_rank(earnings_momentum_analyst_score, 160) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(ts_std_dev(returns, 252), 252))",
    "rank(ts_rank(earnings_momentum_composite_score, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_composite_score, 160) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_composite_score, 120) - ts_rank(ts_std_dev(returns, 252), 252))",
    "rank(ts_rank(earnings_shortfall_metric, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_shortfall_metric, 160) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_shortfall_metric, 120) - ts_rank(cap, 252))",
    "rank(ts_rank(earnings_revision_magnitude, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_torpedo_indicator, 120) - ts_rank(ts_std_dev(returns, 252), 252))",
    "rank(ts_rank(cashflow_per_share_max_guidance_quarterly, 120) - ts_rank(cap, 252))",
]

BEST_CORES = [
    "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 180))",
    "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 160))",
    "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 100))",
]

ALT_DATASET_SEEDS = [
    "rank(ts_zscore(actual_sales_value_annual, 120) - ts_zscore(cap, 120))",
    "rank(ts_zscore(actual_cashflow_per_share_value_quarterly, 120) - ts_zscore(cap, 120))",
    "rank(ts_zscore(actual_eps_value_quarterly, 120) - ts_zscore(returns, 120))",
    "rank(ts_rank(analyst_revision_rank_derivative, 120) - ts_rank(returns, 120))",
    "rank(ts_rank(eps_estimate_value, 120) - ts_rank(eps_previous_estimate_value, 120))",
    "rank(ts_rank(earnings_per_share_reported_value, 120) - ts_rank(eps_estimate_value, 120))",
]

# Farther-distance structural migrations.  The previous decor set was still
# heavily concentrated around rank(ts_rank(fundamental) - ts_rank(returns)).
# These operators change the lineage more aggressively: event shocks, grouped
# neutralization, sparse trade_when gates, price/volume residual helpers, and
# non-return comparator families.  This is intended for persistent
# SELF_CORRELATION failures where window/weight tweaks are too close to the
# submitted parent.
DEEP_MIGRATION_FIELDS = [
    "change_in_eps_surprise",
    "earnings_revision_magnitude",
    "earnings_momentum_analyst_score",
    "earnings_momentum_composite_score",
    "actual_eps_value_quarterly",
    "actual_cashflow_per_share_value_quarterly",
]
SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LOW_VOL_GATE = "ts_rank(ts_std_dev(returns, 10), 252) < 0.8"
LIQUID_GATE = "rank(adv20) > 0.2"
LIQUID_EXIT = "rank(adv20) < 0.05"
OVERNIGHT = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"
INTRADAY = "(close - open) / open"

PASS_CENTERED_SEEDS = [
    "rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + 0.25 * rank(ts_corr(vwap, open, 252)))",
]


def _best_core_decorrelators() -> list[str]:
    variants: list[str] = []
    for core in BEST_CORES:
        # Move away from already-submitted plain rank-minus-return shapes by
        # adding smoothing, gating, and orthogonal residuals. These target the
        # observed failure mode: clean Sharpe but fitness slightly under 1.2 and
        # self-correlation pending/unsafe.
        for win in [3, 5, 10]:
            variants.append(f"rank(ts_decay_linear({core}, {win}))")
            variants.append(f"rank(ts_mean({core}, {win}))")
        variants.append(f"trade_when(rank(adv20) > 0.2, {core}, rank(adv20) < 0.05)")
        variants.append(f"trade_when(rank(cap) > 0.2, {core}, rank(cap) < 0.05)")
        variants.append(f"rank({core} - rank(cap))")
        variants.append(f"rank({core} - rank(adv20))")
        variants.append(f"rank({core} + 0.25 * rank(ts_corr(vwap, open, 252)))")
    return variants


def _medium_distance_migrations() -> list[str]:
    variants: list[str] = []
    # Corrective path after the first deep-migration smoke test: the event-shock
    # family moved far enough to reduce lineage similarity, but several samples
    # lost too much signal.  Prefer medium-distance rewrites that keep the proven
    # economic core while changing neutralization, active universe, and residual
    # helpers.
    cores = [
        "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))",
        "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 180))",
        "rank(ts_rank(earnings_momentum_composite_score, 120) - ts_rank(returns, 252))",
        "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 252))",
    ]
    residuals = [
        f"rank(-ts_corr(abs({OVERNIGHT}), ts_delay(volume / adv20, 1), 20))",
        f"rank(ts_corr(ts_delta(volume / adv20, 1), abs({INTRADAY}), 10))",
        "rank(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2))",
        "rank(ts_percentage(returns, 60, percentage=0.9) - ts_percentage(returns, 60, percentage=0.1))",
    ]
    for core in cores:
        for win in [5, 10, 20]:
            smoothed = f"rank(ts_decay_linear({core}, {win}))"
            variants.append(f"group_neutralize({smoothed}, subindustry)")
            variants.append(f"group_neutralize({smoothed}, {SIZE_BUCKET})")
            variants.append(f"trade_when({LOW_VOL_GATE}, group_neutralize({smoothed}, {SIZE_BUCKET}), -1)")
        variants.append(f"trade_when({LIQUID_GATE}, group_neutralize({core}, {SIZE_BUCKET}), {LIQUID_EXIT})")
        variants.append(f"group_neutralize(rank({core} - rank(ts_std_dev(returns, 30))), {SIZE_BUCKET})")
        variants.append(f"group_neutralize(rank({core} - rank(ts_std_dev(volume / adv20, 30))), {SIZE_BUCKET})")
        for residual in residuals:
            variants.append(f"group_neutralize(rank({core} + {residual}), {SIZE_BUCKET})")
            variants.append(f"trade_when({LIQUID_GATE}, group_neutralize(rank({core} + {residual}), {SIZE_BUCKET}), {LIQUID_EXIT})")
    return variants


def _deep_function_migrations() -> list[str]:
    variants: list[str] = []
    market_ret = "group_mean(returns, 1, market)"
    short_excess = f"ts_mean(returns - {market_ret}, 5)"
    long_excess = f"ts_delay(ts_mean(returns - {market_ret}, 20), 20)"
    pv_residuals = [
        f"rank(-ts_corr(abs({OVERNIGHT}), ts_delay(volume / adv20, 1), 20))",
        f"rank(ts_corr(ts_delta(volume / adv20, 1), abs({INTRADAY}), 10))",
        "rank(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2))",
        "rank(ts_percentage(returns, 60, percentage=0.9) - ts_percentage(returns, 60, percentage=0.1))",
    ]
    for field in DEEP_MIGRATION_FIELDS:
        shock = f"rank(ts_delta(ts_mean({field}, 10), 60))"
        disagreement = f"rank(ts_rank({field}, 60) - ts_rank({field}, 252))"
        zresid = f"rank(ts_zscore({field}, 120) - ts_zscore(cap, 120))"
        volresid = f"rank(ts_rank({field}, 120) - ts_rank(ts_std_dev(returns, 120), 120))"
        variants.extend([
            f"group_neutralize({shock}, subindustry)",
            f"group_neutralize({shock} - rank({short_excess}) - rank({long_excess}), {SIZE_BUCKET})",
            f"trade_when({LOW_VOL_GATE}, group_neutralize({shock}, {SIZE_BUCKET}), -1)",
            f"trade_when({LIQUID_GATE}, group_neutralize({disagreement} * rank(-ts_delta(close, 5)), {SIZE_BUCKET}), {LIQUID_EXIT})",
            f"group_neutralize({disagreement} - rank({short_excess}) - rank({long_excess}), {SIZE_BUCKET})",
            f"group_neutralize({zresid}, {SIZE_BUCKET})",
            f"trade_when({LOW_VOL_GATE}, group_neutralize({volresid}, {SIZE_BUCKET}), -1)",
        ])
        for pv in pv_residuals[:3]:
            variants.append(f"group_neutralize(rank({shock} + {pv}), {SIZE_BUCKET})")
            variants.append(f"trade_when({LIQUID_GATE}, group_neutralize(rank({disagreement} + {pv}), {SIZE_BUCKET}), {LIQUID_EXIT})")
    # A compact set of proven-family rewrites with genuinely different helper
    # operators. These keep the economic signal but should not look like another
    # plain return-rank subtraction to WQ self-correlation checks.
    proven = "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))"
    variants.extend([
        f"trade_when({LOW_VOL_GATE}, group_neutralize({proven}, {SIZE_BUCKET}), -1)",
        f"group_neutralize(rank({proven} - rank(ts_std_dev(volume / adv20, 30))), {SIZE_BUCKET})",
        f"group_neutralize(rank({proven} + rank(-ts_corr(abs({OVERNIGHT}), ts_delay(volume / adv20, 1), 20))), {SIZE_BUCKET})",
        f"trade_when(volume > adv20, group_neutralize(rank({proven} * rank(-ts_delta(close, 5))), {SIZE_BUCKET}), -1)",
    ])
    return variants


def _pass_centered_variants() -> list[str]:
    variants: list[str] = []
    for seed in PASS_CENTERED_SEEDS:
        # Explore nearby weights/windows around the first passing structure.
        for weight in [0.15, 0.2, 0.3, 0.35]:
            variants.append(
                f"rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + {weight} * rank(ts_corr(vwap, open, 252)))"
            )
        for price_b, price_a, win in [
            ("vwap", "close", 252),
            ("vwap", "open", 120),
            ("close", "open", 252),
            ("vwap", "open", 160),
        ]:
            variants.append(
                f"rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + 0.25 * rank(ts_corr({price_b}, {price_a}, {win})))"
            )

        # Slight smoothing/gating around the passed form to probe whether we can
        # keep fitness while changing the self-correlation profile.
        variants.append(f"rank(ts_decay_linear({seed}, 3))")
        variants.append(f"rank(ts_mean({seed}, 3))")
        variants.append(f"trade_when(rank(adv20) > 0.2, {seed}, rank(adv20) < 0.05)")
        variants.append(f"trade_when(rank(cap) > 0.2, {seed}, rank(cap) < 0.05)")
        variants.append(f"rank({seed} - rank(cap))")
        variants.append(f"rank({seed} - rank(adv20))")

        # Second-order decorrelation: change the helper signal family itself,
        # not just its weight/window, to reduce self-correlation against the
        # already-good corr(vwap,open/close,*) branch.
        variants.append(
            "rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + 0.2 * rank(ts_corr(vwap, volume, 120)))"
        )
        variants.append(
            "rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + 0.2 * rank(ts_corr(close, volume, 120)))"
        )
        variants.append(
            "rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) - 0.2 * rank(ts_corr(returns, volume, 120)))"
        )
        variants.append(
            "rank(rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252)) + 0.2 * rank(ts_corr(vwap, adv20, 120)))"
        )

        # Field swap while preserving the passing scaffold.
        variants.append(
            "rank(rank(ts_rank(earnings_momentum_composite_score, 120) - ts_rank(returns, 252)) + 0.2 * rank(ts_corr(vwap, open, 252)))"
        )
        variants.append(
            "rank(rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 252)) + 0.2 * rank(ts_corr(vwap, open, 252)))"
        )
    return variants


def _alt_dataset_variants() -> list[str]:
    variants: list[str] = []
    for seed in ALT_DATASET_SEEDS:
        variants.append(seed)
        variants.append(f"rank(ts_decay_linear({seed}, 3))")
        variants.append(f"rank(ts_mean({seed}, 3))")
        variants.append(f"trade_when(rank(adv20) > 0.15, {seed}, rank(adv20) < 0.05)")
        variants.append(f"rank({seed} - rank(ts_std_dev(returns, 120)))")
        variants.append(f"rank({seed} - rank(cap))")
        variants.append(f"rank({seed} + 0.2 * rank(ts_corr(vwap, volume, 120)))")
    return variants


def _decorrelation_grid() -> list[str]:
    variants: list[str] = []
    preferred_fields = [
        "earnings_momentum_analyst_score",
        "earnings_momentum_composite_score",
        "earnings_shortfall_metric",
        "earnings_revision_magnitude",
        "earnings_torpedo_indicator",
    ]
    preferred_comparators = [
        "returns",
        "ts_std_dev(returns, 252)",
        "cap",
        "adv20",
    ]
    preferred_windows = [80, 100, 120, 140, 160, 180, 200, 252]
    for field, fw, comp, cw in itertools.product(preferred_fields, preferred_windows, preferred_comparators, preferred_windows):
        variants.append(f"rank(ts_rank({field}, {fw}) - ts_rank({comp}, {cw}))")
    for field, fw, comp, cw in itertools.product(preferred_fields[:3], [80, 120, 160, 200, 252], ["returns", "volume", "adv20"], [80, 120, 160, 200, 252]):
        win = min(fw, cw)
        variants.append(f"rank(ts_corr(ts_rank({field}, {fw}), ts_rank({comp}, {cw}), {win}))")
        variants.append(f"-rank(ts_corr(ts_rank({field}, {fw}), ts_rank({comp}, {cw}), {win}))")
    return variants


def generate_decorrelated_expressions(count: int, seeds: Iterable[str] | None = None) -> list[str]:
    seen: set[str] = set()
    expressions: list[str] = []
    # After the first deep-migration batch came back weak, promote
    # medium-distance migrations ahead of event-shock/deep rewrites.  This keeps
    # the self-correlation fix but avoids throwing away too much predictive
    # signal in small cron batches.
    for expr in [*_medium_distance_migrations(), *_pass_centered_variants(), *_best_core_decorrelators(), *_alt_dataset_variants(), *_deep_function_migrations(), *BASE_DECORR, *_decorrelation_grid()]:
        if expr not in seen:
            seen.add(expr)
            expressions.append(expr)
    return expressions[:count]


def append_unique_to_csv(csv_path: str | Path, expressions: Iterable[str], defaults: dict, prefix: str = "decor", max_add: int | None = None) -> int:
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
            if not expr or expr in existing_exprs:
                continue
            alpha_id = f"{prefix}_{hashlib.sha1(expr.encode('utf-8')).hexdigest()[:10]}"
            if alpha_id in existing_ids:
                continue
            writer.writerow({
                "id": alpha_id,
                "expression": expr,
                "region": defaults.get("region", "USA"),
                "universe": defaults.get("universe", "TOP3000"),
                "delay": defaults.get("delay", 1),
                "decay": 6,
                "neutralization": defaults.get("neutralization", "INDUSTRY"),
                "truncation": 0.04,
            })
            existing_exprs.add(expr)
            existing_ids.add(alpha_id)
            added += 1
            if max_add is not None and added >= max_add:
                break
    return added
