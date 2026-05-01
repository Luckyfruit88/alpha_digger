from __future__ import annotations

import csv
import hashlib
import itertools
import re
from pathlib import Path
from typing import Iterable

WINDOW_PAT = re.compile(r"\b(20|40|60|80|100|120|140|160|180|200|220|252)\b")
BASE_FIELDS = [
    "change_in_eps_surprise",
    "earnings_revision_magnitude",
    "earnings_torpedo_indicator",
    "earnings_shortfall_metric",
    "earnings_momentum_composite_score",
    "earnings_momentum_analyst_score",
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
TURNOVER_SMOOTHERS = [3, 5, 10]
BASE_SEEDS = [
    "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 120))",
    "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 80))",
    "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 40))",
    "rank(ts_rank(change_in_eps_surprise, 252) - ts_rank(returns, 60))",
    "rank(ts_rank(earnings_torpedo_indicator, 252) - ts_rank(returns, 40))",
    "ts_corr(vwap, open, 252)",
]

HIGH_FITNESS_CORES = [
    # Current best family: high Sharpe, clean checks, but fitness is capped by
    # modest returns.  Prefer structural changes that lower churn or gate noisy
    # names instead of only mutating windows.
    "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))",
    "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 180))",
    "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 160))",
    "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 252))",
]

# Fitness-first refits deliberately change active universe, neutralization shape,
# and helper residuals.  The goal is not to find a slightly higher Sharpe near
# the same parent, but to turn strong-but-cutoff-stuck ideas into smoother,
# lower-correlation candidates with a better chance of clearing fitness/cutoff.
SIZE_BUCKET = "bucket(rank(cap), range='0.1,1,0.1')"
LOW_VOL_GATE = "ts_rank(ts_std_dev(returns, 10), 252) < 0.8"
LIQUID_GATE = "rank(adv20) > 0.2"
LIQUID_EXIT = "rank(adv20) < 0.05"
OVERNIGHT = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"
INTRADAY = "(close - open) / open"

WINDOW_VARIANTS = {
    20: [40, 60, 80],
    40: [60, 80, 100, 120],
    60: [80, 100, 120, 160],
    80: [60, 100, 120, 140, 160],
    100: [80, 120, 140, 160],
    120: [80, 100, 140, 160, 180, 200, 252],
    140: [100, 120, 160, 180, 220],
    160: [120, 140, 180, 200, 220, 252],
    180: [120, 140, 160, 200, 220, 252],
    200: [140, 160, 180, 220, 252],
    220: [160, 180, 200, 252],
    252: [120, 160, 180, 200, 220],
}


def _window_mutations(expr: str) -> list[str]:
    matches = list(WINDOW_PAT.finditer(expr))
    if not matches:
        return []
    variants: set[str] = set()
    for m in matches:
        value = int(m.group(0))
        for repl in WINDOW_VARIANTS.get(value, []):
            start, end = m.span()
            variants.add(expr[:start] + str(repl) + expr[end:])
    return sorted(variants)


def _structural_grid() -> list[str]:
    variants: list[str] = []
    for field, fw, comp, cw in itertools.product(BASE_FIELDS, WINDOWS, COMPARATORS, WINDOWS):
        if field == "change_in_eps_surprise" and comp == "returns" and fw in (120, 160, 180, 252) and cw in (120, 160, 180, 252):
            # The crowded core family is already saturated; leave it to seeds.
            continue
        variants.append(f"rank(ts_rank({field}, {fw}) - ts_rank({comp}, {cw}))")
    for field, fw, comp, cw in itertools.product(BASE_FIELDS[:4], [80, 120, 160, 200, 252], ["returns", "volume", "adv20"], [80, 120, 160, 200, 252]):
        variants.append(f"-rank(ts_corr(ts_rank({field}, {fw}), ts_rank({comp}, {cw}), {min(fw, cw)}))")
    return variants


def _fitness_variants() -> list[str]:
    variants: list[str] = []
    for core in HIGH_FITNESS_CORES:
        # Smoothing a daily cross-sectional rank often preserves direction but
        # lowers turnover, which can improve WorldQuant fitness when Sharpe is
        # already strong and turnover is the quiet drag.
        for win in TURNOVER_SMOOTHERS:
            variants.append(f"rank(ts_mean({core}, {win}))")
            variants.append(f"rank(ts_decay_linear({core}, {win}))")

        # Liquidity/size gating changes the active universe and is less
        # self-correlated than another plain window tweak. Keep thresholds broad
        # to avoid collapsing coverage.
        variants.append(f"trade_when(rank(adv20) > 0.2, {core}, rank(adv20) < 0.05)")
        variants.append(f"trade_when(rank(cap) > 0.2, {core}, rank(cap) < 0.05)")

        # Orthogonalize against size/liquidity tilts while keeping the signal
        # family recognizable. These are conservative FastExpr operators used
        # commonly on Brain.
        variants.append(f"rank({core} - rank(cap))")
        variants.append(f"rank({core} - rank(adv20))")
        variants.append(f"rank({core} + 0.25 * rank(ts_corr(vwap, open, 252)))")
    return variants


def _fitness_first_refits() -> list[str]:
    variants: list[str] = []
    market_ret = "group_mean(returns, 1, market)"
    short_excess = f"ts_mean(returns - {market_ret}, 5)"
    long_excess = f"ts_delay(ts_mean(returns - {market_ret}, 20), 20)"
    pv_stability = [
        f"rank(-ts_corr(abs({OVERNIGHT}), ts_delay(volume / adv20, 1), 20))",
        f"rank(ts_corr(ts_delta(volume / adv20, 1), abs({INTRADAY}), 10))",
        "rank(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2))",
    ]
    for core in HIGH_FITNESS_CORES:
        # Corrective ordering: start with medium-distance rewrites that preserve
        # the proven core and alter portfolio construction.  Earlier deep/event
        # rewrites were diverse but weak, so refit batches should spend scarce
        # slots on fitness repair with less signal destruction.
        variants.append(f"trade_when({LIQUID_GATE}, group_neutralize({core}, {SIZE_BUCKET}), {LIQUID_EXIT})")
        variants.append(f"group_neutralize(rank({core} - rank({short_excess}) - rank({long_excess})), {SIZE_BUCKET})")
        variants.append(f"group_neutralize(rank({core} - rank(ts_std_dev(returns, 30))), {SIZE_BUCKET})")
        variants.append(f"group_neutralize(rank({core} - rank(ts_std_dev(volume / adv20, 30))), {SIZE_BUCKET})")
        # Smooth and gate second: this explicitly targets fitness/cutoff instead
        # of raw Sharpe.  Use grouped neutralization to move away from crowded
        # parent lineage and reduce broad factor leakage.
        for win in [5, 10, 20]:
            smooth = f"rank(ts_decay_linear({core}, {win}))"
            variants.append(f"group_neutralize({smooth}, subindustry)")
            variants.append(f"group_neutralize({smooth}, {SIZE_BUCKET})")
            variants.append(f"trade_when({LOW_VOL_GATE}, group_neutralize({smooth}, {SIZE_BUCKET}), -1)")
        variants.append(f"trade_when(rank({core}) > 0.75, group_neutralize({core}, {SIZE_BUCKET}), -1)")
        variants.append(f"trade_when((rank({core}) > 0.7) * ({LOW_VOL_GATE}), group_neutralize({core}, subindustry), -1)")
        for pv in pv_stability:
            variants.append(f"group_neutralize(rank({core} + {pv}), {SIZE_BUCKET})")
            variants.append(f"trade_when({LIQUID_GATE}, group_neutralize(rank({core} + {pv}), {SIZE_BUCKET}), {LIQUID_EXIT})")
    return variants


def _alt_function_dataset_grid() -> list[str]:
    fields = [
        "actual_sales_value_annual",
        "actual_cashflow_per_share_value_quarterly",
        "actual_eps_value_quarterly",
        "analyst_revision_rank_derivative",
        "eps_estimate_value",
        "earnings_per_share_reported_value",
    ]
    variants: list[str] = []
    for field in fields:
        variants.append(f"rank(ts_zscore({field}, 120) - ts_zscore(cap, 120))")
        variants.append(f"rank(ts_zscore({field}, 120) - ts_zscore(returns, 120))")
        variants.append(f"rank(ts_delta(ts_mean({field}, 20), 60))")
        variants.append(f"rank(ts_rank({field}, 120) - ts_rank(ts_std_dev(returns, 120), 120))")
        variants.append(f"-rank(ts_corr(ts_rank({field}, 120), ts_rank(volume, 120), 120))")
    variants.append("rank(ts_rank(eps_estimate_value, 120) - ts_rank(eps_previous_estimate_value, 120))")
    variants.append("rank(ts_rank(earnings_per_share_reported_value, 120) - ts_rank(eps_estimate_value, 120))")
    return variants


def generate_refit_expressions(count: int, seeds: Iterable[str] | None = None) -> list[str]:
    source = list(seeds or BASE_SEEDS)
    expressions: list[str] = []
    seen: set[str] = set(source)
    candidates: list[str] = []
    # Fitness-first candidates go first so each small cron batch spends capacity
    # on cutoff/fitness repair before falling back to local window mutations.
    candidates.extend(_fitness_first_refits())
    for expr in source:
        candidates.extend(_window_mutations(expr))
    candidates.extend(_fitness_variants())
    candidates.extend(_alt_function_dataset_grid())
    candidates.extend(_structural_grid())
    for variant in candidates:
        if variant not in seen:
            seen.add(variant)
            expressions.append(variant)
    return expressions[:count]


def append_unique_to_csv(csv_path: str | Path, expressions: Iterable[str], defaults: dict, prefix: str = "refit", max_add: int | None = None) -> int:
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
