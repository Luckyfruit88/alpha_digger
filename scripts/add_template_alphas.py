#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.generator import append_unique_to_csv


DEFAULTS = {
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 10,
    "neutralization": "INDUSTRY",
    "truncation": 0.05,
}


def build_expressions() -> list[str]:
    bucket = "bucket(rank(cap), range='0.1,1,0.1')"
    ir126 = "abs(ts_mean(returns, 126) / (ts_std_dev(returns, 126) + 0.001))"
    ir252 = "abs(ts_mean(returns, 252) / (ts_std_dev(returns, 252) + 0.001))"

    analyst_like = [
        "earnings_momentum_analyst_score",
        "change_in_eps_surprise",
        "earnings_revision_magnitude",
        "analyst_revision_rank_derivative",
        "eps_estimate_value",
        "earnings_per_share_reported_value",
    ]
    fundamental_like = [
        "actual_eps_value_quarterly",
        "actual_cashflow_per_share_value_quarterly",
        "actual_sales_value_annual",
        "earnings_per_share_reported_value",
        "eps_estimate_value",
        "change_in_eps_surprise",
    ]

    exprs: list[str] = []

    # Template #34/#35: asymmetric volatility / upside-vs-downside volatility residual.
    exprs.extend([
        "power(ts_std_dev(abs(returns), 30), 2) - power(ts_std_dev(returns, 30), 2)",
        f"group_neutralize(regression_neut(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2), {ir252}), {bucket})",
        f"group_neutralize(regression_neut(power(ts_std_dev(abs(returns) + returns, 20), 2) - power(ts_std_dev(abs(returns) - returns, 20), 2), {ir126}), {bucket})",
    ])

    # Template #36/#37: reversal / small-and-stable variants, de-IR'ed.
    for field in fundamental_like:
        exprs.append(f"group_neutralize(vector_neut(-ts_delta({field}, 3), {ir252}), subindustry)")
        exprs.append(f"group_neutralize(vector_neut(-{field} * ts_std_dev({field}, 20), {ir252}), subindustry)")
        exprs.append(f"regression_neut(ts_zscore({field}, 252), ts_zscore(cap, 252))")

    # Template #22/#9/#14: analyst high/close residualized by size and IR.
    for field in analyst_like:
        exprs.append(f"regression_neut(vector_neut(ts_rank({field} / close, 120), ts_median(cap, 120)), {ir252})")
        exprs.append(f"rank(vector_neut(vector_neut(group_rank(ts_rank({field}, 60), {bucket}), group_neutralize(close, {bucket})), {ir126}))")

    # Template #40/#42/#43/#44: pure price-volume turnover/shock templates.
    turnover = "volume / sharesout"
    overnight = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"
    shock = "(high - ts_delay(low, 1)) / ts_delay(low, 1)"
    exprs.extend([
        f"ts_corr(ts_delay(({turnover}) - ts_mean({turnover}, 30), 3), abs((close - open) / open - ts_mean((close - open) / open, 30)), 10)",
        f"-ts_corr(abs({overnight}), ts_delay({turnover}, 1), 7)",
        f"-group_rank(group_neutralize(ts_std_dev({turnover}, 20), {bucket}), {bucket})",
        f"group_neutralize(-group_rank(ts_mean(({shock} - log({shock} + 1)) * 2 - power(log({shock} + 1), 2), 24), {bucket}), {bucket})",
        f"trade_when(ts_rank(ts_std_dev(returns, 10), 252) < 0.9, -regression_neut(group_neutralize(ts_std_dev({turnover}, 30) / (ts_mean({turnover}, 30) + 0.001), {bucket}), ts_std_dev(returns, 30)), -1)",
    ])

    # Template #18/#15 inspired: residualize analyst/fundamental signal against short/long excess returns.
    market_ret = "group_mean(returns, 1, market)"
    for field in analyst_like[:4]:
        short = f"ts_mean(returns - {market_ret}, 5)"
        long = f"ts_delay(ts_mean(returns - {market_ret}, 20), 20)"
        exprs.append(f"trade_when(ts_rank(ts_std_dev({market_ret}, 10), 252) < 0.9, group_neutralize(regression_neut(regression_neut({field}, {short}), {long}), {bucket}), -1)")

    # De-duplicate preserving order.
    out: list[str] = []
    seen: set[str] = set()
    for expr in exprs:
        if expr not in seen:
            seen.add(expr)
            out.append(expr)
    return out


def main() -> int:
    added = append_unique_to_csv(ROOT / "alphas.csv", build_expressions(), DEFAULTS, prefix="tmpl")
    print(f"Added {added} template alphas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

