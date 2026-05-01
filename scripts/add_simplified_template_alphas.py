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
    "decay": 12,
    "neutralization": "INDUSTRY",
    "truncation": 0.05,
}


def build_expressions() -> list[str]:
    fields = [
        "earnings_momentum_analyst_score",
        "change_in_eps_surprise",
        "earnings_revision_magnitude",
        "earnings_torpedo_indicator",
        "earnings_shortfall_metric",
        "analyst_revision_rank_derivative",
        "eps_estimate_value",
        "earnings_per_share_reported_value",
        "actual_eps_value_quarterly",
        "actual_cashflow_per_share_value_quarterly",
    ]
    bucket = "bucket(rank(cap), range='0.1,1,0.1')"
    mkt = "group_mean(returns, 1, market)"
    ir = "abs(ts_mean(returns, 252) / (ts_std_dev(returns, 252) + 0.001))"
    exprs: list[str] = []

    # Simplified #36/#37: reversal and small-stable, avoiding vector_neut/regression_neut nesting.
    for f in fields:
        exprs.append(f"group_neutralize(-ts_delta({f}, 3), subindustry)")
        exprs.append(f"group_neutralize(-rank({f}) * rank(ts_std_dev({f}, 20)), subindustry)")
        exprs.append(f"group_neutralize(rank(ts_delta(ts_mean({f}, 20), 60)), {bucket})")
        exprs.append(f"group_neutralize(rank(ts_rank({f}, 120) - ts_rank({ir}, 120)), {bucket})")
        exprs.append(f"trade_when(ts_rank(ts_std_dev(returns, 10), 252) < 0.9, group_neutralize(rank({f}), {bucket}), -1)")

    # Simplified #18/#15: residual idea approximated by subtracting short/long excess return terms.
    for f in fields[:6]:
        short_excess = f"ts_mean(returns - {mkt}, 5)"
        long_excess = f"ts_delay(ts_mean(returns - {mkt}, 20), 20)"
        exprs.append(f"group_neutralize(rank({f}) - rank({short_excess}) - rank({long_excess}), {bucket})")
        exprs.append(f"trade_when(ts_rank(ts_std_dev({mkt}, 10), 252) < 0.9, group_neutralize(rank({f}) - rank({short_excess}), {bucket}), -1)")

    # Simplified price-volume/overnight/turnover templates (#34/#35/#40/#42/#43/#44/#47).
    turn = "volume / sharesout"
    ovn = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"
    intraday = "close / open - 1"
    shock = "(high - ts_delay(low, 1)) / ts_delay(low, 1)"
    exprs.extend([
        "rank(power(ts_std_dev(abs(returns), 30), 2) - power(ts_std_dev(returns, 30), 2))",
        "group_neutralize(rank(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2)), subindustry)",
        f"rank(-ts_corr(abs({ovn}), ts_delay({turn}, 1), 7))",
        f"group_neutralize(rank(-ts_corr(abs({ovn}), ts_delay({turn}, 1), 20)), {bucket})",
        f"rank(ts_corr(ts_delta({turn}, 1), abs({intraday}), 10))",
        f"group_neutralize(rank(-ts_std_dev({turn}, 20) / (ts_mean({turn}, 20) + 0.001)), {bucket})",
        f"group_neutralize(rank(-ts_mean(({shock} - log({shock} + 1)) * 2 - power(log({shock} + 1), 2), 24)), {bucket})",
        "group_neutralize(rank(-ts_sum(returns, 20) * rank(ts_percentage(returns, 60, percentage=0.9) - ts_percentage(returns, 60, percentage=0.1))), bucket(rank(cap), range='0.1,1,0.1'))",
    ])

    # Controlled combinations with the proven family, but more structurally different.
    proven = "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))"
    exprs.extend([
        f"trade_when(ts_rank(ts_std_dev(returns, 10), 252) < 0.8, group_neutralize({proven}, {bucket}), -1)",
        f"group_neutralize(rank({proven} - rank(ts_std_dev({turn}, 30))), {bucket})",
        f"group_neutralize(rank({proven} + rank(-ts_corr(abs({ovn}), ts_delay({turn}, 1), 20))), {bucket})",
        f"trade_when(volume > adv20, group_neutralize(rank({proven} * rank(-ts_delta(close, 5))), {bucket}), -1)",
    ])

    out: list[str] = []
    seen: set[str] = set()
    for e in exprs:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def main() -> int:
    added = append_unique_to_csv(ROOT / "alphas.csv", build_expressions(), DEFAULTS, prefix="simp")
    print(f"Added {added} simplified template alphas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
