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
    "decay": 18,
    "neutralization": "INDUSTRY",
    "truncation": 0.05,
}


def build_expressions() -> list[str]:
    bucket = "bucket(rank(cap), range='0.1,1,0.1')"
    lowvol = "ts_rank(ts_std_dev(returns, 10), 252) < 0.8"
    liquid = "volume > adv20"
    big_move_filter = "abs(returns) < 0.075"
    high_liq = "rank(adv20) > 0.2"
    low_liq_exit = "rank(adv20) < 0.05"
    base = "rank(ts_rank(earnings_momentum_analyst_score, 120) - ts_rank(returns, 252))"
    base80 = "rank(ts_rank(earnings_momentum_analyst_score, 80) - ts_rank(returns, 160))"
    surprise = "rank(ts_rank(change_in_eps_surprise, 120) - ts_rank(returns, 252))"
    rev = "rank(ts_rank(earnings_revision_magnitude, 120) - ts_rank(returns, 252))"
    torpedo = "rank(ts_rank(earnings_torpedo_indicator, 120) - ts_rank(returns, 120))"
    ovn = "(open - ts_delay(close, 1)) / ts_delay(close, 1)"
    turn = "volume / sharesout"
    intraday = "close / open - 1"
    trend20 = "ts_mean(returns, 20)"
    mkt_ret = "group_mean(returns, 1, market)"
    excess5 = f"ts_mean(returns - {mkt_ret}, 5)"
    excess20 = f"ts_delay(ts_mean(returns - {mkt_ret}, 20), 20)"

    seeds = [base, base80, surprise, rev, torpedo]
    exprs: list[str] = []

    for s in seeds:
        exprs.extend([
            f"trade_when({lowvol}, group_neutralize({s}, {bucket}), -1)",
            f"trade_when({liquid}, group_neutralize({s}, {bucket}), -1)",
            f"trade_when({big_move_filter}, group_neutralize({s}, {bucket}), -1)",
            f"trade_when({high_liq}, group_neutralize({s}, {bucket}), {low_liq_exit})",
            f"trade_when(ts_rank(abs({ovn}), 252) < 0.7, group_neutralize({s}, {bucket}), -1)",
            f"trade_when(ts_rank(ts_std_dev({turn}, 20), 252) < 0.7, group_neutralize({s}, {bucket}), -1)",
        ])

    for raw in [
        "earnings_momentum_analyst_score",
        "change_in_eps_surprise",
        "earnings_revision_magnitude",
        "earnings_torpedo_indicator",
        "earnings_shortfall_metric",
    ]:
        delta = f"rank(ts_delta({raw}, 20))"
        shock = f"rank(ts_delta(ts_mean({raw}, 10), 60))"
        disagreement = f"rank(ts_rank({raw}, 60) - ts_rank({raw}, 252))"
        exprs.extend([
            f"trade_when(ts_rank(abs(ts_delta({raw}, 20)), 252) > 0.8, group_neutralize({delta}, {bucket}), -1)",
            f"trade_when({lowvol}, group_neutralize({shock} - rank({trend20}), {bucket}), -1)",
            f"group_neutralize({disagreement} - rank({excess5}) - rank({excess20}), {bucket})",
            f"trade_when({liquid}, group_neutralize({disagreement} * rank(-ts_delta(close, 5)), {bucket}), -1)",
        ])

    pv_events = [
        f"rank(-ts_corr(abs({ovn}), ts_delay({turn}, 1), 20))",
        f"rank(ts_corr(ts_delta({turn}, 1), abs({intraday}), 10))",
        "rank(power(ts_std_dev(abs(returns) - returns, 30), 2) - power(ts_std_dev(abs(returns) + returns, 30), 2))",
        "rank(ts_percentage(returns, 60, percentage=0.9) - ts_percentage(returns, 60, percentage=0.1))",
    ]
    for pv in pv_events:
        exprs.extend([
            f"trade_when({lowvol}, group_neutralize(rank({base} + {pv}), {bucket}), -1)",
            f"trade_when({liquid}, group_neutralize(rank({surprise} + {pv}), {bucket}), -1)",
            f"group_neutralize(rank({base80} * {pv}), {bucket})",
        ])

    for s in [base, surprise, rev]:
        exprs.extend([
            f"trade_when(rank({s}) > 0.8, group_neutralize({s}, {bucket}), -1)",
            f"trade_when(rank({s}) < 0.2, group_neutralize(-{s}, {bucket}), -1)",
            f"trade_when((rank({s}) > 0.75) * ({lowvol}), group_neutralize({s}, {bucket}), -1)",
        ])

    out: list[str] = []
    seen: set[str] = set()
    for expr in exprs:
        if expr not in seen:
            seen.add(expr)
            out.append(expr)
    return out


def main() -> int:
    added = append_unique_to_csv(ROOT / "alphas.csv", build_expressions(), DEFAULTS, prefix="sparse")
    print(f"Added {added} sparse/event alphas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
