from __future__ import annotations

from typing import Any


def evaluate_candidate(row: Any, selection: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    min_sharpe = selection.get("min_sharpe")
    min_fitness = selection.get("min_fitness")
    max_turnover = selection.get("max_turnover")
    require_checks_passed = bool(selection.get("require_checks_passed", False))

    sharpe = _num(_get(row, "sharpe"))
    fitness = _num(_get(row, "fitness"))
    turnover = _num(_get(row, "turnover"))
    checks_passed = _get(row, "checks_passed")

    if min_sharpe is not None and (sharpe is None or sharpe < float(min_sharpe)):
        reasons.append(f"sharpe<{min_sharpe}")
    if min_fitness is not None and (fitness is None or fitness < float(min_fitness)):
        reasons.append(f"fitness<{min_fitness}")
    if max_turnover is not None and (turnover is None or turnover > float(max_turnover)):
        reasons.append(f"turnover>{max_turnover}")
    if require_checks_passed and not _is_true(checks_passed):
        reasons.append("checks_passed!=true")

    return (len(reasons) == 0, reasons)


def _get(row: Any, key: str) -> Any:
    try:
        return row[key]
    except Exception:
        return getattr(row, key, None)


def _num(value: Any) -> float | None:
    try:
        return None if value is None or value == "" else float(value)
    except (TypeError, ValueError):
        return None


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (1, "1", "true", "TRUE", "True"):
        return True
    return False
