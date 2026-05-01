from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(slots=True)
class AlphaTask:
    id: str
    expression: str
    settings: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    attempts: int = 0
    simulation_id: Optional[str] = None
    last_error: Optional[str] = None


@dataclass(slots=True)
class BacktestResult:
    alpha_id: str
    expression: str
    status: str
    simulation_id: Optional[str] = None
    sharpe: Optional[float] = None
    fitness: Optional[float] = None
    turnover: Optional[float] = None
    returns: Optional[float] = None
    drawdown: Optional[float] = None
    margin: Optional[float] = None
    long_count: Optional[int] = None
    short_count: Optional[int] = None
    checks_passed: Optional[bool] = None
    fail_reasons: str = ""
    raw_json: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
