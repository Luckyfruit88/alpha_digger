from __future__ import annotations

import sqlite3
from typing import Any, Iterable


def escape_like(value: str) -> str:
    """Escape user/prefix text for SQLite LIKE with an explicit ESCAPE char."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def exact_prefix_param(prefix: str) -> str:
    return escape_like(prefix) + "%"


def exact_prefix_clause(column: str) -> str:
    return f"{column} LIKE ? ESCAPE '\\'"


def exact_prefix_where(column: str, prefixes: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    prefix_list = list(prefixes)
    if not prefix_list:
        return "1=0", ()
    clause = " OR ".join(exact_prefix_clause(column) for _ in prefix_list)
    return clause, tuple(exact_prefix_param(prefix) for prefix in prefix_list)


def active_backtest_rows(
    conn: sqlite3.Connection,
    retry_window_minutes: int = 45,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Rows that should block launching fresh simulations.

    RUNNING/SUBMITTING are always active. RETRY is considered active when it
    still has a simulation id or was updated recently, because the previous
    launch may still be settling or rate-limited.
    """
    rows = conn.execute(
        """
        select id, status, simulation_id, attempts, updated_at, last_error
        from alpha_tasks
        where status in ('RUNNING', 'SUBMITTING')
           or (
                status = 'RETRY'
                and (
                    simulation_id is not null
                    or datetime(updated_at) >= datetime('now', ?)
                )
           )
        order by datetime(updated_at) desc
        limit ?
        """,
        (f"-{int(retry_window_minutes)} minutes", int(limit)),
    ).fetchall()
    return [dict(row) for row in rows]
