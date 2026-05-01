from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .models import AlphaTask, BacktestResult
from .sqlite_utils import active_backtest_rows, exact_prefix_clause, exact_prefix_param


SCHEMA = """
CREATE TABLE IF NOT EXISTS alpha_tasks (
  id TEXT PRIMARY KEY,
  expression TEXT NOT NULL,
  settings_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PENDING',
  attempts INTEGER NOT NULL DEFAULT 0,
  simulation_id TEXT,
  last_error TEXT,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_results (
  alpha_id TEXT PRIMARY KEY,
  expression TEXT NOT NULL,
  status TEXT NOT NULL,
  simulation_id TEXT,
  sharpe REAL,
  fitness REAL,
  turnover REAL,
  returns REAL,
  drawdown REAL,
  margin REAL,
  long_count INTEGER,
  short_count INTEGER,
  checks_passed INTEGER,
  fail_reasons TEXT,
  raw_json TEXT,
  error TEXT,
  created_at TEXT
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def upsert_tasks(self, tasks: Iterable[AlphaTask]) -> None:
        with self.conn:
            for task in tasks:
                self.conn.execute(
                    """
                    INSERT INTO alpha_tasks (id, expression, settings_json, status, attempts, simulation_id, last_error)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                      expression=excluded.expression,
                      settings_json=excluded.settings_json,
                      updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        task.id,
                        task.expression,
                        json.dumps(task.settings, ensure_ascii=False),
                        task.status,
                        task.attempts,
                        task.simulation_id,
                        task.last_error,
                    ),
                )

    def pending_tasks(self, limit: int | None = None, id_prefix: str | None = None) -> list[AlphaTask]:
        params: list[object] = []
        sql = "SELECT * FROM alpha_tasks WHERE status IN ('PENDING', 'RETRY')"
        if id_prefix:
            sql += f" AND {exact_prefix_clause('id')}"
            params.append(exact_prefix_param(id_prefix))
        sql += " ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            AlphaTask(
                id=row["id"],
                expression=row["expression"],
                settings=json.loads(row["settings_json"]),
                status=row["status"],
                attempts=row["attempts"],
                simulation_id=row["simulation_id"],
                last_error=row["last_error"],
            )
            for row in rows
        ]

    def active_launch_rows(self, retry_window_minutes: int = 45, limit: int = 20) -> list[dict]:
        return active_backtest_rows(self.conn, retry_window_minutes=retry_window_minutes, limit=limit)

    def running_tasks(self) -> list[AlphaTask]:
        rows = self.conn.execute("SELECT * FROM alpha_tasks WHERE status='RUNNING' AND simulation_id IS NOT NULL ORDER BY updated_at").fetchall()
        return [
            AlphaTask(
                id=row["id"],
                expression=row["expression"],
                settings=json.loads(row["settings_json"]),
                status=row["status"],
                attempts=row["attempts"],
                simulation_id=row["simulation_id"],
                last_error=row["last_error"],
            )
            for row in rows
        ]

    def mark_task(self, task: AlphaTask) -> None:
        with self.conn:
            self.conn.execute(
                """
                UPDATE alpha_tasks
                SET status=?, attempts=?, simulation_id=?, last_error=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (task.status, task.attempts, task.simulation_id, task.last_error, task.id),
            )

    def save_result(self, result: BacktestResult) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO backtest_results (
                  alpha_id, expression, status, simulation_id, sharpe, fitness, turnover,
                  returns, drawdown, margin, long_count, short_count, checks_passed,
                  fail_reasons, raw_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(alpha_id) DO UPDATE SET
                  status=excluded.status,
                  simulation_id=excluded.simulation_id,
                  sharpe=excluded.sharpe,
                  fitness=excluded.fitness,
                  turnover=excluded.turnover,
                  returns=excluded.returns,
                  drawdown=excluded.drawdown,
                  margin=excluded.margin,
                  long_count=excluded.long_count,
                  short_count=excluded.short_count,
                  checks_passed=excluded.checks_passed,
                  fail_reasons=excluded.fail_reasons,
                  raw_json=excluded.raw_json,
                  error=excluded.error,
                  created_at=excluded.created_at
                """,
                (
                    result.alpha_id,
                    result.expression,
                    result.status,
                    result.simulation_id,
                    result.sharpe,
                    result.fitness,
                    result.turnover,
                    result.returns,
                    result.drawdown,
                    result.margin,
                    result.long_count,
                    result.short_count,
                    None if result.checks_passed is None else int(result.checks_passed),
                    result.fail_reasons,
                    json.dumps(result.raw_json, ensure_ascii=False),
                    result.error,
                    result.created_at,
                ),
            )

    def all_results(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM backtest_results ORDER BY created_at DESC").fetchall()
