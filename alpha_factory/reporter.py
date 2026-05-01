from __future__ import annotations

import csv
from pathlib import Path
from sqlite3 import Row
from typing import Any

from .database import Database
from .selector import evaluate_candidate

COLUMNS = [
    "alpha_id", "status", "simulation_id", "sharpe", "fitness", "turnover", "returns",
    "drawdown", "margin", "long_count", "short_count", "checks_passed", "fail_reasons", "error", "expression",
]


def export_reports(
    db: Database,
    results_csv: str | Path,
    markdown_report: str | Path,
    selection: dict[str, Any] | None = None,
) -> None:
    rows = db.all_results()
    _write_csv(rows, Path(results_csv))
    _write_markdown(rows, Path(markdown_report), selection or {})


def _write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row[col] for col in COLUMNS})


def _write_markdown(rows: list[Row], path: Path, selection: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# WorldQuant Brain Backtest Report", ""]
    lines.append(f"Total results: {len(rows)}")
    lines.append("")
    if not rows:
        lines.append("No results yet.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    complete = [r for r in rows if r["status"] == "COMPLETE"]
    passed = [r for r in complete if evaluate_candidate(r, selection)[0]] if selection else []
    failed = [r for r in complete if not evaluate_candidate(r, selection)[0]] if selection else []

    lines.append(f"Completed: {len(complete)}")
    lines.append(f"Errored/other: {len(rows) - len(complete)}")
    if selection:
        lines.append(f"Candidate pass: {len(passed)}")
        lines.append(f"Candidate fail: {len(failed)}")
    lines.append("")

    if selection:
        lines.append("## Selection Criteria")
        lines.append("")
        lines.append(f"- Sharpe >= `{selection.get('min_sharpe')}`")
        lines.append(f"- Fitness >= `{selection.get('min_fitness')}`")
        lines.append(f"- Turnover <= `{selection.get('max_turnover')}`")
        lines.append(f"- Checks passed = `{selection.get('require_checks_passed')}`")
        lines.append("")
        lines.append("## Passing Candidates")
        lines.append("")
        if passed:
            for row in passed[:20]:
                lines.extend(_row_block(row, selection))
        else:
            lines.append("No passing candidates yet.")
            lines.append("")

    lines.append("## Top Results")
    lines.append("")
    ranked = sorted(rows, key=lambda r: (r["fitness"] is not None, r["fitness"] or -999), reverse=True)
    for row in ranked[:20]:
        lines.extend(_row_block(row, selection))
    path.write_text("\n".join(lines), encoding="utf-8")


def _row_block(row: Row, selection: dict[str, Any]) -> list[str]:
    passed, reasons = evaluate_candidate(row, selection) if selection else (False, [])
    return [
        f"### {row['alpha_id']} — {row['status']}",
        "",
        f"- Simulation: `{row['simulation_id'] or ''}`",
        f"- Sharpe: `{row['sharpe']}` | Fitness: `{row['fitness']}` | Turnover: `{row['turnover']}`",
        f"- Returns: `{row['returns']}` | Drawdown: `{row['drawdown']}` | Margin: `{row['margin']}`",
        f"- Checks passed: `{row['checks_passed']}`",
        f"- Candidate pass: `{passed}`",
        f"- Candidate reasons: {', '.join(reasons) if reasons else 'pass'}",
        f"- Fail reasons: {row['fail_reasons'] or '—'}",
        f"- Error: {row['error'] or '—'}",
        "",
        "```text",
        row["expression"],
        "```",
        "",
    ]
