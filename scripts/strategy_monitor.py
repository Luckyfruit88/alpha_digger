#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.sqlite_utils import active_backtest_rows, exact_prefix_clause, exact_prefix_param  # noqa: E402

STATE = ROOT / "state" / "strategy_monitor_state.json"
SUMMARY = ROOT / "reports" / "strategy_monitor_latest.md"
LOG = ROOT / "logs" / "strategy_monitor.log"
DB = ROOT / "data" / "backtests.sqlite3"
PY = ROOT / ".venv" / "bin" / "python"

STATE.parent.mkdir(exist_ok=True)
SUMMARY.parent.mkdir(exist_ok=True)
LOG.parent.mkdir(exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], timeout: int = 120) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def auth_status() -> dict:
    code, out = run([str(PY), "scripts/auth_status.py", "--json"], timeout=90)
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        data = {"ok": False, "error": out[-500:]}
    data["returncode"] = code
    return data


def q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


def reset_stuck_running(conn: sqlite3.Connection) -> int:
    # Conservative cleanup: tasks running >4h likely died with a killed process.
    cur = conn.execute(
        """
        update alpha_tasks
        set status='PENDING', simulation_id=NULL,
            last_error='strategy monitor reset stale RUNNING task', updated_at=CURRENT_TIMESTAMP
        where status='RUNNING' and datetime(updated_at) < datetime('now','-4 hours')
        """
    )
    conn.commit()
    return cur.rowcount


def choose_strategy(conn: sqlite3.Connection, auth: dict) -> tuple[str, list[str]]:
    actions: list[str] = []
    if not auth.get("ok"):
        return "auth_wait_generate_only", ["auth unavailable: keep generating/importing only; WQ API steps deferred"]

    active_rows = active_backtest_rows(conn)
    if active_rows:
        examples = ", ".join(f"{row.get('id')}:{row.get('status')}" for row in active_rows[:5])
        return "active_backtest_guard", [f"active backtests present; no new run launched ({examples})"]

    expiry = auth.get("expiry_seconds")
    if isinstance(expiry, (int, float)) and expiry < 1800:
        return "low_session_defer", [f"session short ({int(expiry)}s): defer heavy WQ API; let generators/backlog continue"]

    sparse_pending = q(conn, f"select count(*) n from alpha_tasks where {exact_prefix_clause('id')} and status='PENDING'", (exact_prefix_param("sparse_"),))[0]["n"]
    sparse_done = q(conn, f"select count(*) n from backtest_results where {exact_prefix_clause('alpha_id')}", (exact_prefix_param("sparse_"),))[0]["n"]
    simp_pending = q(conn, f"select count(*) n from alpha_tasks where {exact_prefix_clause('id')} and status='PENDING'", (exact_prefix_param("simp_"),))[0]["n"]
    tmpl_pending = q(conn, f"select count(*) n from alpha_tasks where {exact_prefix_clause('id')} and status='PENDING'", (exact_prefix_param("tmpl_"),))[0]["n"]

    if sparse_pending and sparse_done < 24:
        code, out = run([str(PY), "-m", "alpha_factory.cli", "run", "--limit", "6", "--id-prefix", "sparse_"], timeout=2700)
        actions.append(f"ran sparse_ batch limit=6 rc={code}: {out[-300:]}")
        return "run_sparse_event", actions

    # If sparse is exhausted/weak, sample simplified backlog lightly.
    if simp_pending:
        code, out = run([str(PY), "-m", "alpha_factory.cli", "run", "--limit", "4", "--id-prefix", "simp_"], timeout=2400)
        actions.append(f"ran simp_ batch limit=4 rc={code}: {out[-300:]}")
        return "run_simplified", actions

    # Avoid spending too much on raw tmpl_ due to high ERROR/weak rates.
    if tmpl_pending:
        actions.append("tmpl_ backlog remains, but raw template first sample was weak/high-error; not prioritized")

    return "observe", actions or ["no prioritized experimental backlog selected"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Observe and lightly route WorldQuant alpha strategy state")
    parser.add_argument("--observe-only", action="store_true", help="Collect local DB/report state without auth probes or WQ run launches")
    parser.add_argument("--allow-active-launches", action="store_true", help="Override active RUNNING/SUBMITTING/RETRY guard for launch decisions")
    args = parser.parse_args()

    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    reset_count = 0 if args.observe_only else reset_stuck_running(conn)
    active_rows = active_backtest_rows(conn)
    if args.observe_only:
        auth = {"ok": False, "observe_only": True}
        strategy, actions = "observe_only", ["observe-only mode: skipped auth probe and launch decisions"]
    elif active_rows and not args.allow_active_launches:
        auth = {"ok": False, "skipped": "active launch guard; auth probe deferred"}
        examples = ", ".join(f"{row.get('id')}:{row.get('status')}" for row in active_rows[:5])
        strategy, actions = "active_backtest_guard", [f"active backtests present; no auth probe or new run launched ({examples})"]
    else:
        auth = auth_status()
        strategy, actions = choose_strategy(conn, auth)

    queue = [dict(r) for r in q(conn, "select status, count(*) count from alpha_tasks group by status order by count(*) desc")]
    families = {}
    for prefix in ["sparse_", "simp_", "tmpl_", "decor_", "refit_"]:
        p = exact_prefix_param(prefix)
        families[prefix] = {
            "tasks": [dict(r) for r in q(conn, f"select status, count(*) count from alpha_tasks where {exact_prefix_clause('id')} group by status", (p,))],
            "best": [dict(r) for r in q(conn, """
                select alpha_id, round(sharpe,3) sharpe, round(fitness,3) fitness, round(turnover,3) turnover, fail_reasons
                from backtest_results where """ + exact_prefix_clause("alpha_id") + """ order by fitness desc, sharpe desc limit 5
            """, (p,))],
        }
    recent_pass = [dict(r) for r in q(conn, """
        select alpha_id, round(sharpe,3) sharpe, round(fitness,3) fitness, round(turnover,3) turnover, fail_reasons
        from backtest_results
        where sharpe>=1.6 and fitness>=1.0 and turnover<=0.45
        order by created_at desc limit 10
    """)]

    payload = {
        "timestamp_utc": now(),
        "auth": auth,
        "strategy": strategy,
        "actions": actions,
        "stale_running_reset": reset_count,
        "queue": queue,
        "families": families,
        "recent_pass": recent_pass,
    }
    STATE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        f"# WorldQuant Strategy Monitor — {payload['timestamp_utc']}",
        "",
        f"- Auth: `{auth.get('ok')}` user=`{auth.get('user')}` expiry_seconds=`{auth.get('expiry_seconds')}`",
        f"- Strategy: `{strategy}`",
        f"- Stale RUNNING reset: `{reset_count}`",
        "",
        "## Actions",
        *[f"- {a}" for a in actions],
        "",
        "## Queue",
        *[f"- {r['status']}: {r['count']}" for r in queue],
        "",
        "## Recent pass candidates",
        *[f"- {r['alpha_id']}: S={r['sharpe']} F={r['fitness']} T={r['turnover']} {r['fail_reasons'] or ''}" for r in recent_pass],
    ]
    SUMMARY.write_text("\n".join(md) + "\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp_utc": payload["timestamp_utc"], "strategy": strategy, "actions": actions, "auth_ok": auth.get("ok"), "expiry": auth.get("expiry_seconds")}, ensure_ascii=False) + "\n")
    print(json.dumps({"strategy": strategy, "actions": actions, "auth": auth, "summary": str(SUMMARY)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
