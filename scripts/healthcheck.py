from __future__ import annotations

import json
import os
import sqlite3
import sys
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from alpha_factory.sqlite_utils import exact_prefix_where  # noqa: E402

STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "health_state.json"
DB_PATH = ROOT / "data" / "backtests.sqlite3"
LOG_PATH = ROOT / "logs" / "supervisor.log"
TELEGRAM_ENV = ROOT / "secrets" / "telegram.env"
AUTONOMY_ENV = ROOT / "secrets" / "autonomy.env"
AUTH_RECOVER = ROOT / "scripts" / "auth_recover.py"

UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(UTC)


def load_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()
    return data


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"last_alerts": {}, "consecutive_low_quality_batches": 0}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"last_alerts": {}, "consecutive_low_quality_batches": 0}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def send_telegram(text: str) -> None:
    env = load_env_file(TELEGRAM_ENV)
    token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise RuntimeError("Missing Telegram credentials")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=20)
    response.raise_for_status()


def should_alert(state: dict[str, Any], key: str, cooldown_minutes: int = 240) -> bool:
    last = (state.get("last_alerts") or {}).get(key)
    if not last:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
    except Exception:
        return True
    return now_utc() - last_dt >= timedelta(minutes=cooldown_minutes)


def mark_alerted(state: dict[str, Any], key: str) -> None:
    state.setdefault("last_alerts", {})[key] = now_utc().isoformat()


def pgrep(pattern: str) -> list[str]:
    result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def latest_results(conn: sqlite3.Connection, limit: int = 12) -> list[tuple]:
    cur = conn.cursor()
    cur.execute(
        "select alpha_id, sharpe, fitness, turnover, fail_reasons, created_at from backtest_results order by created_at desc limit ?",
        (limit,),
    )
    return cur.fetchall()


def latest_completed_curated(conn: sqlite3.Connection, limit: int = 10) -> list[tuple]:
    cur = conn.cursor()
    where, params = exact_prefix_where("alpha_id", ["focus_", "hand_", "curated_"])
    cur.execute(
        f"select alpha_id, sharpe, fitness, turnover, returns, fail_reasons, expression from backtest_results where {where} order by created_at desc limit ?",
        params + (limit,),
    )
    return cur.fetchall()


def stop_supervisor() -> None:
    lines = pgrep("alpha_factory.supervisor|python -m alpha_factory.supervisor")
    for line in lines:
        pid = line.split()[0]
        subprocess.run(["kill", pid], check=False)


def pause_pending(conn: sqlite3.Connection, reason: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "update alpha_tasks set status='PAUSED_LOW_QUALITY_AUTO', last_error=?, updated_at=CURRENT_TIMESTAMP where status in ('PENDING','RETRY')",
        (reason,),
    )
    cur.execute(
        "update alpha_tasks set status='PAUSED_REVIEW_RUNNING', last_error=?, updated_at=CURRENT_TIMESTAMP where status in ('RUNNING','SUBMITTING')",
        (reason,),
    )
    conn.commit()


def auth_recover_enabled() -> bool:
    env = load_env_file(AUTONOMY_ENV)
    return env.get("WQ_AUTH_RECOVERY_DISABLED", "").lower() not in {"1", "true", "yes"}


def run_auth_recovery() -> tuple[bool, str]:
    if not AUTH_RECOVER.exists():
        return False, "auth recovery script missing"
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(AUTH_RECOVER)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, output[-1000:]



def critical_capability_probe() -> list[str]:
    """Probe endpoints closer to actual page/submit capability than /authentication.

    This catches half-expired sessions where /authentication still works but
    alpha detail or submit preflight is broken. It uses only safe GET/OPTIONS.
    """
    problems: list[str] = []
    try:
        from alpha_factory.backtester import load_config
        from alpha_factory.brain_client import BrainClient, BrainRateLimit
        cfg = load_config(ROOT / "config.yaml")
        client = BrainClient(**cfg["brain"])
        client.whoami()
        known_alpha = "d5lMo6xw"
        detail = client.get_alpha(known_alpha)
        if not detail.get("id"):
            problems.append("alpha detail probe returned no id")
        # Submit preflight should be 200 OPTIONS even if no actual submit is attempted.
        client._request("OPTIONS", f"/alphas/{known_alpha}/submit")
    except BrainRateLimit as exc:
        # Rate limit means the session is alive enough to be throttled. Treat it
        # as backoff/health degradation, not auth failure; do not stop workers.
        retry = f" retry_after={exc.retry_after}s" if exc.retry_after else ""
        problems.append(f"capability probe rate-limited; backing off.{retry}")
    except Exception as exc:
        problems.append(f"critical capability probe failed: {type(exc).__name__}: {exc}")
    return problems

def main() -> None:
    state = load_state()
    conn = sqlite3.connect(DB_PATH)
    problems: list[str] = []
    actions: list[str] = []
    log_tail = tail_log(LOG_PATH)
    supervisor_lines = pgrep("alpha_factory.supervisor|python -m alpha_factory.supervisor")

    capability_problems = critical_capability_probe()
    if capability_problems:
        problems.extend(capability_problems)
        hard_failures = [p for p in capability_problems if "rate-limited" not in p.lower()]
        if hard_failures:
            recovered = False
            recovery_detail = ""
            if auth_recover_enabled():
                recovered, recovery_detail = run_auth_recovery()
                actions.append(f"auth recovery attempted: {'success' if recovered else 'failed'}")
            if recovered:
                problems = [p for p in problems if p not in capability_problems]
                actions.append("WorldQuant auth recovered; leaving focused workers available")
            else:
                stop_supervisor()
                actions.append("stopped supervisor because critical WQ capability probe failed")
            if should_alert(state, "capability_probe", cooldown_minutes=60):
                send_telegram("[WQ ALERT] WorldQuant page/API capability probe failed." + (" Auth recovery failed. " if auth_recover_enabled() else " ") + "Supervisor stopped. Details: " + "; ".join(hard_failures[:3]) + (f"\nRecovery: {recovery_detail[:500]}" if recovery_detail else ""))
                mark_alerted(state, "capability_probe")
        else:
            actions.append("capability probe rate-limited; leaving workers alone")

    if "unauthorized/expired" in log_tail.lower() or "auth_error" in log_tail.lower():
        problems.append("WorldQuant auth appears expired")
        recovered = False
        recovery_detail = ""
        if auth_recover_enabled():
            recovered, recovery_detail = run_auth_recovery()
            actions.append(f"auth recovery attempted after log auth error: {'success' if recovered else 'failed'}")
        if recovered:
            actions.append("WorldQuant auth recovered after log auth error")
        else:
            stop_supervisor()
            actions.append("stopped supervisor to avoid repeated unauthorized requests")
        if should_alert(state, "auth_error", cooldown_minutes=180):
            send_telegram("[WQ ALERT] WorldQuant auth appears expired." + (" Auth recovery failed; supervisor stopped." if not recovered else " Auth recovery succeeded.") + (f"\nRecovery: {recovery_detail[:500]}" if recovery_detail else ""))
            mark_alerted(state, "auth_error")

    if not supervisor_lines:
        actions.append("supervisor not running")

    rows = latest_results(conn, limit=12)
    low_quality = 0
    total_recent = 0
    for _alpha_id, sharpe, fitness, turnover, fail_reasons, _created_at in rows:
        total_recent += 1
        fr = fail_reasons or ""
        if (sharpe is not None and sharpe < 1.6) or (fitness is not None and fitness < 1.0) or (turnover is not None and turnover > 0.45) or ("LOW_" in fr) or ("HIGH_TURNOVER" in fr):
            low_quality += 1

    # Low-quality batches should slow/stop broad random exploration, but must not
    # permanently block focused refit/decorrelation pipelines after auth recovers.
    if total_recent >= 6 and low_quality / total_recent >= 0.8:
        state["consecutive_low_quality_batches"] = int(state.get("consecutive_low_quality_batches", 0)) + 1
    else:
        state["consecutive_low_quality_batches"] = 0

    if state["consecutive_low_quality_batches"] >= 2:
        problems.append(f"consecutive low-quality batches detected ({state['consecutive_low_quality_batches']})")
        stop_supervisor()
        actions.append("stopped broad supervisor after repeated low-quality batches; focused refit/decor pipelines remain allowed")
        if should_alert(state, "low_quality_pause", cooldown_minutes=360):
            send_telegram(
                f"[WQ ALERT] Broad supervisor paused after consecutive low-quality batches. Focused refit/decor pipelines remain allowed."
            )
            mark_alerted(state, "low_quality_pause")

    # Detect long stale state: no recent completed result for >3h while supervisor is running.
    cur = conn.cursor()
    cur.execute("select created_at from backtest_results order by created_at desc limit 1")
    row = cur.fetchone()
    if row and row[0]:
        try:
            latest_dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
            if supervisor_lines and now_utc() - latest_dt > timedelta(hours=3):
                problems.append("supervisor running but no results updated in >3h")
                if should_alert(state, "stale_results", cooldown_minutes=240):
                    send_telegram("[WQ ALERT] Supervisor appears stale: running, but no new backtest results in over 3 hours.")
                    mark_alerted(state, "stale_results")
        except Exception:
            pass

    # Optional auto-submit candidate report; actual submission remains handled by scripts/auto_submit.py.
    strong = latest_completed_curated(conn, limit=5)
    for alpha_id, sharpe, fitness, turnover, returns, fail_reasons, expr in strong:
        if sharpe is None or fitness is None or turnover is None:
            continue
        if sharpe >= 1.6 and fitness >= 1.0 and turnover <= 0.45 and not (fail_reasons or ""):
            key = f"candidate_ready:{alpha_id}"
            if should_alert(state, key, cooldown_minutes=720):
                send_telegram(
                    "[WQ INFO] Candidate ready for auto-submit threshold:\n"
                    f"alpha={alpha_id}\nsharpe={sharpe}\nfitness={fitness}\nturnover={turnover}\nreturns={returns}\nexpr={expr[:220]}"
                )
                mark_alerted(state, key)

    status_line = {
        "timestamp": now_utc().isoformat(),
        "problems": problems,
        "actions": actions,
        "supervisor_running": bool(supervisor_lines),
        "consecutive_low_quality_batches": state.get("consecutive_low_quality_batches", 0),
    }
    state["last_status"] = status_line
    save_state(state)
    conn.close()
    print(json.dumps(status_line, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            send_telegram(f"[WQ ALERT] Healthcheck script failed: {exc}")
        except Exception:
            pass
        raise
