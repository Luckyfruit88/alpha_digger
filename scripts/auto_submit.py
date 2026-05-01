from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_factory.backtester import load_config  # noqa: E402
from alpha_factory.brain_client import BrainClient  # noqa: E402
from alpha_factory.sqlite_utils import exact_prefix_clause, exact_prefix_param  # noqa: E402
from scripts.healthcheck import send_telegram  # noqa: E402

DB_PATH = ROOT / "data" / "backtests.sqlite3"
AUTONOMY_ENV = ROOT / "secrets" / "autonomy.env"
SUBMISSIONS_PATH = ROOT / "data" / "auto_submissions.jsonl"
AUTO_SUBMIT_STATE_PATH = ROOT / "state" / "auto_submit_state.json"
ML_STATE_PATH = ROOT / "state" / "ml_candidate_scorer_state.json"
D1_TRUTH_TABLE_PATH = ROOT / "state" / "d1_truth_table.json"


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def scorer_index() -> dict[str, dict[str, Any]]:
    payload = load_json(ML_STATE_PATH)
    rows = payload.get("top_candidates") or []
    return {str(row.get("alpha_id")): row for row in rows if isinstance(row, dict) and row.get("alpha_id")}


def should_allow_exploration_detail(
    score_meta: dict[str, Any],
    p_self_corr: float,
    max_self_corr_risk: float,
    max_exploration_p_self_corr: float,
    exploration_checks_used: int,
    exploration_check_budget: int,
) -> tuple[bool, str]:
    if p_self_corr <= max_self_corr_risk:
        return False, ""
    if exploration_checks_used >= exploration_check_budget:
        return False, "self-correlation exploration budget exhausted"
    if p_self_corr > max_exploration_p_self_corr:
        return False, f"exploration p_self_corr_block={p_self_corr:.3f} > {max_exploration_p_self_corr:.2f}"
    if not bool(score_meta.get("exploration_candidate")):
        return False, "not marked as exploration_candidate by scorer"
    reason = str(score_meta.get("exploration_reason") or "scorer exploration candidate")
    return True, reason


def is_enabled(value: str, default: bool = True) -> bool:
    if value == "":
        return default
    return value not in {"0", "false", "False", "no", "NO"}


def is_d1_candidate(alpha_id: str) -> bool:
    return alpha_id.startswith(("d1_", "d1v2_", "d1v23_"))


def should_allow_fde_detail(
    score_meta: dict[str, Any],
    checks_used: int,
    check_budget: int,
    enabled: bool,
) -> tuple[bool, str]:
    if not enabled:
        return False, "FDE disabled"
    if checks_used >= check_budget:
        return False, "FDE detail budget exhausted"
    if not bool(score_meta.get("fde_candidate")):
        return False, "not marked as fde_candidate by scorer"
    return True, f"forced diversity exploration lineage={score_meta.get('lineage_theme', 'unknown')}"


def should_allow_d1_detail(
    local_id: str,
    score_meta: dict[str, Any],
    p_self_corr: float,
    checks_used: int,
    check_budget: int,
    max_p_self_corr: float,
    enabled: bool,
) -> tuple[bool, str]:
    if not enabled:
        return False, "D1 exploration disabled"
    if checks_used >= check_budget:
        return False, "D1 exploration detail budget exhausted"
    if not is_d1_candidate(local_id):
        return False, "not a D1 candidate prefix"
    if int(score_meta.get("pass_quality", 0) or 0) < 1:
        return False, "D1 candidate did not pass quality"
    if p_self_corr > max_p_self_corr:
        return False, f"D1 p_self_corr_block={p_self_corr:.3f} > {max_p_self_corr:.2f}"
    return True, f"D1 relaxed exploration p_self_corr={p_self_corr:.3f}"


def worldquant_alpha_id(raw_json: str | None) -> str | None:
    if not raw_json:
        return None
    try:
        raw = json.loads(raw_json)
    except Exception:
        return None
    detail = raw.get("alpha_detail") if isinstance(raw.get("alpha_detail"), dict) else {}
    return str(detail.get("id") or raw.get("worldquant_alpha_id") or raw.get("alpha") or "") or None


def already_recorded(alpha_id: str) -> bool:
    if not SUBMISSIONS_PATH.exists():
        return False
    return any((f'"alpha_id":"{alpha_id}"' in line) or (f'"alpha_id": "{alpha_id}"' in line) for line in SUBMISSIONS_PATH.read_text(encoding="utf-8").splitlines())


def record(payload: dict[str, Any]) -> None:
    SUBMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUBMISSIONS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_state() -> dict[str, Any]:
    if not AUTO_SUBMIT_STATE_PATH.exists():
        return {"cooldowns": {}}
    try:
        return json.loads(AUTO_SUBMIT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"cooldowns": {}}


def save_state(state: dict[str, Any]) -> None:
    AUTO_SUBMIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_SUBMIT_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def utc_day() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def daily_submit_count(state: dict[str, Any]) -> int:
    today = utc_day()
    if state.get("submit_day") != today:
        state["submit_day"] = today
        state["submit_count"] = 0
    return int(state.get("submit_count", 0) or 0)


def increment_daily_submit_count(state: dict[str, Any]) -> None:
    today = utc_day()
    if state.get("submit_day") != today:
        state["submit_day"] = today
        state["submit_count"] = 0
    state["submit_count"] = int(state.get("submit_count", 0) or 0) + 1


def load_d1_truth_table() -> dict[str, Any]:
    payload = load_json(D1_TRUTH_TABLE_PATH)
    if not payload:
        return {"updated_at": "", "total": 0, "clear": 0, "blocked": 0, "pending": 0, "clear_rate": 0.0, "entries": []}
    payload.setdefault("entries", [])
    return payload


def save_d1_truth_entry(alpha_id: str, family: str, p_self_corr: float, result: str) -> None:
    payload = load_d1_truth_table()
    entries = [entry for entry in payload.get("entries", []) if isinstance(entry, dict) and entry.get("alpha_id") != alpha_id]
    entries.append({
        "alpha_id": alpha_id,
        "family": family,
        "p_self_corr_block": round(p_self_corr, 4),
        "detail_check_result": result,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    })
    counts = {name: sum(1 for entry in entries if entry.get("detail_check_result") == name) for name in ["clear", "blocked", "pending"]}
    payload.update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(entries),
        "clear": counts["clear"],
        "blocked": counts["blocked"],
        "pending": counts["pending"],
        "clear_rate": round(counts["clear"] / max(1, len(entries)), 4),
        "entries": entries,
    })
    D1_TRUTH_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    D1_TRUTH_TABLE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def detail_result_from_unsafe(unsafe_checks: list[str]) -> str:
    if not unsafe_checks:
        return "clear"
    if any(item.startswith("SELF_CORRELATION:PENDING") for item in unsafe_checks):
        return "pending"
    return "blocked"


def cooldown_active(state: dict[str, Any], wq_id: str, hours: int) -> bool:
    value = (state.get("cooldowns") or {}).get(wq_id)
    if not value:
        return False
    try:
        last = datetime.fromisoformat(value)
    except Exception:
        return False
    return datetime.now(timezone.utc) - last < timedelta(hours=hours)


def mark_cooldown(state: dict[str, Any], wq_id: str) -> None:
    state.setdefault("cooldowns", {})[wq_id] = datetime.now(timezone.utc).isoformat()


def candidate_rows(conn: sqlite3.Connection, prefix: str | None = None) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    where = "sharpe is not null and fitness is not null and turnover is not null"
    params: list[Any] = []
    if prefix:
        where += f" and {exact_prefix_clause('alpha_id')}"
        params.append(exact_prefix_param(prefix))
    return conn.execute(
        f"select * from backtest_results where {where} order by sharpe desc, fitness desc limit 80",
        params,
    ).fetchall()


def passes(row: sqlite3.Row, min_sharpe: float, min_fitness: float, max_turnover: float) -> bool:
    fail_reasons = row["fail_reasons"] or ""
    return (
        float(row["sharpe"]) >= min_sharpe
        and float(row["fitness"]) >= min_fitness
        and float(row["turnover"]) <= max_turnover
        and not fail_reasons.strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit threshold-passing WorldQuant alphas")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prefix", default=None, help="Optional local alpha id prefix filter")
    args = parser.parse_args()

    env = load_env_file(AUTONOMY_ENV)
    min_sharpe = float(env.get("AUTO_SUBMIT_MIN_SHARPE", "1.60"))
    min_fitness = float(env.get("AUTO_SUBMIT_MIN_FITNESS", "1.00"))
    max_turnover = float(env.get("AUTO_SUBMIT_MAX_TURNOVER", "0.45"))
    max_self_corr_risk = float(env.get("AUTO_SUBMIT_MAX_P_SELF_CORR", "0.20"))
    exploration_check_budget = int(env.get("AUTO_SUBMIT_SELF_CORR_EXPLORATION_CHECKS", "2"))
    max_exploration_p_self_corr = float(env.get("AUTO_SUBMIT_MAX_EXPLORATION_P_SELF_CORR", "0.75"))
    fde_enabled = is_enabled(env.get("AUTO_SUBMIT_FDE_ENABLED", "1"))
    fde_check_budget = int(env.get("AUTO_SUBMIT_FDE_CHECKS_PER_ROUND", "2"))
    d1_exploration_enabled = is_enabled(env.get("AUTO_SUBMIT_D1_EXPLORATION_ENABLED", "1"))
    d1_max_p_self_corr = float(env.get("AUTO_SUBMIT_D1_MAX_P_SELF_CORR", "0.35"))
    d1_check_budget = int(env.get("AUTO_SUBMIT_D1_CHECKS_PER_ROUND", "2"))
    require_d1_gate = env.get("AUTO_SUBMIT_REQUIRE_D1_READY", "1") not in {"0", "false", "False"}
    daily_limit = int(env.get("AUTO_SUBMIT_DAILY_LIMIT", "1"))

    cfg = load_config(ROOT / "config.yaml")
    client = BrainClient(**cfg["brain"])
    conn = sqlite3.connect(DB_PATH)
    scored = scorer_index()
    candidates = [row for row in candidate_rows(conn, args.prefix) if passes(row, min_sharpe, min_fitness, max_turnover)]
    candidates.sort(
        key=lambda row: float((scored.get(str(row["alpha_id"])) or {}).get("quality_score", 0.0) or 0.0),
        reverse=True,
    )
    state = load_state()
    state["fde_checks_this_round"] = 0
    state["d1_checks_this_round"] = 0
    state.setdefault("fde_total_checks", 0)
    state.setdefault("fde_clear_count", 0)
    state.setdefault("fde_blocked_count", 0)
    state.setdefault("d1_total_checks", 0)
    self_corr_cooldown_hours = int(env.get("AUTO_SUBMIT_SELF_CORR_COOLDOWN_HOURS", "6"))
    max_detail_checks = int(env.get("AUTO_SUBMIT_MAX_DETAIL_CHECKS", "6"))
    submitted = []
    skipped = []
    detail_checks = 0
    exploration_checks = 0
    fde_checks = 0
    d1_checks = 0
    for row in candidates:
        if daily_submit_count(state) >= daily_limit:
            skipped.append((str(row["alpha_id"]), f"daily submit cap reached ({daily_submit_count(state)}/{daily_limit})"))
            continue
        local_id = row["alpha_id"]
        score_meta = scored.get(str(local_id), {})
        submitted_similarity = float(score_meta.get("submitted_similarity_max", 0.0) or 0.0) if score_meta else 0.0
        if submitted_similarity >= 0.85:
            skipped.append((local_id, f"submitted-reference high collision: similarity={submitted_similarity:.3f} nearest={score_meta.get('submitted_nearest_alpha_id')}"))
            continue
        p_self_corr = float(score_meta.get("p_self_corr_block", 1.0)) if score_meta else 1.0
        d1_ready = bool(score_meta.get("d1_ready", 0)) if score_meta else False
        detail_channel = "normal"
        detail_reason = ""
        if p_self_corr > max_self_corr_risk:
            allow, reason = should_allow_exploration_detail(
                score_meta,
                p_self_corr,
                max_self_corr_risk,
                max_exploration_p_self_corr,
                exploration_checks,
                exploration_check_budget,
            )
            if allow:
                detail_channel = "exploration"
                detail_reason = reason
            else:
                fde_allow, fde_reason = should_allow_fde_detail(score_meta, fde_checks, fde_check_budget, fde_enabled)
                if fde_allow:
                    detail_channel = "fde"
                    detail_reason = fde_reason
                else:
                    d1_allow, d1_reason = should_allow_d1_detail(
                        str(local_id),
                        score_meta,
                        p_self_corr,
                        d1_checks,
                        d1_check_budget,
                        d1_max_p_self_corr,
                        d1_exploration_enabled,
                    )
                    if d1_allow:
                        detail_channel = "d1"
                        detail_reason = d1_reason
                    else:
                        skipped.append((local_id, f"pre-submit self-corr gate: p_self_corr_block={p_self_corr:.3f} > {max_self_corr_risk:.2f}; {reason or fde_reason or d1_reason}"))
                        continue
        if require_d1_gate and not d1_ready and detail_channel == "normal":
            d1_allow, d1_reason = should_allow_d1_detail(
                str(local_id),
                score_meta,
                p_self_corr,
                d1_checks,
                d1_check_budget,
                d1_max_p_self_corr,
                d1_exploration_enabled,
            )
            if d1_allow:
                detail_channel = "d1"
                detail_reason = d1_reason
            else:
                skipped.append((local_id, f"pre-submit D1 gate: scorer did not mark candidate as d1_ready; {d1_reason}"))
                continue
        wq_id = worldquant_alpha_id(row["raw_json"])
        if not wq_id:
            skipped.append((local_id, "missing worldquant alpha id"))
            continue
        if already_recorded(wq_id):
            skipped.append((local_id, "already recorded"))
            continue
        if cooldown_active(state, wq_id, self_corr_cooldown_hours):
            skipped.append((local_id, f"self-correlation cooldown active ({self_corr_cooldown_hours}h)"))
            continue
        if detail_checks >= max_detail_checks:
            skipped.append((local_id, "detail check budget exhausted"))
            continue
        if detail_channel == "exploration" and exploration_checks >= exploration_check_budget:
            skipped.append((local_id, "self-correlation exploration budget exhausted"))
            continue
        if detail_channel == "fde" and fde_checks >= fde_check_budget:
            skipped.append((local_id, "FDE detail budget exhausted"))
            continue
        if detail_channel == "d1" and d1_checks >= d1_check_budget:
            skipped.append((local_id, "D1 exploration detail budget exhausted"))
            continue
        if detail_channel == "exploration":
            exploration_checks += 1
            skipped.append((local_id, f"exploration detail check despite p_self_corr={p_self_corr:.3f}: {detail_reason}"))
        elif detail_channel == "fde":
            fde_checks += 1
            state["fde_checks_this_round"] = fde_checks
            state["fde_total_checks"] = int(state.get("fde_total_checks", 0)) + 1
            skipped.append((local_id, f"FDE detail check for label collection despite p_self_corr={p_self_corr:.3f}: {detail_reason}"))
        elif detail_channel == "d1":
            d1_checks += 1
            state["d1_checks_this_round"] = d1_checks
            state["d1_total_checks"] = int(state.get("d1_total_checks", 0)) + 1
            skipped.append((local_id, f"D1 exploration detail check for label collection: {detail_reason}"))
        detail_checks += 1
        detail = client.get_alpha(wq_id)
        if str(detail.get("status", "")).upper() == "SUBMITTED" or detail.get("dateSubmitted"):
            record({"timestamp": datetime.now(timezone.utc).isoformat(), "alpha_id": wq_id, "local_id": local_id, "status": "already_submitted"})
            skipped.append((local_id, "already submitted on WQ"))
            continue
        checks = ((detail.get("is") or {}).get("checks") or [])
        unsafe_checks = []
        for check in checks:
            if not isinstance(check, dict):
                continue
            name = str(check.get("name") or "").upper()
            result = str(check.get("result") or "").upper()
            if name in {"SELF_CORRELATION", "MATCHES_COMPETITION"} and result != "PASS":
                unsafe_checks.append(f"{name}:{result}")
            elif result in {"FAIL", "ERROR"}:
                unsafe_checks.append(f"{name}:{result}")
        detail_result = detail_result_from_unsafe(unsafe_checks)
        if detail_channel == "fde":
            if detail_result == "clear":
                state["fde_clear_count"] = int(state.get("fde_clear_count", 0)) + 1
            else:
                state["fde_blocked_count"] = int(state.get("fde_blocked_count", 0)) + 1
        if detail_channel == "d1":
            save_d1_truth_entry(str(local_id), str(local_id).split("_", 1)[0] + "_", p_self_corr, detail_result)
        if unsafe_checks:
            if any(item.startswith("SELF_CORRELATION:") for item in unsafe_checks):
                mark_cooldown(state, wq_id)
            skipped.append((local_id, "unsafe checks: " + ", ".join(unsafe_checks)))
            continue
        if detail_channel in {"exploration", "fde", "d1"} and (p_self_corr > max_self_corr_risk or (require_d1_gate and not d1_ready)):
            skipped.append((local_id, f"{detail_channel} detail check clear; not submitted because normal submit gates still fail"))
            continue
        if args.dry_run:
            submitted.append({"local_id": local_id, "alpha_id": wq_id, "dry_run": True})
            continue
        response = client.submit_alpha(wq_id)
        increment_daily_submit_count(state)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "local_id": local_id,
            "alpha_id": wq_id,
            "sharpe": row["sharpe"],
            "fitness": row["fitness"],
            "turnover": row["turnover"],
            "returns": row["returns"],
            "expression": row["expression"],
            "response": response,
            "status": "submitted",
        }
        record(payload)
        submitted.append(payload)
        send_telegram(
            "[WQ SUBMITTED] Alpha auto-submitted\n"
            f"local={local_id}\nwq={wq_id}\nsharpe={row['sharpe']} fitness={row['fitness']} turnover={row['turnover']}\n"
            f"expr={row['expression'][:220]}"
        )
    print(json.dumps({"submitted": submitted, "skipped": skipped, "thresholds": {"min_sharpe": min_sharpe, "min_fitness": min_fitness, "max_turnover": max_turnover, "max_p_self_corr": max_self_corr_risk, "exploration_checks": exploration_check_budget, "max_exploration_p_self_corr": max_exploration_p_self_corr, "fde_checks": fde_check_budget, "d1_exploration_checks": d1_check_budget, "d1_max_p_self_corr": d1_max_p_self_corr, "daily_submit_limit": daily_limit, "daily_submit_count": daily_submit_count(state)}}, ensure_ascii=False, default=str))
    save_state(state)
    conn.close()


if __name__ == "__main__":
    main()
