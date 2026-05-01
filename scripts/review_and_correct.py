#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.sqlite_utils import active_backtest_rows, exact_prefix_clause, exact_prefix_param  # noqa: E402

STATE_PATH = ROOT / "state" / "review_correct_state.json"
REPORT_PATH = ROOT / "reports" / "review_correct_latest.md"
LOG_PATH = ROOT / "logs" / "review_correct.log"
DB_PATH = ROOT / "data" / "backtests.sqlite3"
AUTONOMY_ENV = ROOT / "secrets" / "autonomy.env"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
ADAPTIVE_STATE_PATH = ROOT / "state" / "adaptive_sampler_state.json"
PY = ROOT / ".venv" / "bin" / "python"

STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

FAMILIES = ["decor_", "refit_", "sparse_", "simp_", "tmpl_", "repairto_", "repairsc_", "repairsc2_", "repairsc3_", "arm_", "multi_", "super_", "supersc_"]
MULTI_STATE_PATH = ROOT / "state" / "multi_dataset_state.json"
SUPER_STATE_PATH = ROOT / "state" / "superalpha_state.json"
SUPER_REPAIR_STATE_PATH = ROOT / "state" / "super_repair_state.json"
ML_SCORER_STATE_PATH = ROOT / "state" / "ml_candidate_scorer_state.json"
SELF_CORR_TRUTH_PATH = ROOT / "state" / "self_corr_truth_table.json"
MULTI_D1_PANEL_PATH = ROOT / "state" / "multi_d1_panel.json"
D1_GENERATOR_STATE_PATH = ROOT / "state" / "d1_generator_state.json"
LINEAGE_OCCUPANCY_STATE_PATH = ROOT / "state" / "lineage_occupancy.json"
REPAIR_CANDIDATES_STATE_PATH = ROOT / "state" / "repair_candidates_state.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_env(path: Path) -> dict[str, str]:
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


def save_env(path: Path, updates: dict[str, str]) -> None:
    existing_lines = []
    existing = {}
    if path.exists():
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        for line in existing_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            existing[k.strip()] = v.strip()
    existing.update(updates)

    rendered = []
    seen = set()
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            rendered.append(line)
            continue
        k, _ = stripped.split("=", 1)
        key = k.strip()
        if key in updates:
            rendered.append(f"{key}={existing[key]}")
            seen.add(key)
        else:
            rendered.append(line)
            seen.add(key)
    for key, value in updates.items():
        if key not in seen:
            rendered.append(f"{key}={value}")
    path.write_text("\n".join(rendered).rstrip() + "\n", encoding="utf-8")


def run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        out = (stdout + stderr).strip()
        msg = f"TIMEOUT after {timeout}s: {' '.join(cmd)}"
        return 124, (out + "\n" + msg).strip()
    out = ((p.stdout or "") + (p.stderr or "")).strip()
    return p.returncode, out


def is_nonfatal_step_failure(label: str, code: int) -> bool:
    # Monitoring is observational. If it hangs or times out, record it but do
    # not waste the whole 2h correction cycle; downstream generate/import and
    # autosubmit steps can still make progress using current auth + DB state.
    return label == "strategy-monitor" and code != 0


def step_launches_backtest(step: dict) -> bool:
    cmd = [str(x) for x in step.get("cmd", [])]
    if "-m" in cmd and "alpha_factory.cli" in cmd and "run" in cmd:
        return True
    return any(x.endswith("scripts/run_decorrelate_pipeline.sh") or x.endswith("scripts/run_refit_pipeline.sh") for x in cmd)


def fetch_auth() -> dict:
    code, out = run([str(PY), "scripts/auth_status.py", "--json"], timeout=120)
    try:
        data = json.loads(out.splitlines()[-1])
    except Exception:
        data = {"ok": False, "error": out[-500:]}
    data["returncode"] = code
    return data


def q1(conn: sqlite3.Connection, sql: str, params: tuple = ()):
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else 0


def family_metrics(conn: sqlite3.Connection, prefix: str) -> dict:
    p = exact_prefix_param(prefix)
    return {
        "pending": q1(conn, f"select count(*) from alpha_tasks where {exact_prefix_clause('id')} and status='PENDING'", (p,)),
        "running": q1(conn, f"select count(*) from alpha_tasks where {exact_prefix_clause('id')} and status='RUNNING'", (p,)),
        "auth_error": q1(conn, f"select count(*) from alpha_tasks where {exact_prefix_clause('id')} and status='AUTH_ERROR'", (p,)),
        "error": q1(conn, f"select count(*) from alpha_tasks where {exact_prefix_clause('id')} and status='ERROR'", (p,)),
        "complete": q1(conn, f"select count(*) from backtest_results where {exact_prefix_clause('alpha_id')}", (p,)),
        "pass_count": q1(conn, f"select count(*) from backtest_results where {exact_prefix_clause('alpha_id')} and sharpe>=1.6 and fitness>=1.0 and turnover<=0.45", (p,)),
        "strong_pass_count": q1(conn, f"select count(*) from backtest_results where {exact_prefix_clause('alpha_id')} and sharpe>=1.8 and fitness>=1.15 and turnover<=0.30", (p,)),
        "avg_sharpe": q1(conn, f"select round(coalesce(avg(sharpe),0),3) from (select sharpe from backtest_results where {exact_prefix_clause('alpha_id')} and sharpe is not null order by created_at desc limit 30)", (p,)),
        "avg_fitness": q1(conn, f"select round(coalesce(avg(fitness),0),3) from (select fitness from backtest_results where {exact_prefix_clause('alpha_id')} and fitness is not null order by created_at desc limit 30)", (p,)),
        "avg_turnover": q1(conn, f"select round(coalesce(avg(turnover),0),3) from (select turnover from backtest_results where {exact_prefix_clause('alpha_id')} and turnover is not null order by created_at desc limit 30)", (p,)),
        "pending_self_corr": q1(conn, f"select count(*) from backtest_results where {exact_prefix_clause('alpha_id')} and coalesce(fail_reasons,'') like '%SELF_CORRELATION%'", (p,)),
    }


def parse_blockers(log_path: Path, max_lines: int = 400) -> dict:
    counts = Counter()
    examples: list[str] = []
    if not log_path.exists():
        return {"counts": {}, "examples": []}
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    for line in lines:
        if '"skipped":' not in line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        for item in payload.get("skipped", []):
            if len(item) < 2:
                continue
            reason = item[1]
            if "self-correlation cooldown active" in reason:
                key = "self_corr_cooldown"
            elif "SELF_CORRELATION:PENDING" in reason:
                key = "self_corr_pending"
            elif "detail check budget exhausted" in reason:
                key = "detail_budget_exhausted"
            elif "already recorded" in reason:
                key = "already_recorded"
            else:
                key = reason
            counts[key] += 1
            if len(examples) < 8:
                examples.append(f"{item[0]} :: {reason}")
    return {"counts": dict(counts), "examples": examples}


def adaptive_sampler_summary(path: Path = ADAPTIVE_STATE_PATH) -> dict:
    state = load_json(path)
    arms = state.get("arms") or {}
    if not arms:
        return {"enabled": False, "reason": "no adaptive sampler state yet"}
    status_counts = Counter(str(a.get("status", "active")) for a in arms.values())
    family_counts = Counter(str(a.get("function_family", "unknown")) for a in arms.values() if not str(a.get("status", "active")).startswith("paused"))
    promoted = []
    near_miss = []
    for key, arm in arms.items():
        stats = arm.get("stats") or {}
        item = {
            "arm_key": key,
            "dataset_id": arm.get("dataset_id"),
            "field_id": arm.get("field_id"),
            "function_family": arm.get("function_family"),
            "status": arm.get("status"),
            "reward": stats.get("reward"),
            "tested": stats.get("tested", 0),
            "pass": stats.get("pass", 0),
            "near_miss": stats.get("near_miss", 0),
            "avg_fitness": stats.get("avg_fitness", 0),
        }
        if arm.get("status") == "promoted":
            promoted.append(item)
        elif arm.get("status") == "active_near_miss":
            near_miss.append(item)
    promoted.sort(key=lambda x: float(x.get("reward") or 0), reverse=True)
    near_miss.sort(key=lambda x: float(x.get("reward") or 0), reverse=True)
    history = state.get("history") or []
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "arms_total": len(arms),
        "status_counts": dict(status_counts),
        "active_function_families": dict(family_counts),
        "promoted": promoted[:8],
        "near_miss": near_miss[:8],
        "last_event": history[-1] if history else None,
    }


def multi_dataset_summary(path: Path = MULTI_STATE_PATH) -> dict:
    state = load_json(path)
    arms = state.get("arms") or {}
    history = state.get("history") or []
    if not arms:
        return {"enabled": False, "reason": "no multi dataset state yet"}
    status_counts = Counter(str(a.get("status", "active")) for a in arms.values())
    interaction_counts = Counter(str(a.get("interaction_type", "unknown")) for a in arms.values() if not str(a.get("status", "active")).startswith("paused"))
    top = []
    for key, arm in arms.items():
        stats = arm.get("stats") or {}
        if arm.get("status") in {"promoted", "active_near_miss"}:
            top.append({
                "arm_key": key,
                "datasets": [arm.get("primary_dataset"), arm.get("secondary_dataset")],
                "fields": [arm.get("primary_field"), arm.get("secondary_field")],
                "interaction_type": arm.get("interaction_type"),
                "status": arm.get("status"),
                "reward": stats.get("reward"),
                "tested": stats.get("tested", 0),
                "pass": stats.get("pass", 0),
                "near_miss": stats.get("near_miss", 0),
            })
    top.sort(key=lambda x: float(x.get("reward") or 0), reverse=True)
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "arms_total": len(arms),
        "status_counts": dict(status_counts),
        "active_interactions": dict(interaction_counts),
        "top": top[:8],
        "last_event": history[-1] if history else None,
    }


def superalpha_summary(path: Path = SUPER_STATE_PATH) -> dict:
    state = load_json(path)
    history = state.get("history") or []
    built = state.get("built_from") or {}
    if not state:
        return {"enabled": False, "reason": "no superalpha state yet"}
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "built_total": len(built),
        "last_event": history[-1] if history else None,
    }


def super_repair_summary(path: Path = SUPER_REPAIR_STATE_PATH) -> dict:
    state = load_json(path)
    history = state.get("history") or []
    seeded = state.get("seeded_from") or {}
    if not state:
        return {"enabled": False, "reason": "no super repair state yet"}
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "seeded_total": len(seeded),
        "last_event": history[-1] if history else None,
    }


def ml_scorer_summary(path: Path = ML_SCORER_STATE_PATH) -> dict:
    state = load_json(path)
    if not state:
        return {"enabled": False, "reason": "no ML scorer state yet"}
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "model": state.get("model"),
        "summary": state.get("summary", {}),
        "top_candidates": (state.get("top_candidates") or [])[:8],
    }


def lineage_occupancy_summary(path: Path = LINEAGE_OCCUPANCY_STATE_PATH) -> dict:
    state = load_json(path)
    if not state:
        return {"enabled": False, "reason": "no lineage occupancy state yet"}
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "submitted_library_rows": state.get("submitted_library_rows"),
        "top": (state.get("ranked_lineages") or [])[:8],
    }


def self_corr_truth_summary(path: Path = SELF_CORR_TRUTH_PATH) -> dict:
    state = load_json(path)
    if not state:
        return {"enabled": False, "reason": "no self-correlation truth table yet"}
    return {
        "enabled": True,
        "generated_at": state.get("generated_at"),
        "summary": state.get("summary", {}),
    }


def multi_d1_panel_summary(path: Path = MULTI_D1_PANEL_PATH) -> dict:
    state = load_json(path)
    if not state:
        return {"enabled": False, "reason": "no multi D1 panel yet"}
    ff = state.get("function_family") or {}
    ranked = sorted(ff.items(), key=lambda kv: (kv[1].get("d1_ready_rate", 0), kv[1].get("submit_clear_rate", 0), kv[1].get("pass", 0)), reverse=True)
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "summary": state.get("summary", {}),
        "top_function_families": [{"function_family": k, **v} for k, v in ranked[:8]],
    }


def d1_generator_summary(path: Path = D1_GENERATOR_STATE_PATH) -> dict:
    state = load_json(path)
    if not state:
        return {"enabled": False, "reason": "no D1 generator state yet"}
    history = state.get("history") or []
    return {
        "enabled": True,
        "updated_at": state.get("updated_at"),
        "candidate_total": len(state.get("candidates") or {}),
        "last_event": history[-1] if history else None,
    }


FAMILY_PAUSE_SCORE_THRESHOLD = 9.0
FAMILY_STALE_GENERATE_THRESHOLD = 2
FAMILY_HIGH_COLLISION_THRESHOLD = 0.72
FAMILY_HIGH_SELF_CORR_THRESHOLD = 0.68
FAMILY_D1_READY_FLOOR = 0.02


def family_pause_summary(
    metrics: dict[str, dict],
    ml_summary: dict | None = None,
    latest_state: dict | None = None,
) -> dict[str, dict]:
    summary_map: dict[str, dict] = {}
    ml_families = (((ml_summary or {}).get("summary") or {}).get("families") or {})
    top_candidates = (ml_summary or {}).get("top_candidates") or []
    repair_event = None
    previous_watchdog = (latest_state or {}).get("family_watchdog") or {}
    repair_state = load_json(REPAIR_CANDIDATES_STATE_PATH)
    repair_history = repair_state.get("history") or []
    if repair_history:
        repair_event = repair_history[-1]
    actions = (latest_state or {}).get("actions") or []
    for action in actions:
        if action.get("label") == "repair-generate":
            tail = action.get("tail")
            if isinstance(tail, str):
                try:
                    repair_event = json.loads(tail)
                except Exception:
                    pass
            break

    top_by_family: dict[str, list[dict]] = {}
    for row in top_candidates:
        fam = str(row.get("family") or "")
        if not fam:
            continue
        top_by_family.setdefault(fam, []).append(row)

    for family in FAMILIES:
        base = metrics.get(family, {})
        fam_rows = top_by_family.get(family, [])
        ml_count = int(ml_families.get(family, 0) or 0)
        weak_or_higher = sum(1 for row in fam_rows if str(row.get("submitted_collision_level") or "") in {"weak", "medium", "high"})
        high_self_corr = sum(1 for row in fam_rows if float(row.get("p_self_corr_block", 0.0) or 0.0) >= FAMILY_HIGH_SELF_CORR_THRESHOLD)
        d1_ready = sum(1 for row in fam_rows if int(row.get("d1_ready", 0) or 0) == 1)
        sampled = max(1, len(fam_rows))
        collision_rate = weak_or_higher / sampled
        self_corr_rate = high_self_corr / sampled
        d1_rate = d1_ready / sampled
        stale_generate = 0
        score = 0.0
        reasons: list[str] = []
        previous_family_watch = previous_watchdog.get(family) or {}
        if previous_family_watch.get("paused"):
            score += max(float(previous_family_watch.get("score", 0.0) or 0.0), FAMILY_PAUSE_SCORE_THRESHOLD)
            reasons.extend(previous_family_watch.get("reasons") or ["previous watchdog pause still active"])
        elif float(previous_family_watch.get("score", 0.0) or 0.0) >= 5.0:
            score += min(float(previous_family_watch.get("score", 0.0) or 0.0), FAMILY_PAUSE_SCORE_THRESHOLD - 0.1)
            reasons.extend(previous_family_watch.get("reasons") or ["previous watchdog warning still active"])
        if family == "repairsc_" and repair_event:
            submitted_status = repair_event.get("submitted_filter_status") or {}
            if int(repair_event.get("generated", 0) or 0) == 0 and int(submitted_status.get("after", 0) or 0) == 0:
                stale_generate = 1
            if int(submitted_status.get("before", 0) or 0) > 0 and int(submitted_status.get("dropped", 0) or 0) == int(submitted_status.get("before", 0) or 0):
                stale_generate += 1
            if int(repair_event.get("added", 0) or 0) == 0:
                score += 4.0
                reasons.append("repair branch produced zero newly written candidates")
        if ml_count >= 8 and collision_rate >= FAMILY_HIGH_COLLISION_THRESHOLD:
            score += 5.0
            reasons.append(f"submitted-collision-rate={collision_rate:.2f}")
        if ml_count >= 8 and self_corr_rate >= FAMILY_HIGH_SELF_CORR_THRESHOLD:
            score += 4.0
            reasons.append(f"high-p_self_corr-rate={self_corr_rate:.2f}")
        if ml_count >= 8 and d1_rate <= FAMILY_D1_READY_FLOOR:
            score += 2.5
            reasons.append(f"d1_ready_rate={d1_rate:.2f}")
        if base.get("pending_self_corr", 0) >= 20:
            score += 1.5
            reasons.append(f"pending_self_corr={base.get('pending_self_corr', 0)}")
        if stale_generate >= FAMILY_STALE_GENERATE_THRESHOLD:
            score += 5.5
            reasons.append("recent generation fully filtered pre-write")
        paused = score >= FAMILY_PAUSE_SCORE_THRESHOLD
        summary_map[family] = {
            "family": family,
            "score": round(score, 2),
            "paused": paused,
            "reasons": reasons,
            "ml_count": ml_count,
            "collision_rate": round(collision_rate, 4),
            "high_self_corr_rate": round(self_corr_rate, 4),
            "d1_ready_rate": round(d1_rate, 4),
            "pending_self_corr": int(base.get("pending_self_corr", 0) or 0),
            "stale_generate_signals": stale_generate,
        }
    return summary_map

def choose_focus(
    metrics: dict[str, dict],
    ml_summary: dict | None = None,
    multi_panel: dict | None = None,
    latest_state: dict | None = None,
) -> tuple[str, list[str], dict[str, dict]]:
    reasons: list[str] = []
    score = {}
    pause_summary = family_pause_summary(metrics, ml_summary=ml_summary, latest_state=latest_state)
    for family, m in metrics.items():
        s = 0.0
        s += m["strong_pass_count"] * 5
        s += m["pass_count"] * 2
        s += float(m["avg_fitness"] or 0) * 8
        s += float(m["avg_sharpe"] or 0) * 3
        s -= float(m["avg_turnover"] or 0) * 2
        s -= m["auth_error"] * 0.5
        s -= m["error"] * 0.25
        if family == "decor_":
            s += 3
        if family == "multi_":
            s += 2
        if family == "sparse_" and (m["avg_fitness"] or 0) < 0.9:
            s -= 4
        if family == "tmpl_":
            s -= 3
        pause = pause_summary.get(family) or {}
        if pause.get("paused"):
            s -= 18.0
            reasons.append(f"pause/downweight {family}: {'; '.join(pause.get('reasons') or ['family watchdog triggered'])}")
        elif pause.get("score", 0) >= 5.0:
            s -= 18.0
            reasons.append(f"downweight {family}: {'; '.join(pause.get('reasons') or ['family watchdog warning'])}")
        score[family] = s

    ml_d1_ready = ((ml_summary or {}).get("summary") or {}).get("d1_ready", 0)
    top_multi = (multi_panel or {}).get("top_function_families") or []
    best_multi_d1 = max((float(x.get("d1_ready_rate", 0) or 0.0) for x in top_multi), default=0.0)
    best_multi_clear = max((float(x.get("submit_clear_rate", 0) or 0.0) for x in top_multi), default=0.0)
    if ml_d1_ready == 0:
        score["multi_"] = score.get("multi_", 0.0) + 8.0
        score["super_"] = score.get("super_", 0.0) - 2.0
        reasons.append("No D1-ready candidates yet; bias focus toward multi_ as low-correlation supply line")
    if best_multi_d1 > 0 or best_multi_clear > 0:
        score["multi_"] = score.get("multi_", 0.0) + 10.0 + 12.0 * best_multi_d1 + 8.0 * best_multi_clear
        reasons.append(f"multi_ bonus from panel: best_d1_ready_rate={round(best_multi_d1,4)} best_submit_clear_rate={round(best_multi_clear,4)}")

    focus = max(score, key=score.get)
    if (pause_summary.get(focus) or {}).get("paused"):
        fallback = [f for f, _ in sorted(score.items(), key=lambda kv: kv[1], reverse=True) if not (pause_summary.get(f) or {}).get("paused")]
        if fallback:
            reasons.append(f"focus fallback from paused family {focus} to {fallback[0]}")
            focus = fallback[0]
    reasons.append(f"focus={focus} score={round(score[focus], 2)}")
    reasons.append("ranked=" + ", ".join(f"{k}:{round(v,2)}" for k, v in sorted(score.items(), key=lambda kv: kv[1], reverse=True)))
    return focus, reasons, pause_summary


def decide_param_updates(focus: str, metrics: dict[str, dict], blockers: dict, current_env: dict[str, str]) -> tuple[dict[str, str], list[str], list[str]]:
    updates: dict[str, str] = {}
    notes: list[str] = []
    optimizations: list[str] = []
    blocker_counts = blockers.get("counts", {})

    if focus == "decor_":
        updates.update({
            "AUTO_SUBMIT_MIN_SHARPE": "1.60",
            "AUTO_SUBMIT_MIN_FITNESS": "1.00",
            "AUTO_SUBMIT_MAX_TURNOVER": "0.45",
            "AUTO_SUBMIT_MAX_DETAIL_CHECKS": "8",
        })
        notes.append("decor_ strongest: keep submit thresholds standard and allow slightly more detail checks")
    elif focus == "refit_":
        updates.update({
            "AUTO_SUBMIT_MIN_SHARPE": "1.60",
            "AUTO_SUBMIT_MIN_FITNESS": "1.00",
            "AUTO_SUBMIT_MAX_TURNOVER": "0.40",
            "AUTO_SUBMIT_MAX_DETAIL_CHECKS": "6",
        })
        notes.append("refit_ favored: keep standard Sharpe floor and tighter turnover")
    elif focus == "sparse_":
        updates.update({
            "AUTO_SUBMIT_MIN_SHARPE": "1.70",
            "AUTO_SUBMIT_MIN_FITNESS": "1.05",
            "AUTO_SUBMIT_MAX_TURNOVER": "0.35",
            "AUTO_SUBMIT_MAX_DETAIL_CHECKS": "4",
        })
        notes.append("sparse_ weaker historically: require stricter submit quality before spending checks")
    else:
        updates.update({
            "AUTO_SUBMIT_MIN_SHARPE": "1.65",
            "AUTO_SUBMIT_MIN_FITNESS": "1.05",
            "AUTO_SUBMIT_MAX_TURNOVER": "0.40",
            "AUTO_SUBMIT_MAX_DETAIL_CHECKS": "4",
        })
        notes.append("non-decor exploratory mode: use stricter thresholds")

    cooldown = "8" if metrics["decor_"]["pending_self_corr"] > 20 else "6"
    if blocker_counts.get("self_corr_pending", 0) >= 6 or blocker_counts.get("detail_budget_exhausted", 0) >= 4:
        cooldown = "8"
        notes.append("recent autosubmit blockers dominated by pending self-correlation/detail budget; extend cooldown to 8h")
        optimizations.append("Raised AUTO_SUBMIT_SELF_CORR_COOLDOWN_HOURS to 8 to reduce wasteful resubmission pressure on repeatedly blocked decor candidates")

    detail_checks = updates["AUTO_SUBMIT_MAX_DETAIL_CHECKS"]
    if blocker_counts.get("detail_budget_exhausted", 0) >= 4:
        detail_checks = "12" if focus == "decor_" else str(max(int(detail_checks), 8))
        notes.append("detail-check exhaustion detected; increase budget to probe more top decor candidates each cycle")
        optimizations.append("Raised AUTO_SUBMIT_MAX_DETAIL_CHECKS to reduce repeated starvation from detail-check budget exhaustion")
    elif blocker_counts.get("self_corr_pending", 0) >= 8 and focus == "decor_":
        detail_checks = "6"
        notes.append("self-correlation pending dominates; trim detail checks slightly to avoid overspending during blocked windows")
        optimizations.append("Reduced AUTO_SUBMIT_MAX_DETAIL_CHECKS during self-correlation-pending heavy periods to conserve review budget")
    updates["AUTO_SUBMIT_MAX_DETAIL_CHECKS"] = detail_checks
    updates["AUTO_SUBMIT_SELF_CORR_COOLDOWN_HOURS"] = cooldown

    def current_float(key: str, default: float) -> float:
        try:
            return float(current_env.get(key, default))
        except Exception:
            return default

    updates["AUTO_SUBMIT_MIN_SHARPE"] = f"{max(float(updates['AUTO_SUBMIT_MIN_SHARPE']), current_float('AUTO_SUBMIT_MIN_SHARPE', 1.60), 1.60):.2f}"
    updates["AUTO_SUBMIT_MIN_FITNESS"] = f"{max(float(updates['AUTO_SUBMIT_MIN_FITNESS']), current_float('AUTO_SUBMIT_MIN_FITNESS', 1.00), 1.00):.2f}"
    updates["AUTO_SUBMIT_MAX_TURNOVER"] = f"{min(float(updates['AUTO_SUBMIT_MAX_TURNOVER']), current_float('AUTO_SUBMIT_MAX_TURNOVER', 0.45), 0.45):.2f}"
    updates["AUTO_SUBMIT_MAX_P_SELF_CORR"] = "0.20"
    updates["AUTO_SUBMIT_REQUIRE_D1_READY"] = "1"

    for key, value in updates.items():
        if current_env.get(key) != value and not any(key in s for s in optimizations):
            optimizations.append(f"Adjusted {key} from {current_env.get(key)} to {value}")
    if not optimizations:
        optimizations.append("No parameter change was needed; current autonomy settings still match observed blockers and family strength")
    return updates, notes, optimizations


def persist(payload: dict) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# WorldQuant Review & Correct — {payload['timestamp_utc']}",
        "",
        f"- Stage: `{payload.get('stage')}`",
        f"- Auth: `{payload['auth'].get('ok')}` user=`{payload['auth'].get('user')}` expiry_seconds=`{payload['auth'].get('expiry_seconds')}`",
        f"- Focus family: `{payload['focus']}`",
        "",
        "## Focus rationale",
        *[f"- {x}" for x in payload.get("focus_notes", [])],
        "",
        "## Parameter updates",
        *[f"- `{k}={v}`" for k, v in payload.get("param_updates", {}).items()],
        *[f"- note: {x}" for x in payload.get("param_notes", [])],
        "",
        "## Optimization actions this cycle",
        *[f"- {x}" for x in payload.get("optimization_actions", [])],
        "",
        "## Adaptive dataset/field/function sampler",
    ]
    sampler = payload.get("adaptive_sampler", {})
    if sampler.get("enabled"):
        lines.extend([
            f"- updated_at: `{sampler.get('updated_at')}`",
            f"- arms_total: `{sampler.get('arms_total')}`",
            f"- status_counts: `{sampler.get('status_counts')}`",
            f"- active_function_families: `{sampler.get('active_function_families')}`",
        ])
        last_event = sampler.get("last_event") or {}
        if last_event:
            lines.append(f"- last_event added={len(last_event.get('added', []))} selected_arms={len(last_event.get('selected_arms', []))}")
        for item in sampler.get("promoted", [])[:5]:
            lines.append(f"- promoted: {item['dataset_id']} / {item['field_id']} / {item['function_family']} reward={item.get('reward')} pass={item.get('pass')} tested={item.get('tested')}")
        for item in sampler.get("near_miss", [])[:5]:
            lines.append(f"- near_miss: {item['dataset_id']} / {item['field_id']} / {item['function_family']} reward={item.get('reward')} near={item.get('near_miss')} tested={item.get('tested')}")
    else:
        lines.append(f"- not active yet: {sampler.get('reason', 'unknown')}")

    lines.extend(["", "## Multi Data Set Alpha"])
    multi = payload.get("multi_dataset", {})
    if multi.get("enabled"):
        lines.extend([
            f"- updated_at: `{multi.get('updated_at')}`",
            f"- arms_total: `{multi.get('arms_total')}`",
            f"- status_counts: `{multi.get('status_counts')}`",
            f"- active_interactions: `{multi.get('active_interactions')}`",
        ])
        last_event = multi.get("last_event") or {}
        if last_event:
            lines.append(f"- last_event added={len(last_event.get('added', []))} selected_arms={len(last_event.get('selected_arms', []))}")
        for item in multi.get("top", [])[:5]:
            lines.append(f"- top: {item.get('datasets')} / {item.get('fields')} / {item.get('interaction_type')} status={item.get('status')} reward={item.get('reward')} pass={item.get('pass')} tested={item.get('tested')}")
    else:
        lines.append(f"- not active yet: {multi.get('reason', 'unknown')}")

    lines.extend(["", "## Multi Data Set D1 panel"])
    panel = payload.get("multi_d1_panel", {})
    if panel.get("enabled"):
        lines.extend([
            f"- updated_at: `{panel.get('updated_at')}`",
            f"- summary: `{panel.get('summary')}`",
        ])
        for item in panel.get("top_function_families", [])[:6]:
            lines.append(f"- top_ff: {item.get('function_family')} tested={item.get('tested')} pass={item.get('pass')} strong={item.get('strong')} d1_ready_rate={item.get('d1_ready_rate')} self_corr_block_rate={item.get('self_corr_block_rate')} submit_clear_rate={item.get('submit_clear_rate')}")
    else:
        lines.append(f"- not active yet: {panel.get('reason', 'unknown')}")

    lines.extend(["", "## Lineage occupancy hints"])
    lineage = payload.get("lineage_occupancy", {})
    if lineage.get("enabled"):
        lines.extend([
            f"- updated_at: `{lineage.get('updated_at')}`",
            f"- submitted_library_rows: `{lineage.get('submitted_library_rows')}`",
        ])
        for item in lineage.get("top", [])[:8]:
            lines.append(f"- explore: {item.get('lineage')} weight={item.get('weight')} submitted_count={item.get('submitted_count')}")
    else:
        lines.append(f"- not active yet: {lineage.get('reason', 'unknown')}")

    lines.extend(["", "## D1 generator"])
    d1 = payload.get("d1_generator", {})
    if d1.get("enabled"):
        lines.extend([
            f"- updated_at: `{d1.get('updated_at')}`",
            f"- candidate_total: `{d1.get('candidate_total')}`",
        ])
        last_event = d1.get("last_event") or {}
        if last_event:
            lines.append(f"- last_event added={len(last_event.get('added', []))} selected_pairs={len(last_event.get('selected_pairs', []))}")
    else:
        lines.append(f"- not active yet: {d1.get('reason', 'unknown')}")

    lines.extend(["", "## SuperAlpha"])
    superalpha = payload.get("superalpha", {})
    if superalpha.get("enabled"):
        lines.extend([
            f"- updated_at: `{superalpha.get('updated_at')}`",
            f"- built_total: `{superalpha.get('built_total')}`",
        ])
        last_event = superalpha.get("last_event") or {}
        if last_event:
            lines.append(f"- last_event added={len(last_event.get('added', []))} selected_pairs={len(last_event.get('selected_pairs', []))}")
    else:
        lines.append(f"- not active yet: {superalpha.get('reason', 'unknown')}")

    lines.extend(["", "## SuperAlpha self-correlation repair"])
    super_repair = payload.get("super_repair", {})
    if super_repair.get("enabled"):
        lines.extend([
            f"- updated_at: `{super_repair.get('updated_at')}`",
            f"- seeded_total: `{super_repair.get('seeded_total')}`",
        ])
        last_event = super_repair.get("last_event") or {}
        if last_event:
            lines.append(f"- last_event added={last_event.get('added')} generated={last_event.get('generated')} sources={last_event.get('sources')} repair_types={last_event.get('repair_types')}")
    else:
        lines.append(f"- not active yet: {super_repair.get('reason', 'unknown')}")

    watchdog = payload.get("family_watchdog", {})
    if watchdog:
        lines.extend(["", "## Family watchdog"])
        ranked_watchdog = sorted(watchdog.values(), key=lambda x: (x.get("paused", False), x.get("score", 0)), reverse=True)
        for item in ranked_watchdog[:8]:
            state = "paused" if item.get("paused") else "watch"
            reason = "; ".join(item.get("reasons") or ["no trigger"])
            lines.append(
                f"- {item.get('family')} state={state} score={item.get('score')} collision_rate={item.get('collision_rate')} high_self_corr_rate={item.get('high_self_corr_rate')} d1_ready_rate={item.get('d1_ready_rate')} :: {reason}"
            )

    lines.extend(["", "## Machine Learning Alpha scorer"])
    ml = payload.get("ml_scorer", {})
    if ml.get("enabled"):
        lines.extend([
            f"- updated_at: `{ml.get('updated_at')}`",
            f"- model: `{ml.get('model')}`",
            f"- summary: `{ml.get('summary')}`",
        ])
        for item in ml.get("top_candidates", [])[:5]:
            lines.append(f"- top: {item.get('alpha_id')} family={item.get('family')} submit_score={item.get('submit_score')} p_pass={item.get('p_pass')} p_self_corr={item.get('p_self_corr_block')}")
    else:
        lines.append(f"- not active yet: {ml.get('reason', 'unknown')}")
    lines.extend(["", "## Self-correlation truth table"])
    truth = payload.get("self_corr_truth", {})
    if truth.get("enabled"):
        summary = truth.get("summary") or {}
        lines.extend([
            f"- generated_at: `{truth.get('generated_at')}`",
            f"- status_counts: `{summary.get('status_counts', {})}`",
            f"- known_clear_rate: `{summary.get('known_clear_rate')}` known_evidence_count=`{summary.get('known_evidence_count')}`",
            f"- lineage_theme_counts: `{summary.get('lineage_theme_counts', {})}`",
        ])
    else:
        lines.append(f"- not active yet: {truth.get('reason', 'unknown')}")
    lines.extend([
        "",
        "## Recent autosubmit blockers",
    ])
    blocker_counts = payload.get("blockers", {}).get("counts", {})
    if blocker_counts:
        for k, v in sorted(blocker_counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none detected")
    for ex in payload.get("blockers", {}).get("examples", [])[:6]:
        lines.append(f"- example: {ex}")
    lines.extend(["", "## Family metrics"])
    for family in FAMILIES:
        m = payload["metrics"][family]
        lines.append(f"- {family} pending={m['pending']} complete={m['complete']} pass={m['pass_count']} strong={m['strong_pass_count']} avgS={m['avg_sharpe']} avgF={m['avg_fitness']} avgT={m['avg_turnover']} auth_err={m['auth_error']} err={m['error']}")
    lines.extend(["", "## Actions run"])
    active_guard = payload.get("active_launch_guard") or {}
    if active_guard.get("active"):
        lines.extend([
            f"- active_launch_guard: active_rows={active_guard.get('count')} override={active_guard.get('override')} decision={active_guard.get('decision')}",
        ])
        for row in active_guard.get("examples", [])[:5]:
            lines.append(f"- active: {row.get('id')} status={row.get('status')} updated_at={row.get('updated_at')}")
    for action in payload.get("actions", []):
        if action.get("skipped"):
            lines.append(f"- {action['label']} skipped: {action.get('reason','')}")
        else:
            tail = action.get("tail", "").replace("\n", " ")[:350]
            marker = " nonfatal" if action.get("nonfatal") else ""
            lines.append(f"- {action['label']} rc={action['returncode']}{marker} :: {tail}")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "timestamp_utc": payload["timestamp_utc"],
            "stage": payload.get("stage"),
            "focus": payload["focus"],
            "param_updates": payload.get("param_updates", {}),
            "optimization_actions": payload.get("optimization_actions", []),
            "actions": [{k: v for k, v in a.items() if k in {"label", "returncode", "skipped", "reason", "nonfatal"}} for a in payload.get("actions", [])],
        }, ensure_ascii=False) + "\n")


def action_plan(focus: str, auth: dict, metrics: dict[str, dict], blockers: dict, ml_summary: dict | None = None, multi_panel: dict | None = None) -> list[dict]:
    plan: list[dict] = [{"label": "strategy-monitor", "cmd": [str(PY), "scripts/strategy_monitor.py"], "timeout": 900}]
    if not auth.get("ok"):
        plan.append({"label": "auth-down", "cmd": [], "timeout": 0, "skip_reason": "auth unavailable; only monitor/threshold tuning"})
        return plan
    blocker_counts = blockers.get("counts", {})
    repair_pending = (
        metrics.get("repairto_", {}).get("pending", 0)
        + metrics.get("repairsc_", {}).get("pending", 0)
        + metrics.get("repairsc2_", {}).get("pending", 0)
        + metrics.get("repairsc3_", {}).get("pending", 0)
    )
    self_corr_pressure = blocker_counts.get("self_corr_pending", 0) + blocker_counts.get("self_corr_cooldown", 0)
    detail_pressure = blocker_counts.get("detail_budget_exhausted", 0)
    submit_gated_value = (
        metrics.get("repairsc2_", {}).get("pass_count", 0)
        + metrics.get("repairsc3_", {}).get("pass_count", 0)
        + metrics.get("repairsc_", {}).get("strong_pass_count", 0)
    )
    ml_d1_ready = ((ml_summary or {}).get("summary") or {}).get("d1_ready", 0)
    top_multi = (multi_panel or {}).get("top_function_families") or []
    best_multi_d1 = max((float(x.get("d1_ready_rate", 0) or 0.0) for x in top_multi), default=0.0)
    best_multi_clear = max((float(x.get("submit_clear_rate", 0) or 0.0) for x in top_multi), default=0.0)

    repair_budget = 8 if self_corr_pressure >= 20 else 6
    if repair_pending >= 20:
        repair_budget = max(repair_budget, 10)
    if submit_gated_value >= 2:
        repair_budget = max(repair_budget, 12)
    adaptive_budget = 1 if repair_pending >= 20 else 2 if repair_pending >= 8 else 4
    if detail_pressure >= 10:
        adaptive_budget = max(1, adaptive_budget - 1)
    if submit_gated_value >= 2:
        adaptive_budget = 1

    d1_budget = 4 if ml_d1_ready == 0 else 2
    if best_multi_d1 > 0:
        d1_budget = max(d1_budget, 6)
    if best_multi_clear > 0:
        d1_budget = max(d1_budget, 8)

    multi_budget = 2 if repair_pending >= 20 else 4
    if metrics.get("multi_", {}).get("pass_count", 0) > 0 or metrics.get("multi_", {}).get("strong_pass_count", 0) > 0:
        multi_budget = max(multi_budget, 5)
    if best_multi_d1 > 0:
        multi_budget = max(multi_budget, 6)
    if best_multi_clear > 0:
        multi_budget = max(multi_budget, 7)
    if detail_pressure >= 10:
        multi_budget = max(1, multi_budget - 1)

    if best_multi_d1 > metrics.get("super_", {}).get("pass_count", 0) / max(1, metrics.get("super_", {}).get("complete", 1)):
        repair_budget = max(4, repair_budget - 2)
        multi_budget += 1
        d1_budget += 1

    fresh_budget = 8 if ml_d1_ready == 0 else 4
    if repair_pending >= 20 or submit_gated_value >= 2:
        fresh_budget = max(fresh_budget, 10)
    if best_multi_clear > 0 or best_multi_d1 > 0:
        fresh_budget = max(4, fresh_budget - 2)

    plan.append({"label": "fresh-supply-generate", "cmd": [str(PY), "scripts/fresh_supply_generator.py", "--max-add", str(fresh_budget), "--field-limit", "120"], "timeout": 300})
    plan.append({"label": "fresh-supply-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    if fresh_budget > 0:
        plan.append({"label": "fresh-supply-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "2", "--id-prefix", "fresh_"], "timeout": 1800})

    plan.append({"label": "repair-generate", "cmd": [str(PY), "scripts/repair_candidates.py"], "timeout": 300})
    plan.append({"label": "repair-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    repairsc2_limit = 0
    repairsc3_limit = 0
    if metrics.get("repairsc2_", {}).get("pending", 0) > 0 or submit_gated_value >= 2:
        repairsc2_limit = max(2, repair_budget // 3)
    if metrics.get("repairsc3_", {}).get("pending", 0) > 0 or metrics.get("repairsc2_", {}).get("strong_pass_count", 0) >= 1:
        repairsc3_limit = max(2, repair_budget // 4)
    turnover_limit = max(2, (repair_budget - repairsc2_limit - repairsc3_limit) // 2)
    selfcorr_limit = max(2, repair_budget - repairsc2_limit - repairsc3_limit - turnover_limit)
    plan.append({"label": "repair-turnover-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", str(turnover_limit), "--id-prefix", "repairto_"], "timeout": 1800})
    plan.append({"label": "repair-selfcorr-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", str(selfcorr_limit), "--id-prefix", "repairsc_"], "timeout": 1800})
    if repairsc2_limit > 0:
        plan.append({"label": "repair-submitgated-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", str(repairsc2_limit), "--id-prefix", "repairsc2_"], "timeout": 1800})
    if repairsc3_limit > 0:
        plan.append({"label": "repair-submitgated-v2-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", str(repairsc3_limit), "--id-prefix", "repairsc3_"], "timeout": 1800})
    plan.append({"label": "adaptive-sampler", "cmd": [str(PY), "scripts/adaptive_sampler.py", "--max-add", str(adaptive_budget), "--arm-limit", "6"], "timeout": 300})
    plan.append({"label": "adaptive-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    if adaptive_budget > 0:
        plan.append({"label": "adaptive-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "2", "--id-prefix", "arm_"], "timeout": 1800})

    plan.append({"label": "multi-d1-panel", "cmd": [str(PY), "scripts/multi_d1_panel.py"], "timeout": 180})
    plan.append({"label": "lineage-occupancy", "cmd": [str(PY), "scripts/lineage_occupancy.py"], "timeout": 180})
    plan.append({"label": "d1-generate", "cmd": [str(PY), "scripts/d1_generator.py", "--max-add", str(d1_budget), "--pair-limit", "4"], "timeout": 300})
    plan.append({"label": "d1-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    if d1_budget > 0:
        plan.append({"label": "d1-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "2", "--id-prefix", "d1_"], "timeout": 1800})

    plan.append({"label": "multi-generate", "cmd": [str(PY), "scripts/multi_dataset_generator.py", "--max-add", str(multi_budget), "--arm-limit", "6"], "timeout": 300})
    plan.append({"label": "multi-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    if multi_budget > 0:
        plan.append({"label": "multi-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "2", "--id-prefix", "multi_"], "timeout": 1800})

    super_budget = 2 if submit_gated_value >= 2 or self_corr_pressure >= 20 else 1
    if metrics.get("super_", {}).get("pass_count", 0) > 0:
        super_budget = max(super_budget, 3)
    plan.append({"label": "super-build", "cmd": [str(PY), "scripts/superalpha_builder.py", "--max-add", str(super_budget), "--pair-limit", "8"], "timeout": 300})
    plan.append({"label": "super-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
    if super_budget > 0:
        plan.append({"label": "super-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "1", "--id-prefix", "super_"], "timeout": 1800})
    supersc_budget = 3 if metrics.get("super_", {}).get("pending_self_corr", 0) > 0 or metrics.get("super_", {}).get("pass_count", 0) > 0 else 0
    if metrics.get("supersc_", {}).get("pass_count", 0) > 0:
        supersc_budget = max(supersc_budget, 4)
    if supersc_budget > 0:
        plan.append({"label": "supersc-generate", "cmd": [str(PY), "scripts/super_repair_candidates.py", "--max-add", str(supersc_budget), "--source-limit", "4"], "timeout": 300})
        plan.append({"label": "supersc-import", "cmd": [str(PY), "-m", "alpha_factory.cli", "import"], "timeout": 300})
        plan.append({"label": "supersc-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "2", "--id-prefix", "supersc_"], "timeout": 1800})
    plan.append({"label": "self-corr-truth", "cmd": [str(PY), "scripts/self_corr_truth_table.py"], "timeout": 300})
    plan.append({"label": "ml-score", "cmd": [str(PY), "scripts/ml_candidate_scorer.py", "--limit", "80"], "timeout": 300})
    if focus == "decor_":
        plan.extend([
            {"label": "decorrelate", "cmd": ["/bin/zsh", "scripts/run_decorrelate_pipeline.sh"], "timeout": 2400},
            {"label": "autosubmit", "cmd": ["/bin/zsh", "scripts/run_auto_submit.sh"], "timeout": 1800},
        ])
    elif focus == "refit_":
        plan.extend([
            {"label": "refit", "cmd": ["/bin/zsh", "scripts/run_refit_pipeline.sh"], "timeout": 2400},
            {"label": "autosubmit", "cmd": ["/bin/zsh", "scripts/run_auto_submit.sh"], "timeout": 1800},
        ])
    elif focus == "sparse_":
        if metrics["sparse_"]["pending"] > 0:
            plan.append({"label": "sparse-run", "cmd": [str(PY), "-m", "alpha_factory.cli", "run", "--limit", "6", "--id-prefix", "sparse_"], "timeout": 1800})
        plan.append({"label": "autosubmit", "cmd": ["/bin/zsh", "scripts/run_auto_submit.sh"], "timeout": 1800})
    else:
        plan.append({"label": "autosubmit", "cmd": ["/bin/zsh", "scripts/run_auto_submit.sh"], "timeout": 1800})
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Review, correct, and locally route WorldQuant alpha workflows")
    parser.add_argument("--allow-active-launches", action="store_true", help="Override the DB active-run guard and permit new backtest launches")
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    metrics = {family: family_metrics(conn, family) for family in FAMILIES}
    active_rows = active_backtest_rows(conn)
    auth = (
        {"ok": False, "skipped": "active launch guard; auth probe deferred", "returncode": None}
        if active_rows and not args.allow_active_launches
        else fetch_auth()
    )
    current_env = load_env(AUTONOMY_ENV)
    blockers = parse_blockers(AUTO_SUBMIT_LOG)
    ml_summary = ml_scorer_summary()
    multi_panel = multi_d1_panel_summary()
    previous_state = load_json(STATE_PATH)
    focus, focus_notes, family_watchdog = choose_focus(metrics, ml_summary=ml_summary, multi_panel=multi_panel, latest_state=previous_state)
    param_updates, param_notes, optimization_actions = decide_param_updates(focus, metrics, blockers, current_env)
    optimization_actions.append("Activated result-driven repair generator: high-turnover and self-correlation blockers now seed targeted repair candidates before broad exploration")
    optimization_actions.append("Activated Phase 2 submit-gated routing: high-quality repairsc candidates blocked by SELF_CORRELATION now seed second-order repairsc2 variants and receive extra repair budget ahead of exploration")
    optimization_actions.append("Activated stronger Phase 2.5 routing: repairsc2 variants that still hit SELF_CORRELATION now seed repairsc3 structural decorrelation candidates with distinct residual/gating transforms")
    optimization_actions.append("Activated Phase 3 self-correlation mutator: equivalent data-field substitutions (close/open/high/low/vwap/overnight/intraday) and operator-similarity mutations (rank/zscore/ts_rank/median/decay/residual) now seed repairsc3 and supersc variants without arbitrary grouping")
    optimization_actions.append("Activated dataset/field/function-family adaptive sampler: small-batch arms are selected by coverage/alpha-count priors plus observed value density, then promoted or paused by results")
    optimization_actions.append("Activated Multi Data Set Alpha track: cross-dataset field pairs now generate multi_ candidates with rank-spread, zscore-spread, gating, residual-helper, regime, and neutralized-product interactions")
    optimization_actions.append("Activated SuperAlpha builder: strong or self-correlation-blocked components now seed super_ blended/gated candidates to trade raw Sharpe for lower lineage concentration")
    optimization_actions.append("Activated SuperAlpha self-correlation repair: strong super_ candidates blocked by SELF_CORRELATION now seed supersc_ bucket/subindustry residual, market residual, liquidity/low-vol gate, and helper-orthogonalized variants")
    optimization_actions.append("Activated lightweight ML candidate scorer: recent candidates are ranked by quality, structural features, family value-density, and predicted self-correlation risk")
    save_env(AUTONOMY_ENV, param_updates)

    payload = {
        "timestamp_utc": now_iso(),
        "stage": "planned",
        "auth": auth,
        "focus": focus,
        "focus_notes": focus_notes,
        "param_updates": param_updates,
        "param_notes": param_notes,
        "optimization_actions": optimization_actions,
        "fresh_supply": load_json(ROOT / "state" / "fresh_supply_state.json"),
        "adaptive_sampler": adaptive_sampler_summary(),
        "multi_dataset": multi_dataset_summary(),
        "multi_d1_panel": multi_d1_panel_summary(),
        "lineage_occupancy": lineage_occupancy_summary(),
        "d1_generator": d1_generator_summary(),
        "superalpha": superalpha_summary(),
        "super_repair": super_repair_summary(),
        "self_corr_truth": self_corr_truth_summary(),
        "ml_scorer": ml_scorer_summary(),
        "family_watchdog": family_watchdog,
        "blockers": blockers,
        "metrics": metrics,
        "active_launch_guard": {
            "active": bool(active_rows),
            "count": len(active_rows),
            "override": bool(args.allow_active_launches),
            "decision": "allow" if args.allow_active_launches or not active_rows else "skip new backtest launch steps",
            "examples": active_rows[:8],
        },
        "action_plan": action_plan(focus, auth, metrics, blockers, ml_summary=ml_scorer_summary(), multi_panel=multi_d1_panel_summary()),
        "actions": [],
    }
    persist(payload)

    for step in payload["action_plan"]:
        if not step.get("cmd"):
            payload["actions"].append({"label": step["label"], "returncode": None, "skipped": True, "reason": step.get("skip_reason", "")})
            payload["stage"] = f"after_{step['label']}"
            persist(payload)
            continue
        if active_rows and not args.allow_active_launches and step["label"] == "strategy-monitor":
            step = {**step, "cmd": [str(PY), "scripts/strategy_monitor.py", "--observe-only"], "timeout": 180}
        if active_rows and not args.allow_active_launches and step_launches_backtest(step):
            reason = f"active backtest guard: {len(active_rows)} active RUNNING/SUBMITTING/recent RETRY row(s)"
            payload["actions"].append({"label": step["label"], "returncode": None, "skipped": True, "reason": reason})
            payload["stage"] = f"after_{step['label']}"
            persist(payload)
            continue
        code, out = run(step["cmd"], timeout=step["timeout"])
        nonfatal = is_nonfatal_step_failure(step["label"], code)
        payload["actions"].append({"label": step["label"], "returncode": code, "skipped": False, "nonfatal": nonfatal, "tail": out[-1200:]})
        payload["stage"] = f"after_{step['label']}"
        if step["label"] in {"fresh-supply-generate", "fresh-supply-run"}:
            payload["fresh_supply"] = load_json(ROOT / "state" / "fresh_supply_state.json")
        if step["label"] in {"multi-generate", "multi-run", "multi-d1-panel"}:
            payload["multi_dataset"] = multi_dataset_summary()
            payload["multi_d1_panel"] = multi_d1_panel_summary()
        if step["label"] == "lineage-occupancy":
            payload["lineage_occupancy"] = lineage_occupancy_summary()
        if step["label"] in {"d1-generate", "d1-run"}:
            payload["d1_generator"] = d1_generator_summary()
        if step["label"] in {"super-build", "super-run"}:
            payload["superalpha"] = superalpha_summary()
        if step["label"] in {"supersc-generate", "supersc-run"}:
            payload["super_repair"] = super_repair_summary()
        if step["label"] == "ml-score":
            payload["ml_scorer"] = ml_scorer_summary()
        if step["label"] == "self-corr-truth":
            payload["self_corr_truth"] = self_corr_truth_summary()
        persist(payload)
        if code != 0 and not nonfatal:
            break

    payload["stage"] = "completed"
    persist(payload)
    print(json.dumps({
        "focus": focus,
        "param_updates": param_updates,
        "optimization_actions": optimization_actions,
        "actions": payload["actions"],
        "report": str(REPORT_PATH)
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
