#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "backtests.sqlite3"
AUTO_SUBMIT_LOG = ROOT / "logs" / "auto_submit.log"
SUBMISSIONS_PATH = ROOT / "data" / "auto_submissions.jsonl"
AUTO_SUBMIT_STATE_PATH = ROOT / "state" / "auto_submit_state.json"
SUPER_STATE_PATH = ROOT / "state" / "superalpha_state.json"
D1_STATE_PATH = ROOT / "state" / "d1_generator_state.json"
D1_TRUTH_TABLE_PATH = ROOT / "state" / "d1_truth_table.json"
MULTI_STATE_PATH = ROOT / "state" / "multi_dataset_state.json"
REPAIR_STATE_PATH = ROOT / "state" / "repair_candidates_state.json"
OUT_JSON = ROOT / "state" / "self_corr_truth_table.json"
OUT_REPORT = ROOT / "reports" / "self_corr_truth_table.md"

PASS_THRESHOLD = {"sharpe": 1.6, "fitness": 1.0, "turnover": 0.45}
STATUS_PRIORITY = {
    "unknown": 0,
    "predicted_blocked": 1,
    "clear": 2,
    "cooldown": 3,
    "pending": 4,
    "blocked": 5,
    "retired": 6,
}

FIELD_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
FUNCTION_TOKENS = {
    "abs",
    "bucket",
    "group_mean",
    "group_neutralize",
    "rank",
    "range",
    "signed_power",
    "trade_when",
    "ts_corr",
    "ts_decay_linear",
    "ts_delta",
    "ts_mean",
    "ts_median",
    "ts_rank",
    "ts_std_dev",
    "ts_zscore",
}
REPAIR_PREFIXES = ("repairsc_", "repairsc2_", "repairsc3_")
GROUP_TOKENS = {"industry", "market", "sector", "subindustry"}
PRICE_TOKENS = {"close", "open", "high", "low", "vwap", "returns", "ret", "daily_return", "price", "volume"}
LIQUIDITY_VOL_TOKENS = {"adv20", "turnover", "liquidity", "volatility", "std", "std_dev", "spread"}
ANALYST_EARNINGS_TOKENS = {
    "analyst",
    "anl",
    "eps",
    "earnings",
    "surprise",
    "revision",
    "estimate",
    "rating",
    "netincome",
    "income",
}
FUNDAMENTAL_VALUATION_TOKENS = {
    "assets",
    "asset",
    "book",
    "cash",
    "debt",
    "enterprise",
    "ev",
    "fundamental",
    "liabilities",
    "margin",
    "revenue",
    "sales",
    "value",
}
MARKET_SIZE_TOKENS = {"cap", "marketcap", "size"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def family(alpha_id: str) -> str:
    return alpha_id.split("_", 1)[0] + "_" if "_" in alpha_id else alpha_id


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def worldquant_alpha_id(raw_json: str | None) -> str | None:
    if not raw_json:
        return None
    try:
        raw = json.loads(raw_json)
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    detail = raw.get("alpha_detail") if isinstance(raw.get("alpha_detail"), dict) else {}
    value = detail.get("id") or raw.get("worldquant_alpha_id") or raw.get("alpha") or raw.get("id")
    return str(value) if value else None


def pass_quality(sharpe: Any, fitness: Any, turnover: Any) -> bool:
    s = to_float(sharpe)
    f = to_float(fitness)
    t = to_float(turnover)
    return bool(
        s is not None
        and f is not None
        and t is not None
        and s >= PASS_THRESHOLD["sharpe"]
        and f >= PASS_THRESHOLD["fitness"]
        and t <= PASS_THRESHOLD["turnover"]
    )


def important_tokens(expression: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for token in FIELD_RE.findall(expression or ""):
        t = token.lower()
        if t in FUNCTION_TOKENS or t in GROUP_TOKENS or t in {"nan", "inf", "true", "false"}:
            continue
        if t not in seen and not t.isdigit():
            seen.add(t)
            out.append(token)
    return out[:24]


def token_matches(tokens: list[str], needles: set[str]) -> bool:
    lowered = [t.lower() for t in tokens]
    return any(any(needle in token for needle in needles) for token in lowered)


def classify_lineage(expression: str, datasets_or_fields: list[str] | None = None) -> dict[str, Any]:
    tokens = important_tokens(expression)
    merged = tokens + [str(x) for x in (datasets_or_fields or []) if x]
    themes: list[str] = []
    if token_matches(merged, ANALYST_EARNINGS_TOKENS):
        themes.append("analyst_earnings")
    if token_matches(merged, PRICE_TOKENS | {"pv1"}):
        themes.append("price_volume")
    if token_matches(merged, LIQUIDITY_VOL_TOKENS):
        themes.append("liquidity_volatility")
    if token_matches(merged, FUNDAMENTAL_VALUATION_TOKENS):
        themes.append("fundamental_valuation")
    if token_matches(merged, MARKET_SIZE_TOKENS):
        themes.append("market_size")
    if not themes:
        theme = "unknown"
    elif len(set(themes)) == 1:
        theme = themes[0]
    else:
        # Analyst/earnings + returns appears constantly in this repo; keeping the
        # dominant economic lineage visible is more useful than calling it mixed.
        non_price = [x for x in themes if x not in {"price_volume", "market_size"}]
        theme = non_price[0] if len(set(non_price)) == 1 else "mixed"
    uses_price_corr = "ts_corr" in (expression or "") and token_matches(tokens, PRICE_TOKENS | LIQUIDITY_VOL_TOKENS)
    return {
        "theme": theme,
        "datasets_or_fields": merged[:28],
        "uses_returns_or_price_corr": bool(uses_price_corr),
        "parent_ids": [],
    }


def parent_metadata(
    super_state_path: Path = SUPER_STATE_PATH,
    d1_state_path: Path = D1_STATE_PATH,
    multi_state_path: Path = MULTI_STATE_PATH,
    repair_state_path: Path = REPAIR_STATE_PATH,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    super_state = load_json(super_state_path)
    for alpha_id, meta in (super_state.get("built_from") or {}).items():
        out[str(alpha_id)] = {
            "parent_ids": [str(x) for x in meta.get("parent_alpha_ids") or [] if x],
            "datasets_or_fields": [str(x) for x in meta.get("parent_families") or [] if x],
        }
    d1_state = load_json(d1_state_path)
    for alpha_id, meta in (d1_state.get("candidates") or {}).items():
        fields = [
            meta.get("primary_dataset"),
            meta.get("secondary_dataset"),
            meta.get("primary_field"),
            meta.get("secondary_field"),
            *(meta.get("anchor_datasets") or []),
        ]
        parent_ids = [str(meta.get("anchor_id"))] if meta.get("anchor_id") else []
        out[str(alpha_id)] = {
            "parent_ids": [x for x in parent_ids if x],
            "datasets_or_fields": [str(x) for x in fields if x],
        }
    multi_state = load_json(multi_state_path)
    for alpha_id, meta in (multi_state.get("alphas") or {}).items():
        fields = list(meta.get("datasets") or []) + list(meta.get("fields") or [])
        out[str(alpha_id)] = {
            "parent_ids": [],
            "datasets_or_fields": [str(x) for x in fields if x],
        }
    repair_state = load_json(repair_state_path)
    for alpha_id, parent_id in (repair_state.get("seeded_from") or {}).items():
        current = out.setdefault(str(alpha_id), {"parent_ids": [], "datasets_or_fields": []})
        if parent_id:
            current.setdefault("parent_ids", []).append(str(parent_id))
    return out


def repair_depth(alpha_id: str, parent_map: dict[str, list[str]], seen: set[str] | None = None) -> int:
    seen = seen or set()
    if alpha_id in seen:
        return 0
    seen.add(alpha_id)
    own = 1 if alpha_id.startswith(REPAIR_PREFIXES) else 0
    parents = parent_map.get(alpha_id) or []
    if not parents:
        return own
    return own + max(repair_depth(parent_id, parent_map, seen.copy()) for parent_id in parents)


def reason_category(reason: str) -> str:
    r = reason.lower()
    if "pre-submit self-corr gate" in r:
        return "pre_submit_self_corr_gate"
    if "self-correlation cooldown active" in r:
        return "self_corr_cooldown"
    if "unsafe checks:" in r and "self_correlation:pending" in r:
        return "self_corr_pending"
    if "unsafe checks:" in r and "self_correlation:fail" in r:
        return "self_corr_blocked"
    if "unsafe checks:" in r and "self_correlation:" in r:
        return "self_corr_unsafe"
    if "already submitted" in r:
        return "already_submitted"
    if "already recorded" in r:
        return "already_recorded"
    if "detail check budget" in r:
        return "detail_budget"
    if "missing worldquant alpha id" in r:
        return "missing_worldquant_alpha_id"
    if "pre-submit d1 gate" in r:
        return "pre_submit_d1_gate"
    if "retired" in r and "self_corr_blocked" in r:
        return "retired_self_corr_blocked"
    if "detail check clear" in r:
        return "detail_check_clear"
    return r[:80]


def status_from_reason(reason: str) -> str | None:
    r = reason.lower()
    if "unsafe checks:" in r and "self_correlation:fail" in r:
        return "blocked"
    if "unsafe checks:" in r and "self_correlation:pending" in r:
        return "pending"
    if "self-correlation cooldown active" in r:
        return "cooldown"
    if "pre-submit self-corr gate" in r:
        return "predicted_blocked"
    if "retired" in r and "self_corr_blocked" in r:
        return "retired"
    if "detail check clear" in r:
        return "clear"
    if "already submitted" in r:
        return "clear"
    return None


def add_evidence(row: dict[str, Any], status: str | None, evidence: str, seen_at: str | None = None) -> None:
    if status and STATUS_PRIORITY[status] > STATUS_PRIORITY[row["self_corr_status"]]:
        row["self_corr_status"] = status
    if evidence not in row["evidence"]:
        row["evidence"].append(evidence)
    if seen_at:
        current = row.get("last_seen_at")
        row["last_seen_at"] = max(str(current or ""), seen_at)


def build_truth_table(
    db_path: Path = DB_PATH,
    auto_submit_log: Path = AUTO_SUBMIT_LOG,
    submissions_path: Path = SUBMISSIONS_PATH,
    auto_submit_state_path: Path = AUTO_SUBMIT_STATE_PATH,
    super_state_path: Path = SUPER_STATE_PATH,
    d1_state_path: Path = D1_STATE_PATH,
    multi_state_path: Path = MULTI_STATE_PATH,
    d1_truth_table_path: Path = D1_TRUTH_TABLE_PATH,
    repair_state_path: Path = REPAIR_STATE_PATH,
) -> dict[str, Any]:
    parents = parent_metadata(super_state_path, d1_state_path, multi_state_path, repair_state_path)
    rows: dict[str, dict[str, Any]] = {}

    if db_path.exists():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        for r in conn.execute(
            """
            select alpha_id, expression, sharpe, fitness, turnover, raw_json, fail_reasons, created_at
            from backtest_results
            """
        ).fetchall():
            alpha_id = str(r["alpha_id"])
            meta = parents.get(alpha_id, {})
            lineage = classify_lineage(str(r["expression"] or ""), list(meta.get("datasets_or_fields") or []))
            lineage["parent_ids"] = list(meta.get("parent_ids") or [])
            rows[alpha_id] = {
                "alpha_id": alpha_id,
                "family": family(alpha_id),
                "expression": str(r["expression"] or ""),
                "sharpe": to_float(r["sharpe"]),
                "fitness": to_float(r["fitness"]),
                "turnover": to_float(r["turnover"]),
                "pass_quality": pass_quality(r["sharpe"], r["fitness"], r["turnover"]),
                "worldquant_alpha_id": worldquant_alpha_id(r["raw_json"]),
                "self_corr_status": "unknown",
                "evidence": [],
                "last_seen_at": str(r["created_at"] or ""),
                "skip_counts": {},
                "lineage": lineage,
            }
            fail = str(r["fail_reasons"] or "")
            if "SELF_CORRELATION" in fail:
                add_evidence(rows[alpha_id], "blocked", "backtest_results: fail_reasons SELF_CORRELATION", str(r["created_at"] or ""))

        for r in conn.execute("select id, expression, updated_at from alpha_tasks").fetchall():
            alpha_id = str(r["id"])
            if alpha_id in rows:
                continue
            meta = parents.get(alpha_id, {})
            lineage = classify_lineage(str(r["expression"] or ""), list(meta.get("datasets_or_fields") or []))
            lineage["parent_ids"] = list(meta.get("parent_ids") or [])
            rows[alpha_id] = {
                "alpha_id": alpha_id,
                "family": family(alpha_id),
                "expression": str(r["expression"] or ""),
                "sharpe": None,
                "fitness": None,
                "turnover": None,
                "pass_quality": False,
                "worldquant_alpha_id": None,
                "self_corr_status": "unknown",
                "evidence": [],
                "last_seen_at": str(r["updated_at"] or ""),
                "skip_counts": {},
                "lineage": lineage,
            }
        conn.close()

    for payload in iter_json_lines(auto_submit_log):
        seen_at = str(payload.get("timestamp") or payload.get("timestamp_utc") or "")
        for item in payload.get("skipped", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            alpha_id, reason = str(item[0]), str(item[1])
            if alpha_id not in rows:
                rows[alpha_id] = {
                    "alpha_id": alpha_id,
                    "family": family(alpha_id),
                    "expression": "",
                    "sharpe": None,
                    "fitness": None,
                    "turnover": None,
                    "pass_quality": False,
                    "worldquant_alpha_id": None,
                    "self_corr_status": "unknown",
                    "evidence": [],
                    "last_seen_at": seen_at,
                    "skip_counts": {},
                    "lineage": classify_lineage(""),
                }
            category = reason_category(reason)
            rows[alpha_id]["skip_counts"][category] = int(rows[alpha_id]["skip_counts"].get(category, 0)) + 1
            if category == "pre_submit_self_corr_gate":
                match = re.search(r"p_self_corr_block=([0-9.]+)", reason)
                suffix = f" p={match.group(1)}" if match else ""
                evidence = f"auto_submit: pre-submit gate{suffix}"
            else:
                evidence = f"auto_submit: {reason[:140]}"
            add_evidence(rows[alpha_id], status_from_reason(reason), evidence, seen_at)
        for item in payload.get("submitted", []) or []:
            if not isinstance(item, dict):
                continue
            alpha_id = str(item.get("local_id") or "")
            if alpha_id and alpha_id in rows:
                if item.get("alpha_id") and not rows[alpha_id].get("worldquant_alpha_id"):
                    rows[alpha_id]["worldquant_alpha_id"] = str(item.get("alpha_id"))
                add_evidence(rows[alpha_id], "clear", "auto_submit: submitted", seen_at)

    for payload in iter_json_lines(submissions_path):
        local_id = str(payload.get("local_id") or "")
        status = str(payload.get("status") or "").lower()
        if not local_id or local_id not in rows:
            continue
        if payload.get("alpha_id") and not rows[local_id].get("worldquant_alpha_id"):
            rows[local_id]["worldquant_alpha_id"] = str(payload.get("alpha_id"))
        if status in {"submitted", "already_submitted"}:
            add_evidence(rows[local_id], "clear", status, str(payload.get("timestamp") or ""))

    cooldowns = (load_json(auto_submit_state_path).get("cooldowns") or {})
    cooldown_wq_ids = {str(k) for k in cooldowns}
    for row in rows.values():
        if row.get("worldquant_alpha_id") in cooldown_wq_ids:
            add_evidence(row, "cooldown", "cooldown_state")

    for entry in (load_json(d1_truth_table_path).get("entries") or []):
        if not isinstance(entry, dict):
            continue
        alpha_id = str(entry.get("alpha_id") or "")
        if not alpha_id:
            continue
        if alpha_id not in rows:
            rows[alpha_id] = {
                "alpha_id": alpha_id,
                "family": family(alpha_id),
                "expression": "",
                "sharpe": None,
                "fitness": None,
                "turnover": None,
                "pass_quality": False,
                "worldquant_alpha_id": None,
                "self_corr_status": "unknown",
                "evidence": [],
                "last_seen_at": str(entry.get("checked_at") or ""),
                "skip_counts": {},
                "lineage": classify_lineage(""),
            }
        result = str(entry.get("detail_check_result") or "pending")
        status = "clear" if result == "clear" else "pending" if result == "pending" else "blocked"
        add_evidence(rows[alpha_id], status, f"d1_truth_table: {result}", str(entry.get("checked_at") or ""))

    repair_state = load_json(repair_state_path)
    for item in repair_state.get("retired", []) or []:
        if not isinstance(item, dict):
            continue
        alpha_id = str(item.get("alpha_id") or "")
        if not alpha_id:
            continue
        if alpha_id not in rows:
            rows[alpha_id] = {
                "alpha_id": alpha_id,
                "family": family(alpha_id),
                "expression": "",
                "sharpe": None,
                "fitness": None,
                "turnover": None,
                "pass_quality": False,
                "worldquant_alpha_id": None,
                "self_corr_status": "unknown",
                "evidence": [],
                "last_seen_at": str(item.get("timestamp_utc") or ""),
                "skip_counts": {},
                "lineage": classify_lineage(""),
            }
        add_evidence(rows[alpha_id], "retired", f"repair_candidates: RETIRED repair_depth={item.get('repair_depth')} reason={item.get('reason')}", str(item.get("timestamp_utc") or ""))

    parent_map = {alpha_id: list((row.get("lineage") or {}).get("parent_ids") or []) for alpha_id, row in rows.items()}
    for alpha_id, row in rows.items():
        row["repair_depth"] = repair_depth(alpha_id, parent_map)

    records = sorted(rows.values(), key=lambda r: (r.get("family") or "", r.get("alpha_id") or ""))
    summary = summarize(records)
    return {"version": 1, "generated_at": now_iso(), "summary": summary, "alphas": records}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(r.get("self_corr_status", "unknown")) for r in records)
    repair_depth_counts = {"0": 0, "1": 0, "2": 0, "3+": 0}
    family_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "pass_quality": 0})
    theme_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "pass_quality": 0})
    known = clear = 0
    for r in records:
        depth = int(to_float(r.get("repair_depth"), 0) or 0)
        if depth >= 3:
            repair_depth_counts["3+"] += 1
        else:
            repair_depth_counts[str(depth)] += 1
        fam = str(r.get("family") or "unknown")
        theme = str((r.get("lineage") or {}).get("theme") or "unknown")
        family_counts[fam]["count"] += 1
        theme_counts[theme]["count"] += 1
        if r.get("pass_quality"):
            family_counts[fam]["pass_quality"] += 1
            theme_counts[theme]["pass_quality"] += 1
        if r.get("self_corr_status") in {"clear", "blocked", "pending", "cooldown"}:
            known += 1
            if r.get("self_corr_status") == "clear":
                clear += 1
    repeated = sorted(
        (
            {
                "alpha_id": r["alpha_id"],
                "status": r["self_corr_status"],
                "skip_counts": r.get("skip_counts", {}),
                "total_skips": sum(int(v) for v in (r.get("skip_counts") or {}).values()),
            }
            for r in records
            if r.get("self_corr_status") in {"blocked", "pending", "cooldown", "predicted_blocked"}
        ),
        key=lambda x: x["total_skips"],
        reverse=True,
    )[:20]
    return {
        "total": len(records),
        "status_counts": dict(status_counts),
        "family_counts": dict(sorted(family_counts.items())),
        "lineage_theme_counts": dict(sorted(theme_counts.items())),
        "repair_depth_distribution": repair_depth_counts,
        "known_clear_rate": round(clear / known, 4) if known else None,
        "known_evidence_count": known,
        "top_repeated_blocked_local_ids": repeated,
    }


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    summary = payload.get("summary") or {}
    lines = [
        f"# Self-Correlation Truth Table - {payload.get('generated_at')}",
        "",
        f"- total alpha ids: `{summary.get('total', 0)}`",
        f"- status_counts: `{summary.get('status_counts', {})}`",
        f"- known_clear_rate: `{summary.get('known_clear_rate')}` over `{summary.get('known_evidence_count', 0)}` known evidence rows",
        "",
        "## By Family",
    ]
    for fam, stats in (summary.get("family_counts") or {}).items():
        lines.append(f"- {fam}: count={stats.get('count')} pass_quality={stats.get('pass_quality')}")
    lines.extend(["", "## By Lineage Theme"])
    for theme, stats in (summary.get("lineage_theme_counts") or {}).items():
        lines.append(f"- {theme}: count={stats.get('count')} pass_quality={stats.get('pass_quality')}")
    lines.extend(["", "## Repair Depth Distribution"])
    for depth, count in (summary.get("repair_depth_distribution") or {}).items():
        lines.append(f"- {depth}: {count}")
    lines.extend(["", "## Top Repeated Blocked / Predicted Local IDs"])
    for item in (summary.get("top_repeated_blocked_local_ids") or [])[:15]:
        lines.append(f"- {item.get('alpha_id')} status={item.get('status')} skips={item.get('total_skips')} categories={item.get('skip_counts')}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a self-correlation evidence truth table from local WorldQuant logs/state")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--auto-submit-log", type=Path, default=AUTO_SUBMIT_LOG)
    parser.add_argument("--submissions", type=Path, default=SUBMISSIONS_PATH)
    parser.add_argument("--auto-submit-state", type=Path, default=AUTO_SUBMIT_STATE_PATH)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-report", type=Path, default=OUT_REPORT)
    args = parser.parse_args()

    payload = build_truth_table(
        db_path=args.db,
        auto_submit_log=args.auto_submit_log,
        submissions_path=args.submissions,
        auto_submit_state_path=args.auto_submit_state,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(payload, args.out_report)
    print(json.dumps({"generated_at": payload["generated_at"], "summary": payload["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
