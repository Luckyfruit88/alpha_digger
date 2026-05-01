#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.backtester import load_config
from alpha_factory.brain_client import BrainClient, BrainRateLimit

DEFAULT_OUTPUT = ROOT / "data" / "submitted_alpha_library.jsonl"
SAFE_SETTING_KEYS = {"region", "universe", "delay", "decay", "neutralization", "truncation", "pasteurization", "language", "nanHandling", "unitHandling"}
SAFE_METRIC_KEYS = {"sharpe", "fitness", "turnover", "returns", "drawdown", "margin", "longCount", "shortCount"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        return {}
    return {k: settings.get(k) for k in SAFE_SETTING_KEYS if k in settings}


def metric_source(row: dict[str, Any]) -> dict[str, Any]:
    is_obj = row.get("is") if isinstance(row.get("is"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    merged = {**metrics, **is_obj}
    return {k: merged.get(k) for k in SAFE_METRIC_KEYS if k in merged}


def row_expression(row: dict[str, Any]) -> str:
    regular = row.get("regular") if isinstance(row.get("regular"), dict) else {}
    code = regular.get("code") or row.get("expression") or row.get("code")
    return str(code or "").strip()


def normalize_row(row: dict[str, Any], fetched_at: str) -> dict[str, Any] | None:
    date_submitted = row.get("dateSubmitted") or row.get("date_submitted")
    expr = row_expression(row)
    if not date_submitted or not expr:
        return None
    return {
        "alpha_id": str(row.get("id") or row.get("alpha_id") or ""),
        "status": row.get("status"),
        "stage": row.get("stage"),
        "date_submitted": date_submitted,
        "settings": compact_settings(row.get("settings")),
        "metrics": metric_source(row),
        "expression": expr,
        "fetched_at": fetched_at,
    }


def response_rows(data: Any) -> tuple[list[dict[str, Any]], int | None]:
    if isinstance(data, dict):
        raw = data.get("results") or data.get("alphas") or data.get("data") or []
        count = data.get("count") if isinstance(data.get("count"), int) else None
    elif isinstance(data, list):
        raw = data
        count = None
    else:
        raw = []
        count = None
    rows = [r for r in raw if isinstance(r, dict)]
    return rows, count


def fetch(limit: int, page_size: int, max_pages: int | None, sleep_seconds: float, output: Path, dry_run: bool = False) -> dict[str, Any]:
    cfg = load_config(ROOT / "config.yaml")
    client = BrainClient(**cfg["brain"])
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    offset = 0
    page = 0
    total_seen = 0
    fetched_at = now_iso()
    while len(kept) < limit:
        if max_pages is not None and page >= max_pages:
            break
        endpoint = f"/users/self/alphas?order=-dateSubmitted&limit={page_size}&offset={offset}"
        try:
            resp = client._request("GET", endpoint)
        except BrainRateLimit as exc:
            wait = exc.retry_after or max(5, int(sleep_seconds))
            print(json.dumps({"rate_limited": True, "retry_after": wait, "page": page}, ensure_ascii=False), file=sys.stderr)
            time.sleep(wait)
            continue
        data = resp.json() if resp.content else {}
        rows, count = response_rows(data)
        if not rows:
            break
        total_seen += len(rows)
        for row in rows:
            norm = normalize_row(row, fetched_at)
            if not norm:
                continue
            aid = norm.get("alpha_id") or f"row-{offset}-{len(kept)}"
            if aid in seen:
                continue
            seen.add(str(aid))
            kept.append(norm)
            if len(kept) >= limit:
                break
        page += 1
        offset += page_size
        if count is not None and offset >= count:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    if not dry_run:
        with output.open("w", encoding="utf-8") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {"fetched_at": fetched_at, "pages": page, "api_rows_seen": total_seen, "submitted_kept": len(kept), "output": str(output), "dry_run": dry_run}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch read-only submitted/ACTIVE WorldQuant alpha library")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = fetch(args.limit, args.page_size, args.max_pages, args.sleep_seconds, args.output, args.dry_run)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
