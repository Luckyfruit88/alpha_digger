from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alpha_factory.backtester import load_config  # noqa: E402
from alpha_factory.brain_client import BrainClient, BrainRateLimit  # noqa: E402

KEYWORDS = {
    "revision": 18,
    "surprise": 18,
    "estimate": 14,
    "forecast": 14,
    "growth": 12,
    "margin": 12,
    "accrual": 12,
    "quality": 10,
    "momentum": 10,
    "value": 9,
    "earnings": 9,
    "cash flow": 8,
    "cashflow": 8,
    "sales": 7,
    "revenue": 7,
    "debt": 6,
    "short": 6,
    "sentiment": 12,
    "news": 10,
    "analyst": 8,
}

CATEGORY_BONUS = {
    "Analyst": 18,
    "Fundamental": 16,
    "Model": 12,
    "News": 12,
    "Sentiment": 12,
    "Option": 10,
    "Price Volume": -8,
}


def text_of(field: dict[str, Any]) -> str:
    parts = [field.get("id", ""), field.get("description", "")]
    for key in ("dataset", "category", "subcategory"):
        val = field.get(key)
        if isinstance(val, dict):
            parts.extend([str(val.get("id", "")), str(val.get("name", ""))])
    return " ".join(parts).lower()


def score_field(field: dict[str, Any]) -> float:
    score = 0.0
    coverage = field.get("coverage")
    if isinstance(coverage, (int, float)):
        score += 35 * max(0.0, min(1.0, float(coverage)))
        if coverage < 0.35:
            score -= 20
    date_coverage = field.get("dateCoverage")
    if isinstance(date_coverage, (int, float)):
        score += 10 * max(0.0, min(1.0, float(date_coverage)))
    user_count = field.get("userCount")
    if isinstance(user_count, (int, float)):
        score += max(0.0, 15 - math.log1p(user_count) * 3.0)
    alpha_count = field.get("alphaCount")
    if isinstance(alpha_count, (int, float)):
        score += max(0.0, 15 - math.log1p(alpha_count) * 2.5)
    category = (field.get("category") or {}).get("name") if isinstance(field.get("category"), dict) else None
    if category:
        score += CATEGORY_BONUS.get(str(category), 0)
    hay = text_of(field)
    for keyword, bonus in KEYWORDS.items():
        if keyword in hay:
            score += bonus
    if field.get("type") == "MATRIX":
        score += 5
    return round(score, 3)


def flatten(field: dict[str, Any]) -> dict[str, Any]:
    dataset = field.get("dataset") or {}
    category = field.get("category") or {}
    subcategory = field.get("subcategory") or {}
    return {
        "score": score_field(field),
        "id": field.get("id", ""),
        "description": field.get("description", ""),
        "dataset_id": dataset.get("id", "") if isinstance(dataset, dict) else "",
        "dataset_name": dataset.get("name", "") if isinstance(dataset, dict) else "",
        "category": category.get("name", "") if isinstance(category, dict) else "",
        "subcategory": subcategory.get("name", "") if isinstance(subcategory, dict) else "",
        "region": field.get("region", ""),
        "delay": field.get("delay", ""),
        "universe": field.get("universe", ""),
        "type": field.get("type", ""),
        "coverage": field.get("coverage", ""),
        "dateCoverage": field.get("dateCoverage", ""),
        "userCount": field.get("userCount", ""),
        "alphaCount": field.get("alphaCount", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and score WorldQuant Brain data fields slowly/resumably")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-pages", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=30.0)
    parser.add_argument("--out", default="data/data_fields_scored.csv")
    parser.add_argument("--raw-dir", default="data/data_fields_raw")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    client = BrainClient(**cfg["brain"])
    out = ROOT / args.out
    raw_dir = ROOT / args.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict[str, Any]] = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing[row["id"]] = row

    offset = 0
    pages_done = 0
    total_count = None
    while pages_done < args.max_pages:
        path = f"/data-fields?instrumentType=EQUITY&region=USA&universe=TOP3000&delay=1&limit={args.limit}&offset={offset}"
        try:
            response = client._request("GET", path)
        except BrainRateLimit as exc:
            wait = exc.retry_after or max(args.sleep, 60)
            print(f"rate limited; sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        data = response.json()
        total_count = data.get("count", total_count)
        results = data.get("results") or []
        if not results:
            break
        (raw_dir / f"fields_offset_{offset}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        for field in results:
            row = flatten(field)
            existing[row["id"]] = row
        pages_done += 1
        print(f"fetched offset={offset} got={len(results)} total={total_count} unique={len(existing)}", flush=True)
        offset += args.limit
        if total_count is not None and offset >= int(total_count):
            break
        time.sleep(args.sleep)

    rows = sorted(existing.values(), key=lambda r: float(r.get("score") or 0), reverse=True)
    fieldnames = [
        "score", "id", "description", "dataset_id", "dataset_name", "category", "subcategory",
        "region", "delay", "universe", "type", "coverage", "dateCoverage", "userCount", "alphaCount",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} scored fields to {out}")
    print("top 20:")
    for row in rows[:20]:
        print(f"{row['score']}\t{row['id']}\t{row['category']}\tcoverage={row['coverage']}\tusers={row['userCount']}\talpha={row['alphaCount']}\t{row['description'][:90]}")


if __name__ == "__main__":
    main()
