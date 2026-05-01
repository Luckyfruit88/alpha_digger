from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import yaml

from .brain_client import BrainAuthError, BrainClient, BrainRateLimit
from .database import Database
from .models import AlphaTask, BacktestResult


CSV_SETTING_FIELDS = {
    "type", "instrument_type", "region", "universe", "delay", "decay", "neutralization",
    "truncation", "pasteurization", "unit_handling", "nan_handling", "language", "visualization",
}


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_alpha_csv(path: str | Path, defaults: dict[str, Any]) -> list[AlphaTask]:
    tasks: list[AlphaTask] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            expression = (row.get("expression") or "").strip()
            if not expression:
                continue
            alpha_id = (row.get("id") or f"alpha_{i:05d}").strip()
            settings = dict(defaults)
            for key in CSV_SETTING_FIELDS:
                value = row.get(key)
                if value not in (None, ""):
                    settings[key] = _coerce_value(value)
            tasks.append(AlphaTask(id=alpha_id, expression=expression, settings=settings))
    return tasks


class BacktestWorker:
    def __init__(self, config: dict[str, Any], db: Database):
        self.config = config
        self.db = db
        brain_cfg = config["brain"]
        self.client = BrainClient(**brain_cfg)
        self.defaults = config["simulation_defaults"]
        self.worker_cfg = config["worker"]

    def build_payload(self, task: AlphaTask) -> dict[str, Any]:
        settings = dict(self.defaults)
        settings.update(task.settings)
        return {
            "type": settings.pop("type", "REGULAR"),
            "settings": {
                "instrumentType": settings.pop("instrument_type", "EQUITY"),
                "region": settings.pop("region", "USA"),
                "universe": settings.pop("universe", "TOP3000"),
                "delay": settings.pop("delay", 1),
                "decay": settings.pop("decay", 0),
                "neutralization": settings.pop("neutralization", "INDUSTRY"),
                "truncation": settings.pop("truncation", 0.08),
                "pasteurization": settings.pop("pasteurization", "ON"),
                "unitHandling": settings.pop("unit_handling", "VERIFY"),
                "nanHandling": settings.pop("nan_handling", "OFF"),
                "language": settings.pop("language", "FASTEXPR"),
                "visualization": settings.pop("visualization", False),
            },
            "regular": task.expression,
        }

    def run_pending(self, limit: int | None = None, id_prefix: str | None = None) -> None:
        tasks = self.db.pending_tasks(limit=limit, id_prefix=id_prefix)
        last_submit = 0.0
        for task in tasks:
            elapsed = time.monotonic() - last_submit
            min_gap = float(self.worker_cfg.get("min_submit_interval_seconds", 8))
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
            self.run_one(task)
            last_submit = time.monotonic()

    def submit_pending(self, limit: int | None = None, id_prefix: str | None = None) -> int:
        tasks = self.db.pending_tasks(limit=limit, id_prefix=id_prefix)
        submitted = 0
        last_submit = 0.0
        for task in tasks:
            elapsed = time.monotonic() - last_submit
            min_gap = float(self.worker_cfg.get("min_submit_interval_seconds", 8))
            if elapsed < min_gap:
                time.sleep(min_gap - elapsed)
            try:
                task.attempts += 1
                task.status = "SUBMITTING"
                self.db.mark_task(task)
                simulation_id = self.client.create_simulation(self.build_payload(task))
                task.simulation_id = simulation_id
                task.status = "RUNNING"
                task.last_error = None
                self.db.mark_task(task)
                submitted += 1
            except Exception as exc:
                task.status = "RETRY"
                task.last_error = str(exc)
                self.db.mark_task(task)
            last_submit = time.monotonic()
        return submitted

    def resume_running(self) -> int:
        count = 0
        for task in self.db.running_tasks():
            if not task.simulation_id:
                continue
            raw = self.client.get_simulation(task.simulation_id)
            progress = raw.get("progress")
            status = str(raw.get("status") or raw.get("state") or "").upper()
            if status in {"COMPLETE", "COMPLETED", "DONE", "ERROR", "FAILED", "FAIL", "WARNING", "WARN"}:
                raw = self.enrich_with_alpha_detail(raw)
                result = parse_result(task, raw)
                task.status = result.status
                task.last_error = result.error
                self.db.save_result(result)
                self.db.mark_task(task)
                count += 1
            else:
                task.last_error = f"still running; progress={progress}"
                self.db.mark_task(task)
        return count

    def run_one(self, task: AlphaTask) -> None:
        max_retries = int(self.worker_cfg.get("max_retries", 3))
        poll_interval = int(self.worker_cfg.get("poll_interval_seconds", 15))
        max_poll_seconds = self.worker_cfg.get("max_poll_seconds")
        max_poll_seconds = None if max_poll_seconds in (None, 0, "") else int(max_poll_seconds)
        while task.attempts < max_retries:
            task.attempts += 1
            try:
                task.status = "SUBMITTING"
                self.db.mark_task(task)
                simulation_id = self.client.create_simulation(self.build_payload(task))
                task.simulation_id = simulation_id
                task.status = "RUNNING"
                self.db.mark_task(task)

                raw = self.client.wait_for_simulation(
                    simulation_id,
                    poll_interval_seconds=poll_interval,
                    max_poll_seconds=max_poll_seconds,
                )
                raw = self.enrich_with_alpha_detail(raw)
                result = parse_result(task, raw)
                task.status = result.status
                task.last_error = result.error
                self.db.save_result(result)
                self.db.mark_task(task)
                return
            except BrainRateLimit as exc:
                wait = exc.retry_after if exc.retry_after and self.worker_cfg.get("respect_retry_after", True) else 60
                task.status = "RETRY"
                task.last_error = f"rate limited; retrying after {wait}s"
                self.db.mark_task(task)
                time.sleep(wait)
            except BrainAuthError as exc:
                task.status = "AUTH_ERROR"
                task.last_error = str(exc)
                self.db.mark_task(task)
                self.db.save_result(BacktestResult(
                    alpha_id=task.id,
                    expression=task.expression,
                    status="AUTH_ERROR",
                    simulation_id=task.simulation_id,
                    error=str(exc),
                ))
                if self.worker_cfg.get("stop_on_auth_error", True):
                    raise
                return
            except Exception as exc:
                task.status = "RETRY" if task.attempts < max_retries else "ERROR"
                task.last_error = str(exc)
                self.db.mark_task(task)
                if task.status == "ERROR":
                    self.db.save_result(BacktestResult(
                        alpha_id=task.id,
                        expression=task.expression,
                        status="ERROR",
                        simulation_id=task.simulation_id,
                        error=str(exc),
                    ))
                    return
                time.sleep(10 * task.attempts)

    def enrich_with_alpha_detail(self, raw: dict[str, Any]) -> dict[str, Any]:
        alpha = raw.get("alpha")
        alpha_id = alpha.get("id") if isinstance(alpha, dict) else alpha
        if not alpha_id:
            return raw
        detail = self.client.get_alpha(str(alpha_id))
        return {**raw, "alpha_detail": detail}


def parse_result(task: AlphaTask, raw: dict[str, Any]) -> BacktestResult:
    detail = raw.get("alpha_detail") if isinstance(raw.get("alpha_detail"), dict) else {}
    status = str(raw.get("status") or raw.get("state") or detail.get("stage") or "UNKNOWN").upper()
    alpha = raw.get("alpha") if isinstance(raw.get("alpha"), dict) else {}
    alpha_id = alpha.get("id") if alpha else raw.get("alpha")
    pnl = raw.get("pnl") if isinstance(raw.get("pnl"), dict) else {}
    is_metrics = detail.get("is") if isinstance(detail.get("is"), dict) else {}
    checks = is_metrics.get("checks") or raw.get("checks") or raw.get("is") or []

    metrics = is_metrics or (raw.get("is") if isinstance(raw.get("is"), dict) else raw)
    fail_reasons = _extract_fail_reasons(checks)
    checks_passed = None if not checks else not bool(fail_reasons)

    normalized_status = "COMPLETE" if status in {"COMPLETE", "COMPLETED", "DONE", "IS"} else "ERROR" if status in {"ERROR", "FAILED", "FAIL", "WARNING", "WARN"} else status
    return BacktestResult(
        alpha_id=task.id,
        expression=task.expression,
        status=normalized_status,
        simulation_id=task.simulation_id,
        sharpe=_num(metrics.get("sharpe")),
        fitness=_num(metrics.get("fitness")),
        turnover=_num(metrics.get("turnover")),
        returns=_num(metrics.get("returns") or metrics.get("return")),
        drawdown=_num(metrics.get("drawdown")),
        margin=_num(metrics.get("margin")),
        long_count=_int(metrics.get("longCount") or metrics.get("long_count") or pnl.get("longCount")),
        short_count=_int(metrics.get("shortCount") or metrics.get("short_count") or pnl.get("shortCount")),
        checks_passed=checks_passed,
        fail_reasons="; ".join(fail_reasons),
        raw_json={**raw, "worldquant_alpha_id": alpha_id},
        error=raw.get("message") or alpha.get("message") if normalized_status == "ERROR" else None,
    )


def _extract_fail_reasons(checks: Any) -> list[str]:
    reasons: list[str] = []
    if isinstance(checks, dict):
        checks = checks.get("checks") or checks.get("results") or []
    if isinstance(checks, list):
        for item in checks:
            if not isinstance(item, dict):
                continue
            passed = item.get("result") or item.get("status") or item.get("passed")
            is_fail = passed is False or str(passed).upper() in {"FAIL", "FAILED", "ERROR"}
            if is_fail:
                reasons.append(str(item.get("name") or item.get("check") or item.get("message") or item))
    return reasons


def _coerce_value(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None
