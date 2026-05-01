from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from .backtester import BacktestWorker, load_alpha_csv, load_config
from .database import Database
from .generator import append_unique_to_csv, generate_expressions
from .reporter import export_reports


def configure_logging(root: Path) -> None:
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "supervisor.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously supervise WorldQuant Brain backtests")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--interval", type=int, default=120, help="Seconds between supervisor cycles")
    parser.add_argument("--batch-size", type=int, default=1, help="New pending simulations per cycle")
    parser.add_argument("--once", action="store_true", help="Run one supervisor cycle and exit")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    root = cfg_path.parent
    configure_logging(root)
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    db = Database(root / paths["sqlite_db"])

    logging.info("supervisor started; submit mode=backtest-only; final alpha submission=disabled")
    try:
        while True:
            try:
                gen_cfg = cfg.get("generator", {})
                generated_added = 0
                if gen_cfg.get("enabled", False):
                    expressions = generate_expressions(
                        int(gen_cfg.get("per_cycle", 5)),
                        seed=gen_cfg.get("seed"),
                        fields=gen_cfg.get("fields"),
                    )
                    generated_added = append_unique_to_csv(
                        root / paths["input_csv"],
                        expressions,
                        cfg["simulation_defaults"],
                    )
                    logging.info("generated alphas; attempted=%s added=%s", len(expressions), generated_added)

                tasks = load_alpha_csv(root / paths["input_csv"], cfg["simulation_defaults"])
                db.upsert_tasks(tasks)
                worker = BacktestWorker(cfg, db)
                completed = worker.resume_running()
                submitted = worker.submit_pending(limit=args.batch_size)
                export_reports(db, root / paths["results_csv"], root / paths["markdown_report"], cfg.get("selection"))
                logging.info(
                    "cycle ok; generated_added=%s imported=%s submitted=%s resumed_completed=%s batch_size=%s",
                    generated_added,
                    len(tasks),
                    submitted,
                    completed,
                    args.batch_size,
                )
            except Exception as exc:
                logging.exception("cycle failed: %s", exc)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        db.close()
        logging.info("supervisor stopped")


if __name__ == "__main__":
    main()
