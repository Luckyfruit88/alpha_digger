from __future__ import annotations

import argparse
from pathlib import Path

from .backtester import BacktestWorker, load_alpha_csv, load_config
from .database import Database
from .decorrelate_generator import append_unique_to_csv as append_decor_unique_to_csv, generate_decorrelated_expressions
from .reporter import export_reports
from .refit_generator import append_unique_to_csv as append_refit_unique_to_csv, generate_refit_expressions


def main() -> None:
    parser = argparse.ArgumentParser(description="WorldQuant Brain session-based alpha backtest worker")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    import_cmd = sub.add_parser("import", help="Import alpha expressions from CSV into SQLite queue")
    import_cmd.add_argument("--csv", default=None, help="Override input CSV path")

    run_cmd = sub.add_parser("run", help="Run pending backtests")
    run_cmd.add_argument("--limit", type=int, default=None, help="Max number of pending tasks to run")
    run_cmd.add_argument("--id-prefix", default=None, help="Only run pending tasks whose id starts with this prefix")
    run_cmd.add_argument("--allow-active-launches", action="store_true", help="Override the active RUNNING/SUBMITTING/RETRY launch guard")

    sub.add_parser("resume", help="Poll and resume active RUNNING simulations without launching new ones")

    refit_cmd = sub.add_parser("refit-generate", help="Generate variants by refitting top/high-quality alpha templates")
    refit_cmd.add_argument("--count", type=int, default=10, help="How many refit expressions to generate")
    refit_cmd.add_argument("--csv", default=None, help="Override output CSV path")

    decor_cmd = sub.add_parser("decorrelate-generate", help="Generate structurally decorrelated variants for self-correlation failures")
    decor_cmd.add_argument("--count", type=int, default=10, help="How many decorrelated expressions to generate")
    decor_cmd.add_argument("--csv", default=None, help="Override output CSV path")

    sub.add_parser("report", help="Export CSV and Markdown reports")
    sub.add_parser("check-auth", help="Probe browser-cookie based WorldQuant Brain session")

    args = parser.parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    root = cfg_path.parent
    paths = cfg["paths"]
    db = Database(root / paths["sqlite_db"])

    try:
        if args.command == "import":
            csv_path = Path(args.csv) if args.csv else root / paths["input_csv"]
            tasks = load_alpha_csv(csv_path, cfg["simulation_defaults"])
            db.upsert_tasks(tasks)
            print(f"Imported/updated {len(tasks)} alpha tasks from {csv_path}")
        elif args.command == "run":
            active = db.active_launch_rows()
            if active and not args.allow_active_launches:
                print(f"Backtest launch skipped: active tasks present ({len(active)} shown); use resume or --allow-active-launches to override.")
                return
            worker = BacktestWorker(cfg, db)
            worker.run_pending(limit=args.limit, id_prefix=args.id_prefix)
            export_reports(db, root / paths["results_csv"], root / paths["markdown_report"], cfg.get("selection"))
            print("Backtest run finished; reports exported.")
        elif args.command == "resume":
            worker = BacktestWorker(cfg, db)
            count = worker.resume_running()
            export_reports(db, root / paths["results_csv"], root / paths["markdown_report"], cfg.get("selection"))
            print(f"Resume finished; completed {count} active simulations.")
        elif args.command == "refit-generate":
            csv_path = Path(args.csv) if args.csv else root / paths["input_csv"]
            # Generate an oversized pool because many early variants may already
            # exist; append_unique_to_csv will keep only genuinely new rows.
            exprs = generate_refit_expressions(max(args.count * 20, args.count))
            added = append_refit_unique_to_csv(csv_path, exprs, cfg["simulation_defaults"], max_add=args.count)
            print(f"Generated {len(exprs)} refit candidates; added {added} to {csv_path}")
        elif args.command == "decorrelate-generate":
            csv_path = Path(args.csv) if args.csv else root / paths["input_csv"]
            exprs = generate_decorrelated_expressions(max(args.count * 20, args.count))
            added = append_decor_unique_to_csv(csv_path, exprs, cfg["simulation_defaults"], max_add=args.count)
            print(f"Generated {len(exprs)} decorrelated candidates; added {added} to {csv_path}")
        elif args.command == "report":
            export_reports(db, root / paths["results_csv"], root / paths["markdown_report"], cfg.get("selection"))
            print("Reports exported.")
        elif args.command == "check-auth":
            worker = BacktestWorker(cfg, db)
            info = worker.client.whoami()
            print(info or "Auth probe succeeded, but endpoint returned an empty response.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
