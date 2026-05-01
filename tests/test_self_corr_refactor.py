from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.auto_submit import should_allow_d1_detail, should_allow_exploration_detail, should_allow_fde_detail
from scripts.ml_candidate_scorer import field_diversity_index, p_self_corr_histogram
from scripts.self_corr_truth_table import build_truth_table, classify_lineage


def create_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table backtest_results (
          alpha_id text primary key,
          expression text not null,
          status text not null,
          simulation_id text,
          sharpe real,
          fitness real,
          turnover real,
          returns real,
          drawdown real,
          margin real,
          long_count integer,
          short_count integer,
          checks_passed integer,
          fail_reasons text,
          raw_json text,
          error text,
          created_at text
        )
        """
    )
    conn.execute(
        """
        create table alpha_tasks (
          id text primary key,
          expression text not null,
          settings_json text not null,
          status text not null default 'PENDING',
          attempts integer not null default 0,
          simulation_id text,
          last_error text,
          updated_at text default current_timestamp
        )
        """
    )
    rows = [
        ("pred_1", "rank(earnings_momentum_analyst_score)", None),
        ("pend_1", "rank(change_in_eps_surprise)", None),
        ("cool_1", "rank(ts_rank(close, 120) - ts_rank(volume, 120))", {"id": "WQCOOL"}),
        ("clear_1", "rank(ts_rank(assets, 120) - ts_rank(cap, 120))", {"id": "WQCLEAR"}),
    ]
    for alpha_id, expr, raw in rows:
        conn.execute(
            """
            insert into backtest_results
            (alpha_id, expression, status, sharpe, fitness, turnover, fail_reasons, raw_json, created_at)
            values (?, ?, 'COMPLETE', 1.8, 1.1, 0.2, '', ?, '2026-04-30T00:00:00+00:00')
            """,
            (alpha_id, expr, json.dumps(raw) if raw else None),
        )
    conn.commit()
    conn.close()


class SelfCorrRefactorTests(unittest.TestCase):
    def test_truth_table_classifies_self_corr_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "backtests.sqlite3"
            create_db(db)
            auto_log = root / "auto_submit.log"
            auto_log.write_text(
                "\n".join(
                    [
                        json.dumps({"skipped": [["pred_1", "pre-submit self-corr gate: p_self_corr_block=0.305 > 0.20"]]}),
                        json.dumps({"skipped": [["pend_1", "unsafe checks: SELF_CORRELATION:PENDING"]]}),
                    ]
                ),
                encoding="utf-8",
            )
            submissions = root / "auto_submissions.jsonl"
            submissions.write_text(json.dumps({"local_id": "clear_1", "alpha_id": "WQCLEAR", "status": "already_submitted"}) + "\n", encoding="utf-8")
            submit_state = root / "auto_submit_state.json"
            submit_state.write_text(json.dumps({"cooldowns": {"WQCOOL": "2026-04-30T00:00:00+00:00"}}), encoding="utf-8")

            payload = build_truth_table(
                db_path=db,
                auto_submit_log=auto_log,
                submissions_path=submissions,
                auto_submit_state_path=submit_state,
                super_state_path=root / "missing_super.json",
                d1_state_path=root / "missing_d1.json",
                multi_state_path=root / "missing_multi.json",
                d1_truth_table_path=root / "missing_d1_truth.json",
                repair_state_path=root / "missing_repair.json",
            )
            rows = {row["alpha_id"]: row for row in payload["alphas"]}
            self.assertEqual(rows["pred_1"]["self_corr_status"], "predicted_blocked")
            self.assertEqual(rows["pend_1"]["self_corr_status"], "pending")
            self.assertEqual(rows["cool_1"]["self_corr_status"], "cooldown")
            self.assertEqual(rows["clear_1"]["self_corr_status"], "clear")

    def test_truth_table_computes_repair_depth_from_repair_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "backtests.sqlite3"
            create_db(db)
            conn = sqlite3.connect(db)
            chain = [
                ("repairsc_a", "pred_1"),
                ("repairsc2_b", "repairsc_a"),
                ("repairsc3_c", "repairsc2_b"),
            ]
            for alpha_id, _ in chain:
                conn.execute(
                    """
                    insert into backtest_results
                    (alpha_id, expression, status, sharpe, fitness, turnover, fail_reasons, raw_json, created_at)
                    values (?, 'rank(assets)', 'COMPLETE', 1.8, 1.1, 0.2, '', null, '2026-04-30T00:00:00+00:00')
                    """,
                    (alpha_id,),
                )
            conn.commit()
            conn.close()
            repair_state = root / "repair_state.json"
            repair_state.write_text(json.dumps({"seeded_from": dict(chain)}), encoding="utf-8")
            payload = build_truth_table(
                db_path=db,
                auto_submit_log=root / "missing.log",
                submissions_path=root / "missing.jsonl",
                auto_submit_state_path=root / "missing_submit.json",
                super_state_path=root / "missing_super.json",
                d1_state_path=root / "missing_d1.json",
                multi_state_path=root / "missing_multi.json",
                d1_truth_table_path=root / "missing_d1_truth.json",
                repair_state_path=repair_state,
            )
            rows = {row["alpha_id"]: row for row in payload["alphas"]}
            self.assertEqual(rows["repairsc3_c"]["repair_depth"], 3)
            self.assertEqual(payload["summary"]["repair_depth_distribution"]["3+"], 1)

    def test_lineage_classifier_detects_core_themes(self) -> None:
        self.assertEqual(classify_lineage("rank(change_in_eps_surprise)")["theme"], "analyst_earnings")
        self.assertEqual(classify_lineage("rank(ts_rank(close, 120) - ts_rank(volume, 120))")["theme"], "price_volume")
        self.assertEqual(classify_lineage("rank(ts_std_dev(returns, 20) / adv20)")["theme"], "liquidity_volatility")

    def test_exploration_detail_budget_does_not_bypass_limits(self) -> None:
        allow, reason = should_allow_exploration_detail(
            {"exploration_candidate": True, "exploration_reason": "quality_pass under-sampled lineage"},
            p_self_corr=0.55,
            max_self_corr_risk=0.20,
            max_exploration_p_self_corr=0.75,
            exploration_checks_used=0,
            exploration_check_budget=2,
        )
        self.assertTrue(allow)
        self.assertIn("under-sampled", reason)

        allow, reason = should_allow_exploration_detail(
            {"exploration_candidate": True},
            p_self_corr=0.80,
            max_self_corr_risk=0.20,
            max_exploration_p_self_corr=0.75,
            exploration_checks_used=0,
            exploration_check_budget=2,
        )
        self.assertFalse(allow)
        self.assertIn(">", reason)

        allow, reason = should_allow_exploration_detail(
            {"exploration_candidate": True},
            p_self_corr=0.55,
            max_self_corr_risk=0.20,
            max_exploration_p_self_corr=0.75,
            exploration_checks_used=2,
            exploration_check_budget=2,
        )
        self.assertFalse(allow)
        self.assertIn("budget", reason)

    def test_fde_and_d1_detail_gates_are_label_only_candidates(self) -> None:
        allow, reason = should_allow_fde_detail({"fde_candidate": True, "lineage_theme": "market_size"}, 0, 2, True)
        self.assertTrue(allow)
        self.assertIn("forced diversity", reason)

        allow, reason = should_allow_fde_detail({"fde_candidate": False}, 0, 2, True)
        self.assertFalse(allow)
        self.assertIn("fde_candidate", reason)

        allow, reason = should_allow_d1_detail(
            "d1v23_abc",
            {"pass_quality": 1},
            p_self_corr=0.34,
            checks_used=0,
            check_budget=2,
            max_p_self_corr=0.35,
            enabled=True,
        )
        self.assertTrue(allow)
        self.assertIn("D1 relaxed", reason)

        allow, reason = should_allow_d1_detail(
            "d1v23_abc",
            {"pass_quality": 1},
            p_self_corr=0.36,
            checks_used=0,
            check_budget=2,
            max_p_self_corr=0.35,
            enabled=True,
        )
        self.assertFalse(allow)
        self.assertIn(">", reason)

    def test_scorer_structural_summaries(self) -> None:
        self.assertGreaterEqual(field_diversity_index("rank(ts_rank(assets, 120) - ts_rank(cap, 120))", ["revenue"]), 3)
        self.assertEqual(
            p_self_corr_histogram([
                {"p_self_corr_block": 0.1},
                {"p_self_corr_block": 0.3},
                {"p_self_corr_block": 0.5},
                {"p_self_corr_block": 0.7},
                {"p_self_corr_block": 0.9},
            ]),
            {"0-0.2": 1, "0.2-0.4": 1, "0.4-0.6": 1, "0.6-0.8": 1, "0.8-1.0": 1},
        )


if __name__ == "__main__":
    unittest.main()
