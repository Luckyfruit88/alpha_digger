from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.submitted_similarity import SIMILARITY_WEIGHTS, build_features, expression_features, load_submitted_features, score_against_submitted


class SubmittedSimilarityTests(unittest.TestCase):
    def test_extracts_fields_operators_windows_lineages(self):
        feats = expression_features("rank(ts_corr(close, volume, 20)) + group_neutralize(returns, industry)")
        self.assertIn("close", feats["fields"])
        self.assertIn("volume", feats["fields"])
        self.assertIn("returns", feats["fields"])
        self.assertIn("ts_corr", feats["operators"])
        self.assertIn(20, feats["windows"])
        self.assertIn("price_return", feats["lineages"])
        self.assertIn("correlation", feats["lineages"])

    def test_similarity_higher_for_related_expression(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lib = root / "lib.jsonl"
            out = root / "features.json"
            report = root / "report.md"
            row = {
                "alpha_id": "submitted_a",
                "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                "expression": "rank(ts_corr(close, volume, 20)) + rank(returns)",
                "date_submitted": "2026-04-30T00:00:00-04:00",
            }
            lib.write_text(json.dumps(row) + "\n", encoding="utf-8")
            build_features(lib, out, report)
            feats = load_submitted_features(out)
            related = score_against_submitted("rank(ts_corr(close, adv20, 20)) + rank(returns)", {"region": "USA", "universe": "TOP3000", "delay": 1}, feats)
            unrelated = score_against_submitted("ts_zscore(debt_to_assets, 120) - rank(market_cap)", {"region": "USA", "universe": "TOP3000", "delay": 1}, feats)
            self.assertGreater(related["max_similarity"], unrelated["max_similarity"])
            self.assertIn(related["collision_level"], {"weak", "medium", "high"})

    def test_similarity_weights_prioritize_lineage_and_skeleton(self):
        self.assertEqual(SIMILARITY_WEIGHTS["lineage_overlap"], 0.35)
        self.assertEqual(SIMILARITY_WEIGHTS["skeleton_sequence_similarity"], 0.20)
        self.assertEqual(SIMILARITY_WEIGHTS["settings_similarity"], 0.00)
        self.assertAlmostEqual(sum(SIMILARITY_WEIGHTS.values()), 1.0)

    def test_new_weights_keep_known_blocked_cases_detectable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lib = root / "lib.jsonl"
            out = root / "features.json"
            report = root / "report.md"
            submitted_expr = "group_neutralize(rank(ts_rank(earnings_momentum_analyst_score, 120) + ts_corr(close, volume, 20) - ts_rank(returns, 252)), industry)"
            rows = [
                {
                    "alpha_id": "kq1kVqP8",
                    "settings": {"region": "USA", "universe": "TOP3000", "delay": 1},
                    "expression": submitted_expr,
                    "date_submitted": "2026-04-30T00:00:00-04:00",
                }
            ]
            lib.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
            build_features(lib, out, report)
            feats = load_submitted_features(out)
            known_blocked = [
                "group_neutralize(rank(ts_rank(earnings_momentum_analyst_score, 120) + ts_corr(close, adv20, 20) - ts_rank(returns, 252)), industry)",
                "group_neutralize(rank(ts_zscore(earnings_momentum_analyst_score, 120) + ts_corr(vwap, volume, 20) - ts_rank(returns, 252)), industry)",
            ]
            for expr in known_blocked:
                scored = score_against_submitted(expr, {"region": "USA", "universe": "TOP3000", "delay": 1}, feats)
                self.assertGreaterEqual(scored["max_similarity"], 0.65)


if __name__ == "__main__":
    unittest.main()
