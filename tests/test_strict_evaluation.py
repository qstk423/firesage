import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "scripts"))

from e2e_local_test import hits_expected_article, summarize_timings


class StrictEvaluationTest(unittest.TestCase):
    def _response(self, citation):
        return {
            "structured": {"basis": [citation]},
            "references": [
                {
                    "article": "高层规定·第七条",
                    "law_name": "高层民用建筑消防安全管理规定",
                },
                {
                    "article": "消防法·第七条",
                    "law_name": "中华人民共和国消防法（2021修正）",
                },
            ],
        }

    def test_requires_law_name_and_article_number(self):
        row = {"expected_articles": ["高层规定·第七条"]}
        right = self._response(
            "《高层民用建筑消防安全管理规定》第七条：业主单位应当履行职责")
        wrong_law = self._response(
            "《中华人民共和国消防法（2021修正）》第七条：鼓励消防技术创新")
        self.assertTrue(hits_expected_article(row, right))
        self.assertFalse(hits_expected_article(row, wrong_law))

    def test_no_gold_is_not_applicable(self):
        self.assertIsNone(hits_expected_article({"expected_articles": []}, {}))

    def test_timing_summary_ignores_missing_values(self):
        rows = [
            {"timing": {"route_ms": 10, "retrieval": {"bm25_ms": 5}}},
            {"timing": {"route_ms": 20, "retrieval": {"bm25_ms": 7}}},
            {"timing": {}},
        ]
        summary = summarize_timings(rows)
        self.assertEqual(summary["route_ms"]["n"], 2)
        self.assertEqual(summary["retrieval.bm25_ms"]["max"], 7)


if __name__ == "__main__":
    unittest.main()
