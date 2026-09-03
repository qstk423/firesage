import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from rag.pipeline import Pipeline


class FireSageSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "backend", "data", "chunks.json"), encoding="utf-8") as source:
            cls.pipeline = Pipeline(json.load(source))

    def test_fire_lane_penalty(self):
        result = self.pipeline.ask("占用消防通道怎么处罚")
        self.assertFalse(result["refused"])
        self.assertEqual(result["references"][0]["article"], "消防法·第六十条")

    def test_emergency_route(self):
        self.assertEqual(self.pipeline.ask("家里着火了现在怎么办")["intent"], "emergency")

    def test_out_of_scope_refusal(self):
        self.assertTrue(self.pipeline.ask("红烧肉怎么做")["refused"])


if __name__ == "__main__":
    unittest.main()
