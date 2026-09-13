import os
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "scripts"))

from serve_local_qwen import (  # noqa: E402
    CompleteJSONObjectStoppingCriteria,
    CONCISE_SUFFIX,
    select_generation_budget,
)


class _PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[int(i)] for i in list(ids))


class LocalQwenLatencyTest(unittest.TestCase):
    def test_dynamic_budgets(self):
        env = {
            "LOCAL_LLM_SIMPLE_TOKENS": "340",
            "LOCAL_LLM_COMPLEX_TOKENS": "420",
            "LOCAL_LLM_EMERGENCY_TOKENS": "220",
        }
        with patch.dict(os.environ, env, clear=False):
            self.assertEqual(
                select_generation_budget("占用消防通道怎么处罚", ["消防法·第六十条"], 480, 480),
                (340, "simple"),
            )
            self.assertEqual(
                select_generation_budget("分别说明单位责任以及处罚流程", ["消防法·第六十条"], 480, 480),
                (420, "complex"),
            )
            self.assertEqual(
                select_generation_budget("厨房着火了怎么办", [], 480, 480),
                (220, "emergency"),
            )
            self.assertEqual(
                select_generation_budget("普通问题", ["消防法·第六十条"], 200, 480),
                (200, "simple"),
            )

    def test_concise_contract(self):
        self.assertIn("basis 最多2条", CONCISE_SUFFIX)
        self.assertIn("完整 JSON", CONCISE_SUFFIX)

    def test_stops_at_complete_json_not_brace_inside_string(self):
        pieces = ["说明", '{"conclusion":"含有{符号}",', '"basis":[]', "}", "多余文本"]
        tok = _PieceTokenizer(pieces)
        stopper = CompleteJSONObjectStoppingCriteria(tok, prompt_length=2)
        generated = []
        results = []
        for token_id in range(len(pieces)):
            generated.append(token_id)
            ids = np.array([[90, 91, *generated]])
            results.append(bool(stopper(ids, None)))
        self.assertEqual(results, [False, False, False, True, False])


if __name__ == "__main__":
    unittest.main()
