import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from rag.intent import route
from rag.pipeline import Pipeline
from rag.scene import structure
from rag.verifier import _cn_to_int, verify


class IntentRouteTest(unittest.TestCase):
    """真应急 vs 法规咨询：避免“火灾隐患…怎么办”误入应急。"""

    def test_true_emergencies(self):
        for q in [
            "家里着火了现在怎么办",
            "家里厨房着火了怎么办！",
            "楼下电动车充电着火了，火很大",
            "地铁车厢里着火了怎么逃生",
            "楼道里冒烟很大，我现在该怎么逃生？",
            "房间里全是烟出不去，救命",
            "宿舍楼发生火灾，现场有爆炸声",
        ]:
            self.assertEqual(route(q), "emergency", q)

    def test_law_not_emergency(self):
        for q in [
            "火灾隐患不能确保消防安全时单位应当怎么办？",
            "查出火灾隐患了但是一时半会改不掉，该怎么办？",
            "公司从来不组织消防演练，员工都不知道着火了往哪跑",
            "写字楼外面的大广告牌把窗户都挡住了，出了火灾跑不出去怎么办？",
            "着火点怎么确定",
            "消防救援机构监督检查中发现火灾隐患应当怎么处理？",
            "单位对存在的火灾隐患应当怎么整改？",
            "报警有什么规定",
            "单位发生火灾时应当怎么做？",
            "因消防安全责任不落实发生较大火灾事故会追究谁的责任？",
            "火灾发生时高层公共建筑内谁负责组织引导人员疏散？",
            "高层民用建筑施工期间发生火灾，消防安全责任在谁？",
            "单位发生火灾后第一时间应当做什么？可以不报警先自己灭火吗？",
        ]:
            self.assertEqual(route(q), "law", q)


class SceneTest(unittest.TestCase):
    def test_colloquial_scene(self):
        s = structure("楼梯口让纸箱堵得过不去了")
        self.assertIn("占用疏散通道", s["behaviors"])
        self.assertIn("疏散通道", s["objects"])

    def test_penalty_channel_clarify(self):
        s = structure("占用通道罚多少钱")
        self.assertIn("channel_type", s["ambiguity"])
        self.assertIn("subject", s["ambiguity"])

    def test_no_false_clarify_when_specified(self):
        s = structure("个人占用疏散通道怎么处罚")
        self.assertEqual(s["ambiguity"], [])

    def test_rewrite(self):
        s = structure("楼道堆放杂物违反什么规定")
        self.assertIn("疏散通道", s["rewrite"])


class VerifierTest(unittest.TestCase):
    def test_cn_to_int(self):
        self.assertEqual(_cn_to_int("五千"), 5000)
        self.assertEqual(_cn_to_int("五万"), 50000)
        self.assertEqual(_cn_to_int("二十万"), 200000)
        self.assertEqual(_cn_to_int("一百二十三"), 123)
        self.assertEqual(_cn_to_int("2000"), 2000)

    def test_fine_consistency(self):
        refs = [{"article": "消防法·第六十条", "text": "责令改正，处五千元以上五万元以下罚款"}]
        ok = verify("根据消防法第六十条，对单位处五千元以上五万元以下罚款。", refs, "单位占用疏散通道怎么处罚")
        self.assertTrue(ok["passed"])
        bad = verify("根据消防法第六十条，对单位处一万元以上十万元以下罚款。", refs, "单位占用疏散通道怎么处罚")
        self.assertFalse(bad["passed"])

    def test_fabricated_citation(self):
        refs = [{"article": "消防法·第六十条", "text": "责令改正，处五千元以上五万元以下罚款"}]
        from rag.verifier import set_article_keys
        set_article_keys({"消防法·第六十条", "消防法·第七十二条"})
        result = verify("依据消防法第七十二条处五千元罚款。", refs, "单位占用疏散通道怎么处罚")
        self.assertFalse(result["passed"])
        self.assertTrue(any("不在检索依据" in i for i in result["issues"]))


class FireSageSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(ROOT, "backend", "data", "chunks.json"), encoding="utf-8") as source:
            cls.pipeline = Pipeline(json.load(source))

    def test_fire_lane_penalty(self):
        result = self.pipeline.ask("单位占用消防车通道怎么处罚")
        self.assertFalse(result["refused"])
        self.assertEqual(result["references"][0]["article"], "消防法·第六十条")

    def test_ambiguous_penalty_clarifies(self):
        result = self.pipeline.ask("占用通道罚多少钱")
        self.assertEqual(result["intent"], "clarify")
        self.assertIn("通道", result["answer"])

    def test_killer_case(self):
        result = self.pipeline.ask("楼道堆放杂物违反什么规定")
        self.assertFalse(result["refused"])
        self.assertIn(result["references"][0]["article"],
                      ("消防法·第二十八条", "消防法·第六十条", "高层规定·第二十八条",
                       "61号令·第二十一条"))

    def test_structured_answer_shape(self):
        result = self.pipeline.ask("消防控制室必须24小时值班吗")
        self.assertFalse(result["refused"])
        self.assertIn("conclusion", result["structured"])
        self.assertIn("【结论】", result["answer"])
        self.assertIn("stages", result)
        self.assertIn("latency_ms", result)

    def test_multi_turn(self):
        first = self.pipeline.ask("单位占用疏散通道怎么处罚")
        follow = self.pipeline.ask("那具体罚多少钱", "单位占用疏散通道怎么处罚")
        self.assertTrue(follow["context_used"])
        self.assertFalse(follow["refused"])

    def test_emergency_route(self):
        self.assertEqual(self.pipeline.ask("家里着火了现在怎么办")["intent"], "emergency")

    def test_out_of_scope_refusal(self):
        self.assertTrue(self.pipeline.ask("红烧肉怎么做")["refused"])


if __name__ == "__main__":
    unittest.main()
