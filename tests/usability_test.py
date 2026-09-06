#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消安智答实用性测试脚本
自动测试问答功能，记录结果并生成报告
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path

class UsabilityTester:
    """实用性测试器"""

    def __init__(self, base_url: str = "http://localhost:8319"):
        self.base_url = base_url
        self.results = []
        self.test_cases = []
        self.passed = 0
        self.failed = 0

    def load_test_cases(self):
        """加载测试用例"""
        self.test_cases = [
            # ===== 1. 核心功能测试 =====
            {
                "id": "T1",
                "category": "法规问答",
                "question": "楼道堆放杂物违反什么规定",
                "expect_keywords": ["消防法", "第二十八条", "楼道"],
                "expect_not_keywords": [],
                "description": "应引用消防法具体条款"
            },
            {
                "id": "T2",
                "category": "应急指引",
                "question": "家里着火了现在怎么办",
                "expect_keywords": ["119", "逃生", "报警"],
                "expect_not_keywords": ["罚款", "处罚"],
                "description": "应给出应急指引，而非法规处罚"
            },
            {
                "id": "T3",
                "category": "宽问引导",
                "question": "我想了解一下消防方面的知识",
                "expect_keywords": ["可以问", "比如", "示例"],
                "expect_not_keywords": ["消防法第", "【法规依据】", "【结论】"],
                "description": "应引导提问，而非直接给法规"
            },
            {
                "id": "T4",
                "category": "安全拒答",
                "question": "今天天气怎么样",
                "expect_keywords": ["无法", "不涉及", "无关", "不回答"],
                "expect_not_keywords": ["晴天", "温度", "天气"],
                "description": "应礼貌拒答无关问题"
            },

            # ===== 2. 回答质量测试 =====
            {
                "id": "T5",
                "category": "可溯源性",
                "question": "电动车充电有什么规定",
                "expect_keywords": ["规定", "充电", "停放"],
                "expect_not_keywords": [],
                "check_evidence": True,
                "description": "回答应包含法规依据"
            },
            {
                "id": "T6",
                "category": "多轮对话",
                "question": "那具体罚多少钱",
                "context": "电动车充电有什么规定",
                "expect_keywords": ["罚款", "元", "处罚"],
                "expect_not_keywords": [],
                "description": "应结合上下文回答罚款金额"
            },

            # ===== 3. 边界测试 =====
            {
                "id": "T7",
                "category": "库外问题",
                "question": "《网络安全法》有什么规定",
                "expect_keywords": ["未收录", "不涉及", "无法", "超出范围"],
                "expect_not_keywords": ["网络安全法第", "网络"],
                "description": "应明确告知知识库外"
            },
            {
                "id": "T8",
                "category": "模糊问题",
                "question": "处罚",
                "expect_keywords": ["请问", "补充", "具体", "哪类"],
                "expect_not_keywords": [],
                "description": "应引导用户补充信息"
            },
            {
                "id": "T9",
                "category": "乱码处理",
                "question": "@@@@",
                "expect_keywords": ["无法理解", "请重新", "没听明白"],
                "expect_not_keywords": [],
                "description": "应友好提示无效输入"
            },
        ]
        print(f"📋 加载了 {len(self.test_cases)} 个测试用例")

    def ask(self, question: str, previous: str = None) -> Dict:
        """发送问答请求"""
        payload = {"question": question}
        if previous:
            payload["previous_question"] = previous

        try:
            resp = requests.post(
                f"{self.base_url}/api/ask",
                json=payload,
                timeout=60
            )
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            else:
                return {"success": False, "error": f"HTTP {resp.status_code}"}
        except requests.exceptions.Timeout:
            return {"success": False, "error": "请求超时"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": "服务未启动，请先运行 python3 main.py"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_answer(self, answer: str, keywords: List[str], not_keywords: List[str]) -> Tuple[bool, str]:
        """检查回答是否包含/不包含关键词"""
        if not answer:
            return False, "回答为空"

        # 检查必须包含的关键词
        if keywords:
            found = [kw for kw in keywords if kw in answer]
            if not found:
                return False, f"缺少预期关键词: {keywords}"

        # 检查不应包含的关键词
        if not_keywords:
            found = [kw for kw in not_keywords if kw in answer]
            if found:
                return False, f"包含不应有的关键词: {found}"

        return True, "通过"

    def run_test(self, test_case: Dict) -> Dict:
        """运行单个测试"""
        result = {
            "id": test_case["id"],
            "category": test_case["category"],
            "question": test_case["question"],
            "description": test_case["description"],
            "passed": False,
            "answer": "",
            "errors": [],
            "response_time": 0
        }

        # 发送请求
        start_time = time.time()
        resp = self.ask(
            test_case["question"],
            test_case.get("context")
        )
        elapsed = (time.time() - start_time) * 1000
        result["response_time"] = elapsed

        if not resp["success"]:
            result["errors"].append(resp["error"])
            return result

        answer = resp["data"].get("answer", "")
        result["answer"] = answer

        # 检查回答质量
        passed, msg = self.check_answer(
            answer,
            test_case.get("expect_keywords", []),
            test_case.get("expect_not_keywords", [])
        )

        if not passed:
            result["errors"].append(msg)
        else:
            result["passed"] = True

        # 额外检查：是否有证据链（如果是T5）
        # API 契约：证据链在 references；答案含【法规依据】也视为可溯源
        if test_case.get("check_evidence"):
            evidence = resp["data"].get("references") or resp["data"].get("citations") or []
            has_cite_block = "【法规依据】" in answer or "法规依据" in answer
            if not evidence and not has_cite_block:
                result["errors"].append("缺少证据链/引用来源")
                result["passed"] = False

        return result

    def run_all_tests(self):
        """运行所有测试"""
        self.load_test_cases()

        print("\n" + "="*70)
        print("🔥 消安智答实用性测试开始")
        print(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")

        for i, test in enumerate(self.test_cases, 1):
            print(f"[{i}/{len(self.test_cases)}] 测试 {test['id']}: {test['description']}")
            print(f"  问题: {test['question']}")

            result = self.run_test(test)
            self.results.append(result)

            # 实时显示结果
            if result["passed"]:
                print(f"  ✅ 通过 (响应: {result['response_time']:.0f}ms)")
                self.passed += 1
            else:
                print(f"  ❌ 失败")
                for err in result["errors"]:
                    print(f"     - {err}")
                self.failed += 1
            print()

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成测试报告"""
        total = self.passed + self.failed

        print("\n" + "="*70)
        print("📊 测试报告")
        print("="*70)
        print(f"  总测试数: {total}")
        print(f"  通过: {self.passed}")
        print(f"  失败: {self.failed}")
        print(f"  通过率: {self.passed/total*100:.1f}%")

        # 按分类统计
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = {"passed": 0, "total": 0}
            categories[cat]["total"] += 1
            if r["passed"]:
                categories[cat]["passed"] += 1

        print("\n  分类统计:")
        for cat, stats in categories.items():
            rate = stats["passed"]/stats["total"]*100
            print(f"    {cat}: {stats['passed']}/{stats['total']} ({rate:.0f}%)")

        # 保存详细报告
        report_file = Path("test_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": self.passed,
                    "failed": self.failed,
                    "pass_rate": self.passed/total*100
                },
                "details": self.results
            }, f, ensure_ascii=False, indent=2)

        print(f"\n  详细报告已保存: {report_file}")

        if self.failed == 0:
            print("\n🎉 所有测试通过！系统实用性良好")
        else:
            print(f"\n⚠️ 有 {self.failed} 项测试失败，请检查上述错误信息")

# ===== 额外工具：批量测试 =====
def batch_test_from_file(filepath: str):
    """从文件批量读取问题测试"""
    tester = UsabilityTester()

    with open(filepath, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"📋 从 {filepath} 读取了 {len(questions)} 个问题")

    for q in questions:
        print(f"\n问: {q}")
        resp = tester.ask(q)
        if resp["success"]:
            answer = resp["data"].get("answer", "")
            print(f"答: {answer[:150]}..." if len(answer) > 150 else f"答: {answer}")
        else:
            print(f"错: {resp['error']}")

# ===== 命令行入口 =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="消安智答实用性测试工具")
    parser.add_argument(
        "--url",
        default="http://localhost:8319",
        help="服务地址 (默认: http://localhost:8319)"
    )
    parser.add_argument(
        "--batch",
        help="批量测试：从文本文件读取问题列表"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速测试：只运行核心用例（前4个）"
    )

    args = parser.parse_args()

    tester = UsabilityTester(base_url=args.url)

    if args.batch:
        batch_test_from_file(args.batch)
    elif args.quick:
        tester.load_test_cases()
        tester.test_cases = tester.test_cases[:4]
        tester.run_all_tests()
    else:
        tester.run_all_tests()
