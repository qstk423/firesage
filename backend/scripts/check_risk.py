# -*- coding: utf-8 -*-
"""风险分级验收（B3）：10 条固定题，纯规则判定，毫秒级完成、结果稳定可复现。

用法：
    python check_risk.py
只调 structure + route + assess（不走检索/LLM），因此标签与线上 /api/ask 的
risk 字段同源（pipeline 中 risk 同样在检索前计算）。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.intent import route
from rag.risk import assess
from rag.scene import structure

# (问题, 期望等级) —— 覆盖 紧急/高/中/低 四级
CASES = [
    ("家里着火了现在怎么办", "emergency"),
    ("楼里着火了浓烟很大我被困在卧室", "emergency"),
    ("楼道被纸箱堵死了过不去", "high"),
    ("电动车飞线充电违法吗", "high"),
    ("私家车占用消防车通道怎么处罚", "high"),
    ("单位不搞消防演练违反什么规定", "medium"),
    ("消防控制室没人值班", "medium"),
    ("消防安全职责有哪些", "low"),
    ("高层民用建筑的定义", "low"),
    ("物业消防巡查多久一次", "low"),
]


def main():
    ok = True
    print(f"{'结果':<6}{'期望':<12}{'实际':<12}短因 / 问题")
    for q, expected in CASES:
        scene = structure(q)
        intent = route(q)
        # 与 pipeline._ask 同步的兜底：拒答词表未命中但场景要素齐全 → 按法规问
        if intent == "refuse" and (scene.get("behaviors") or scene.get("objects")):
            intent = "law"
        risk = assess(q, scene, intent)
        got = risk["risk_level"]
        mark = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"{mark:<7}{expected:<12}{got:<12}{risk['risk_reasons']}  {q}")
    print("\n结论：", "全部通过" if ok else "存在失败用例")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
