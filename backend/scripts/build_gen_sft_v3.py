#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 LoRA v3 训练数据：基于 v2，强化拒答纪律（不修改 v1/v2 数据与 test 集）。

针对 v2 评测暴露的拒答问题（n01 basis:["无"]、o07/n09 编造库外依据）：
1. 强制校验：所有知识库外样本（金标空依据）basis 必须严格为 []，
   出现 ["无"]/["未知"]/占位符即修复为 []（防御性，当前审计为 0 条）；
2. 新增 10 条知识库外拒答样本，与 test 集（n01/n09/o04/o07/o14 等）
   及 v2 已有 15 条拒答零重叠；
3. intent 分布更均衡：拒答类 20→30。

产物：backend/data/gen_sft_train_v3.jsonl
"""
from __future__ import annotations

import json
import os
import re

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_TRAIN = os.path.join(BACKEND_DIR, "data", "gen_sft_train_v2.jsonl")
V3_TRAIN = os.path.join(BACKEND_DIR, "data", "gen_sft_train_v3.jsonl")

INSTRUCTION = (
    "你是消防法规助手。请严格依据给定条款，输出 JSON："
    '{"conclusion","conditions","basis","supplement","confidence"}。'
    "不得编造未提供的条文。"
)

# 新增 10 条拒答问题（避开 v2 已有 15 条与 test 集全部问题）
NEW_REFUSE_QUESTIONS = [
    "公司注册资本最低需要多少钱？",
    "最近国际上有什么新闻大事？",
    "帮我订一张明天去北京的机票",
    "房贷利率现在是多少？",
    "世界杯哪支球队实力最强？",
    "个人所得税专项附加扣除都有哪些？",
    "怎么申请出国留学？",
    "推荐几本好看的小说",
    "手机内存不足该怎么清理？",
    "邻居装修噪音扰民应该找谁投诉？",
]

REFUSE_GOLD = {
    "conclusion": "该问题超出当前消防法规知识库范围，暂不回答以免误导。",
    "conditions": "知识库外 / 非消防主题",
    "basis": [],
    "supplement": "请改换消防合规或应急相关问题，或咨询属地消防救援机构。",
    "confidence": "high",
}

PLACEHOLDER_PAT = re.compile(r"^\s*(无|未知|暂无|N/?A|不适用)\s*$")


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def sanitize_basis(row: dict) -> bool:
    """知识库外样本 basis 强制 []；清除占位符。返回是否发生修复。"""
    gold = row["output"]
    basis = gold.get("basis") or []
    fixed = False
    # 占位符条目 → 视为空
    if basis and all(PLACEHOLDER_PAT.match(str(b)) for b in basis):
        gold["basis"] = []
        row["output_text"] = json.dumps(gold, ensure_ascii=False)
        fixed = True
    # 金标无依据（expected_articles 空 且 非应急/闲聊）但 basis 非空 → 清空
    if (not row.get("expected_articles")) and row.get("intent") != "emergency" and (gold.get("basis") or []):
        gold["basis"] = []
        row["output_text"] = json.dumps(gold, ensure_ascii=False)
        fixed = True
    return fixed


def main() -> None:
    rows = load_jsonl(V2_TRAIN)
    print(f"[载入] v2 训练集 {len(rows)} 条 ← {V2_TRAIN}")

    # 1. 强制校验与修复
    fixed = sum(1 for r in rows if sanitize_basis(r))
    n_empty = sum(1 for r in rows if not (r["output"].get("basis") or []))
    print(f"[校验] basis 占位符/违规修复 {fixed} 条；当前空 basis 样本 {n_empty} 条")

    # 2. 追加 10 条新拒答
    aug = []
    for i, q in enumerate(NEW_REFUSE_QUESTIONS, 1):
        aug.append({
            "id": f"syn_refuse2_{i:02d}", "split": "train_v3", "intent": "refuse",
            "instruction": INSTRUCTION,
            "input": f"用户问题：{q}\n\n依据条款：\n（无可用条款）",
            "output": dict(REFUSE_GOLD),
            "output_text": json.dumps(REFUSE_GOLD, ensure_ascii=False),
            "expected_articles": [],
        })
    v3 = rows + aug

    # 3. 保存
    with open(V3_TRAIN, "w", encoding="utf-8") as f:
        for r in v3:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = Counter(r["intent"] for r in v3)
    print(f"[保存] v3 训练集 → {V3_TRAIN}")
    print(f"  共 {len(v3)} 条，intent 分布：{dict(dist)}")


if __name__ == "__main__":
    main()
