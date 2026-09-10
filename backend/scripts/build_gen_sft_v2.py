#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 LoRA v2 训练数据：审计 + 定向增强（不修改 v1 数据与 test 集）。

针对 v1 评测报告的三个短板：
1. 审计所有 assistant 输出完整性（JSON 可解析、五字段齐全）
   与 basis 引用格式（《法规名》第X条：…）；
2. 增加知识库外问题的明确拒答样本（与 test 集 o04/o07/o14/n01/n09 不重叠）；
3. 增加紧急场景样本，首句直接给出可执行措施 + 119 引导
   （与 test 集 e08/e09/e15 场景不重叠）；
4. 统计 max_length=512 下的截断情况（供 v2 训练用）。

产物：backend/data/gen_sft_train_v2.jsonl
"""
from __future__ import annotations

import json
import os
import re

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_TRAIN = os.path.join(BACKEND_DIR, "data", "gen_sft_train.jsonl")
V2_TRAIN = os.path.join(BACKEND_DIR, "data", "gen_sft_train_v2.jsonl")
MODEL_DIR = os.path.join(BACKEND_DIR, "models", "Qwen2.5-3B-Instruct")

FIELDS = ["conclusion", "conditions", "basis", "supplement", "confidence"]
# 引用格式：《法规名》+ 条款号（法律条文"第X条"或标准数字条款"7.10"/"5.1.4"）+ 冒号
BASIS_PAT = re.compile(r"^《.+》(第[一二三四五六七八九十百零]+条|\d+(?:\.\d+)*)[：:]")
INSTRUCTION = (
    "你是消防法规助手。请严格依据给定条款，输出 JSON："
    '{"conclusion","conditions","basis","supplement","confidence"}。'
    "不得编造未提供的条文。"
)

# ---- 增强样本：知识库外拒答（与 test 集问题不重叠）----
REFUSE_QUESTIONS = [
    "消防员的招录条件和待遇是怎样的？",
    "我想考注册消防工程师，需要什么报考条件？",
    "消防工程专业哪些大学比较好？",
    "今天适合穿什么衣服出门？",
    "帮我写一首关于春天的诗",
    "明天股市会涨吗？",
    "红烧肉怎么做才好吃？",
    "高铁票退票手续费怎么算？",
    "小区物业费的收取标准是什么？",
    "家里空调不制冷了怎么维修？",
    "无人机航拍需要办理什么手续？",
    "食品经营许可证怎么办里？",
    "遇到网络诈骗应该去哪举报？",
    "哪个品牌的灭火器性价比高？",
    "有什么好看的消防主题电影推荐？",
]

# ---- 增强样本：紧急场景（首句=可执行措施+119；与 test 集场景不重叠）----
EMERGENCY_CASES = [
    ("家里炒菜的油锅突然着火了，火苗蹿得很高", None),
    ("半夜发现楼下电动车棚着火了，噼里啪啦在响", None),
    ("实验室酒精灯打翻了，实验台烧起来了", None),
    ("被子挨着电暖器烤焦冒烟了，屋里都是烟味", None),
    ("家里燃气热水器冒黑烟，天花板都熏黑了", None),
    ("楼道里闻到很浓的煤气味，头有点晕",
     {"conclusion": "请立即开窗通风、关闭燃气阀门，撤离到室外安全处拨打119或燃气抢修电话，切勿开关电器或使用明火。",
      "conditions": "燃气泄漏 / 有中毒风险"}),
    ("商场购物时火警警报响了，扶梯口有烟冒出来",
     {"conclusion": "请立即停止购物，按疏散指示标志就近撤离，切勿乘坐电梯扶梯，撤离到安全区域后拨打119。",
      "conditions": "公共场所火警 / 疏散撤离"}),
]

EMERGENCY_GOLD = {
    "conclusion": "请立即拨打119，并按火灾应急指引疏散。",
    "conditions": "真火情 / 现场危险",
    "basis": ["国家消防救援局公共场所火灾应急处置指引"],
    "supplement": "紧急情况以现场指挥与119指令为准。",
    "confidence": "high",
}
REFUSE_GOLD = {
    "conclusion": "该问题超出当前消防法规知识库范围，暂不回答以免误导。",
    "conditions": "知识库外 / 非消防主题",
    "basis": [],
    "supplement": "请改换消防合规或应急相关问题，或咨询属地消防救援机构。",
    "confidence": "high",
}


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def audit(rows: list[dict]) -> dict:
    """审计 assistant 输出完整性与 basis 引用格式。"""
    stats = {
        "total": len(rows), "json_ok": 0, "fields_ok": 0,
        "basis_fmt_checked": 0, "basis_fmt_ok": 0,
        "empty_basis": 0, "law_no_article": 0,
    }
    problems = []
    for r in rows:
        try:
            data = json.loads(r["output_text"])
            stats["json_ok"] += 1
        except Exception:
            problems.append(f"{r['id']}: output_text 不可解析为 JSON")
            continue
        if all(f in data for f in FIELDS):
            stats["fields_ok"] += 1
        else:
            problems.append(f"{r['id']}: 字段缺失")
        basis = data.get("basis")
        if not basis:
            stats["empty_basis"] += 1
            if r.get("intent") == "law":
                stats["law_no_article"] += 1
            continue
        if r.get("intent") in ("refuse", "emergency", "chitchat"):
            continue  # 这些金标的 basis 不适用《法规名》第X条格式
        stats["basis_fmt_checked"] += 1
        if isinstance(basis, list) and all(BASIS_PAT.match(str(b)) for b in basis):
            stats["basis_fmt_ok"] += 1
        else:
            problems.append(f"{r['id']}: basis 缺少《法规名》第X条引用格式")
    stats["problems"] = problems
    return stats


def make_row(id_: str, intent: str, question: str, gold: dict) -> dict:
    return {
        "id": id_, "split": "train_v2", "intent": intent,
        "instruction": INSTRUCTION,
        "input": f"用户问题：{question}\n\n依据条款：\n（无可用条款）",
        "output": gold,
        "output_text": json.dumps(gold, ensure_ascii=False),
        "expected_articles": [],
    }


def token_stats(rows: list[dict], max_length: int) -> dict:
    """统计 max_length 下的截断情况（答案优先策略：prompt 被截，答案保完整）。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    n, truncated, ans_truncated, prompt_kept_min = 0, 0, 0, 10 ** 9
    for r in rows:
        mp = [{"role": "system", "content": r["instruction"]},
              {"role": "user", "content": r["input"]}]
        mf = mp + [{"role": "assistant", "content": r["output_text"]}]
        pi = tok(tok.apply_chat_template(mp, add_generation_prompt=True, tokenize=False),
                 add_special_tokens=False)["input_ids"]
        fi = tok(tok.apply_chat_template(mf, add_generation_prompt=False, tokenize=False),
                 add_special_tokens=False)["input_ids"]
        comp = len(fi) - len(pi)
        n += 1
        if len(fi) > max_length:
            truncated += 1
            if comp > max_length - 1:
                ans_truncated += 1
            prompt_kept_min = min(prompt_kept_min, max(0, max_length - min(comp, max_length - 1)))
    return {"n": n, "truncated": truncated, "answer_truncated": ans_truncated,
            "min_prompt_kept_tokens": prompt_kept_min if truncated else None}


def main() -> None:
    rows = load_jsonl(V1_TRAIN)
    print(f"[载入] v1 训练集 {len(rows)} 条 ← {V1_TRAIN}")

    # ---- 1. 审计 ----
    stats = audit(rows)
    print("\n[审计] assistant 输出完整性与引用格式")
    print(f"  JSON 可解析        : {stats['json_ok']}/{stats['total']}")
    print(f"  五字段齐全         : {stats['fields_ok']}/{stats['total']}")
    print(f"  basis 引用格式合规 : {stats['basis_fmt_ok']}/{stats['basis_fmt_checked']}"
          f"（法条类样本；其余为拒答/应急/闲聊金标，不适用该格式）")
    print(f"  空 basis 样本      : {stats['empty_basis']}（其中 law 类无条款 {stats['law_no_article']} 条）")
    if stats["problems"]:
        print("  问题明细：")
        for p in stats["problems"]:
            print(f"    - {p}")
    else:
        print("  无格式问题，金标引用格式全部合规。")

    # ---- 2. 增强：拒答 + 紧急 ----
    aug = []
    for i, q in enumerate(REFUSE_QUESTIONS, 1):
        aug.append(make_row(f"syn_refuse_{i:02d}", "refuse", q, REFUSE_GOLD))
    for i, (q, override) in enumerate(EMERGENCY_CASES, 1):
        gold = dict(EMERGENCY_GOLD)
        if override:
            gold.update(override)
        aug.append(make_row(f"syn_emerg_{i:02d}", "emergency", q, gold))
    v2 = rows + aug
    print(f"\n[增强] 拒答 +{len(REFUSE_QUESTIONS)}，紧急 +{len(EMERGENCY_CASES)}"
          f" → v2 共 {len(v2)} 条（与 test 集零重叠）")

    # ---- 3. 长度统计（512 与 256 对照）----
    print("\n[长度] 答案优先截断策略下的统计")
    for ml in (512, 256):
        s = token_stats(v2, ml)
        extra = f"，最短 prompt 保留 {s['min_prompt_kept_tokens']} token" if s["truncated"] else ""
        print(f"  max_length={ml}: 截断 {s['truncated']}/{s['n']}"
              f"（其中答案本身被截 {s['answer_truncated']}）{extra}")

    # ---- 4. 保存 ----
    with open(V2_TRAIN, "w", encoding="utf-8") as f:
        for r in v2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    dist = Counter(r["intent"] for r in v2)
    print(f"\n[保存] v2 训练集 → {V2_TRAIN}")
    print(f"  intent 分布：{dict(dist)}")


if __name__ == "__main__":
    main()
