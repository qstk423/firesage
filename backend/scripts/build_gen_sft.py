#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 FireEval + 法规全文构造生成式监督微调（SFT）数据。

每条样本：问题 + 给定条款原文 → 结构化 JSON 答案（仅依据给定条款）。
产物供后续 LoRA / QLoRA 使用；本仓库不强制本机训练大模型。

用法：
    python3 backend/scripts/build_gen_sft.py
    python3 backend/scripts/build_gen_sft.py --split train
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.pipeline import Pipeline  # noqa: E402

EVAL_DIR = os.path.join(BACKEND_DIR, "eval")
OUT_DIR = os.path.join(BACKEND_DIR, "data", "models", "sft")


def _load_articles(pipeline: Pipeline) -> dict:
    return pipeline.article_text


def _build_output(case: dict, articles: list[dict]) -> dict:
    """用金标 must + 条款原文构造忠实结构化答案（供 SFT 目标）。"""
    musts = case.get("must_conclusions") or []
    basis = []
    for art in articles:
        snippet = (art.get("text") or "")[:120].rstrip("。") + "。"
        basis.append(f"《{art.get('law_name') or art.get('law')}》{art['num'].split('·')[-1]}：{snippet}")
    if case.get("intent") == "emergency":
        return {
            "conclusion": "请立即拨打119，并按火灾应急指引疏散。",
            "conditions": "真火情 / 现场危险",
            "basis": ["国家消防救援局公共场所火灾应急处置指引"],
            "supplement": "紧急情况以现场指挥与119指令为准。",
            "confidence": "high",
        }
    if case.get("should_refuse") or case.get("intent") == "refuse":
        return {
            "conclusion": "该问题超出当前消防法规知识库范围，暂不回答以免误导。",
            "conditions": "知识库外 / 非消防主题",
            "basis": [],
            "supplement": "请改换消防合规或应急相关问题，或咨询属地消防救援机构。",
            "confidence": "high",
        }
    conclusion = "；".join(musts) if musts else (
        (articles[0]["text"][:80] + "…") if articles else "依据给定条款作答。"
    )
    return {
        "conclusion": conclusion,
        "conditions": "无特殊限制",
        "basis": basis[:3],
        "supplement": "请以官方条文及属地执行为准。",
        "confidence": "high" if articles else "low",
    }


def _format_context(articles: list[dict]) -> str:
    parts = []
    for art in articles:
        parts.append(f"【{art['num']} {art.get('title', '')}】{art.get('text', '')}")
    return "\n\n".join(parts) if parts else "（无可用条款）"


def build_split(name: str, article_map: dict) -> list[dict]:
    path = os.path.join(EVAL_DIR, f"fireeval_v1_{name}.json")
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    rows = []
    for case in cases:
        arts = []
        for key in case.get("expected_articles") or []:
            if key in article_map:
                arts.append(article_map[key])
        # 拒答/应急也可无条款
        if case.get("intent") == "law" and not arts and case.get("should_refuse"):
            pass
        output = _build_output(case, arts)
        row = {
            "id": case["id"],
            "split": name,
            "intent": case.get("intent"),
            "instruction": (
                "你是消防法规助手。请严格依据给定条款，输出 JSON："
                '{"conclusion","conditions","basis","supplement","confidence"}。'
                "不得编造未提供的条文。"
            ),
            "input": f"用户问题：{case['question']}\n\n依据条款：\n{_format_context(arts)}",
            "output": output,
            "output_text": json.dumps(output, ensure_ascii=False),
            "expected_articles": case.get("expected_articles") or [],
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    args = parser.parse_args()

    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        pipeline = Pipeline(json.load(f))
    article_map = _load_articles(pipeline)

    os.makedirs(OUT_DIR, exist_ok=True)
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    summary = {}
    for name in splits:
        rows = build_split(name, article_map)
        out_path = os.path.join(OUT_DIR, f"gen_sft_{name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = {"n": len(rows), "path": out_path}
        print(f"[sft] {name}: {len(rows)} → {out_path}")

    meta_path = os.path.join(OUT_DIR, "gen_sft_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"splits": summary, "format": "instruction-input-output JSONL for LoRA/QLoRA"},
                  f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
