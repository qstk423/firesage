#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时质量自检：LITE / 向量 / 精排 / LLM 是否就绪（试点 B5）。

用法：
    python3 backend/scripts/check_runtime.py
    python3 backend/scripts/check_runtime.py --ping-llm   # 真发一条极短补全（耗额度）
    python3 backend/scripts/check_runtime.py --strict     # full 档才 exit 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)


def _lite() -> bool:
    return os.getenv("FIRESAGE_LITE", "").strip() in ("1", "true", "True", "yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ping-llm", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = {
        "lite": _lite(),
        "llm_configured": False,
        "llm_enabled": False,
        "llm_ping": None,
        "vector_available": False,
        "vector_name": "",
        "cross_encoder": False,
        "quality_tier": "lite",
        "hints": [],
    }

    from rag.llm import LLMClient
    from rag.semantic_index import FIRESAGE_LITE, SemanticVectorIndex, CrossEncoderReranker

    llm = LLMClient()
    report["llm_configured"] = bool(llm.base_url and llm.api_key and llm.model)
    report["llm_enabled"] = bool(llm.enabled)
    if not llm.enabled:
        report["hints"].append("未配置 LLM_*：将走抽取式回答。复制 backend/.env.example → .env.local 填写。")

    if args.ping_llm and llm.enabled:
        try:
            out = llm.complete("只回复 ok", "ping")
            report["llm_ping"] = "ok" if out else "empty"
        except Exception as exc:
            report["llm_ping"] = f"error: {exc}"
            report["hints"].append("LLM ping 失败：检查网络与密钥。")

    # 用空语料探测模型能否加载（不建全量索引）
    if FIRESAGE_LITE or _lite():
        report["hints"].append("FIRESAGE_LITE=1：跳过 bge。全量请 unset FIRESAGE_LITE 或改用 docker-compose.full.yml。")
    else:
        vec = SemanticVectorIndex([])
        report["vector_available"] = bool(getattr(vec, "available", False))
        report["vector_name"] = getattr(vec, "name", "")
        if not report["vector_available"]:
            report["hints"].append("bge-m3 不可用：检查 HF 缓存或 HF_ENDPOINT。")
        ce = CrossEncoderReranker([])
        report["cross_encoder"] = bool(getattr(ce, "available", False))
        if not report["cross_encoder"]:
            report["hints"].append("精排模型不可用：将仅用法规意图重排。")

    lite = report["lite"] or _lite()
    if lite or (not report["vector_available"] and not report["llm_enabled"]):
        report["quality_tier"] = "lite"
    elif report["llm_enabled"] and report["vector_available"] and report["cross_encoder"]:
        report["quality_tier"] = "full"
    else:
        report["quality_tier"] = "partial"

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and report["quality_tier"] != "full":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
