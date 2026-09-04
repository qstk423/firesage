#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索消融：对比 BM25 / +向量 / 三路融合 / 全链路重排 的 Hit@1、Hit@3、MRR。

用法：
    python3 scripts/ablation_retrieval.py --split dev
    python3 scripts/ablation_retrieval.py --split all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.retriever import HybridRetriever  # noqa: E402

MODES = (
    ("bm25", "仅 BM25"),
    ("bm25_vector", "BM25 + 向量 RRF"),
    ("hybrid", "三路 RRF（无重排）"),
    ("full", "三路 RRF + 法规重排 + CE"),
)


def _dataset_path(split: str) -> str:
    if split == "all":
        return os.path.join(BACKEND_DIR, "eval", "fireeval_v1.json")
    return os.path.join(BACKEND_DIR, "eval", f"fireeval_v1_{split}.json")


def _metrics_for_mode(retriever: HybridRetriever, cases: list[dict], mode: str) -> dict:
    retrieval_cases = hit1 = hit3 = 0
    reciprocal_rank = 0.0
    t0 = time.time()
    for case in cases:
        expected = set(case.get("expected_articles") or [])
        if not expected or case.get("should_refuse"):
            continue
        if case.get("intent") == "emergency":
            continue
        retrieval_cases += 1
        ranked, _ = retriever.retrieve(case["question"], top_k=5, mode=mode)
        arts = [item["article"] for item in ranked]
        if arts and arts[0] in expected:
            hit1 += 1
        if expected.intersection(arts[:3]):
            hit3 += 1
        first_rank = next((i for i, a in enumerate(arts, 1) if a in expected), None)
        if first_rank:
            reciprocal_rank += 1 / first_rank
    n = max(1, retrieval_cases)
    return {
        "mode": mode,
        "retrieval_cases": retrieval_cases,
        "hit_at_1": round(hit1 / n, 4),
        "hit_at_3": round(hit3 / n, 4),
        "mrr": round(reciprocal_rank / n, 4),
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    args = parser.parse_args()

    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(_dataset_path(args.split), encoding="utf-8") as f:
        cases = json.load(f)

    retriever = HybridRetriever(chunks)
    rows = []
    for mode, label in MODES:
        row = _metrics_for_mode(retriever, cases, mode)
        row["label"] = label
        rows.append(row)
        print(f"[{mode}] Hit@1={row['hit_at_1']:.4f} Hit@3={row['hit_at_3']:.4f} "
              f"MRR={row['mrr']:.4f} n={row['retrieval_cases']} ({row['elapsed_s']}s) — {label}")

    print(json.dumps({"split": args.split, "ablation": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
