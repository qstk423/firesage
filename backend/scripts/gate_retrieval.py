#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索-only Hit 门禁（不跑 LLM 生成）：防 PR 默默打烂召回。

用法：
    python3 backend/scripts/gate_retrieval.py --split dev
    python3 backend/scripts/gate_retrieval.py --split test --strict
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.retriever import HybridRetriever  # noqa: E402

THRESHOLDS = {
    "hit_at_1": 0.80,
    "hit_at_3": 0.93,
}


def _load_ablation():
    path = os.path.join(BACKEND_DIR, "scripts", "ablation_retrieval.py")
    spec = importlib.util.spec_from_file_location("ablation_retrieval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="dev")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--mode", default="full",
                        choices=["bm25", "bm25_vector", "hybrid", "full"])
    args = parser.parse_args()

    abl = _load_ablation()
    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(abl._dataset_path(args.split), encoding="utf-8") as f:
        cases = json.load(f)

    t0 = time.time()
    retriever = HybridRetriever(chunks)
    metrics = abl._metrics_for_mode(retriever, cases, args.mode)
    metrics["split"] = args.split
    metrics["mode"] = args.mode
    metrics["degraded_vector"] = bool(getattr(retriever, "degraded_vector", False))
    metrics["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    if args.strict:
        ok = True
        for key, bound in THRESHOLDS.items():
            val = metrics.get(key, 0)
            passed = val >= bound
            mark = "PASS" if passed else "FAIL"
            print(f"[{mark}] {key}: {val} (bound: {bound})")
            ok = ok and passed
        if not ok:
            raise SystemExit(1)
        print("检索门禁通过。")


if __name__ == "__main__":
    main()
