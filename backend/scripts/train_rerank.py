#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练条款重排微调分类器（可离线复现）。

正样本：FireEval 问题 × 金标条款
负样本：同题下检索召回的非金标条款（不足则随机抽其它条款）

用法：
    python3 backend/scripts/train_rerank.py
    python3 backend/scripts/train_rerank.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.pipeline import Pipeline as RagPipeline  # noqa: E402
from rag.rerank_ml import MODEL_DIR, RERANK_MODEL_PATH, _pair_text  # noqa: E402

EVAL_DIR = os.path.join(BACKEND_DIR, "eval")
random.seed(42)


def _load_cases(name: str) -> list[dict]:
    with open(os.path.join(EVAL_DIR, f"fireeval_v1_{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _article_blob(pipeline: RagPipeline, article_id: str) -> str:
    art = pipeline.article_text.get(article_id) or {}
    return f"{art.get('title', '')} {art.get('text', '')}".strip()


def _build_pairs(pipeline: RagPipeline, cases: list[dict], hard_neg_k: int = 4):
    X, y = [], []
    all_ids = list(pipeline.article_text.keys())
    for case in cases:
        if case.get("intent") != "law" or case.get("should_refuse"):
            continue
        gold = [a for a in (case.get("expected_articles") or []) if a in pipeline.article_text]
        if not gold:
            continue
        q = case["question"]
        for gid in gold:
            X.append(_pair_text(q, _article_blob(pipeline, gid)))
            y.append(1)
        # 硬负例：当前检索 top 非金标
        ranked, _ = pipeline.retriever.retrieve(q, top_k=8, mode="hybrid")
        negs = [c["article"] for c in ranked if c["article"] not in gold]
        while len(negs) < hard_neg_k:
            cand = random.choice(all_ids)
            if cand not in gold and cand not in negs:
                negs.append(cand)
        for nid in negs[:hard_neg_k]:
            X.append(_pair_text(q, _article_blob(pipeline, nid)))
            y.append(0)
    return X, y


def _clf() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=40000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.5,
            solver="lbfgs",
        )),
    ])


def _hit_at_k(pipeline: RagPipeline, cases: list[dict], k: int = 1, use_ml: bool = True) -> float:
    """在 full 检索上估 Hit@k（可选关闭 ML 重排对比）。"""
    from rag import rerank_ml as rm
    old_clf, old_loaded = rm._clf, rm._loaded
    old_env = os.environ.get("RERANK_ML")
    try:
        if use_ml:
            os.environ["RERANK_ML"] = "1"
            rm._clf = old_clf if old_clf is not None else rm._load()
            rm._loaded = True
        else:
            os.environ["RERANK_ML"] = "0"
            rm._clf = None
            rm._loaded = True
        ok = total = 0
        for case in cases:
            gold = set(case.get("expected_articles") or [])
            if not gold or case.get("intent") != "law" or case.get("should_refuse"):
                continue
            total += 1
            ranked, _ = pipeline.retriever.retrieve(case["question"], top_k=max(3, k), mode="full")
            arts = [c["article"] for c in ranked[:k]]
            if k == 1:
                if arts and arts[0] in gold:
                    ok += 1
            else:
                if gold.intersection(arts):
                    ok += 1
        return ok / total if total else 0.0
    finally:
        rm._clf, rm._loaded = old_clf, old_loaded
        if old_env is None:
            os.environ.pop("RERANK_ML", None)
        else:
            os.environ["RERANK_ML"] = old_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="对比开启/关闭微调重排的 Hit@1")
    args = parser.parse_args()

    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        pipeline = RagPipeline(json.load(f))

    train_cases = _load_cases("train")
    X, y = _build_pairs(pipeline, train_cases)
    print(f"[train] pairs={len(X)} pos={sum(y)} neg={len(y) - sum(y)}")

    model = _clf()
    model.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, RERANK_MODEL_PATH)
    print(f"[save] {RERANK_MODEL_PATH}")

    # 重新加载到模块
    from rag import rerank_ml
    rerank_ml._clf = model
    rerank_ml._loaded = True

    metrics = {
        "train_pairs": len(X),
        "pos": int(sum(y)),
        "neg": int(len(y) - sum(y)),
        "model_path": RERANK_MODEL_PATH,
        "train_acc": float(model.score(X, y)),
    }

    if args.compare:
        compare = []
        for split in ("dev", "test"):
            cases = _load_cases(split)
            # 需要先有模型文件；对比 off/on
            hit_off = _hit_at_k(pipeline, cases, k=1, use_ml=False)
            hit_on = _hit_at_k(pipeline, cases, k=1, use_ml=True)
            hit3_on = _hit_at_k(pipeline, cases, k=3, use_ml=True)
            row = {
                "split": split,
                "n_law": sum(1 for c in cases if c.get("intent") == "law" and not c.get("should_refuse") and c.get("expected_articles")),
                "hit1_rules_only": round(hit_off, 4),
                "hit1_with_ml": round(hit_on, 4),
                "hit3_with_ml": round(hit3_on, 4),
            }
            compare.append(row)
            print(row)
        metrics["compare"] = compare
        with open(os.path.join(MODEL_DIR, "rerank_compare.json"), "w", encoding="utf-8") as f:
            json.dump({"compare": compare}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODEL_DIR, "rerank_clf_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
