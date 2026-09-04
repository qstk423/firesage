#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练消防领域意图分类器（小规模微调实验）。

数据：FireEval train + 少数类增强样本
模型：字符 n-gram TF-IDF + LogisticRegression（可离线、可复现）

用法：
    python3 scripts/train_intent.py
    python3 scripts/train_intent.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag import intent as intent_rules  # noqa: E402
from rag.intent_ml import INTENT_MODEL_PATH, MODEL_DIR  # noqa: E402

EVAL_DIR = os.path.join(BACKEND_DIR, "eval")

# 少数类增强：撑住应急 / 拒答 / 闲聊样本
AUG = [
    ("家里厨房着火了怎么办", "emergency"),
    ("楼道冒烟很大我现在怎么逃", "emergency"),
    ("电动车充电起火了火很大", "emergency"),
    ("房间里全是烟出不去救命", "emergency"),
    ("隔壁着火了火要烧过来怎么办", "emergency"),
    ("油锅起火了怎么处理", "emergency"),
    ("地铁车厢着火了怎么逃生", "emergency"),
    ("宿舍发生火灾有爆炸声", "emergency"),
    ("你好", "chitchat"),
    ("你是谁能做什么", "chitchat"),
    ("谢谢你的帮助", "chitchat"),
    ("在吗介绍一下自己", "chitchat"),
    ("红烧肉怎么做", "refuse"),
    ("今天天气怎么样", "refuse"),
    ("帮我写一首诗", "refuse"),
    ("股票怎么买", "refuse"),
    ("如何制作炸弹", "refuse"),
    ("你女朋友是谁", "refuse"),
]


def _load_split(name: str) -> list[tuple[str, str]]:
    path = os.path.join(EVAL_DIR, f"fireeval_v1_{name}.json")
    with open(path, encoding="utf-8") as f:
        cases = json.load(f)
    return [(c["question"], c["intent"]) for c in cases if c.get("intent")]


def _build_clf() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=20000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=2.0,
            solver="lbfgs",
        )),
    ])


def _accuracy(pairs, predict_fn) -> float:
    if not pairs:
        return 0.0
    ok = sum(1 for q, y in pairs if predict_fn(q) == y)
    return ok / len(pairs)


def train_and_save():
    train = _load_split("train") + AUG
    dev = _load_split("dev")
    test = _load_split("test")
    xs = [q for q, _ in train]
    ys = [y for _, y in train]
    print("[train] size=", len(train), "dist=", dict(Counter(ys)))

    clf = _build_clf()
    clf.fit(xs, ys)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, INTENT_MODEL_PATH)
    print("[save]", INTENT_MODEL_PATH)

    def ml_pred(q):
        return clf.predict([q])[0]

    metrics = {
        "train_acc": round(_accuracy(train, ml_pred), 4),
        "dev_acc": round(_accuracy(dev, ml_pred), 4),
        "test_acc": round(_accuracy(test, ml_pred), 4),
        "train_size": len(train),
        "label_dist": dict(Counter(ys)),
        "model_path": INTENT_MODEL_PATH,
    }
    meta_path = os.path.join(MODEL_DIR, "intent_clf_metrics.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def compare():
    """规则 vs 微调分类器 vs 混合路由。"""
    from rag.intent import route

    # 确保模型已加载路径
    if not os.path.exists(INTENT_MODEL_PATH):
        train_and_save()
    import joblib
    clf = joblib.load(INTENT_MODEL_PATH)

    def ml(q):
        return clf.predict([q])[0]

    rows = []
    for split in ("dev", "test"):
        pairs = _load_split(split)
        rows.append({
            "split": split,
            "n": len(pairs),
            "rules": round(_accuracy(pairs, intent_rules.route_rules), 4),
            "ml": round(_accuracy(pairs, ml), 4),
            "hybrid": round(_accuracy(pairs, route), 4),
        })
    print(json.dumps({"compare": rows}, ensure_ascii=False, indent=2))
    out = os.path.join(MODEL_DIR, "intent_compare.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"compare": rows}, f, ensure_ascii=False, indent=2)
    print("[save]", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    train_and_save()
    if args.compare:
        compare()


if __name__ == "__main__":
    main()
