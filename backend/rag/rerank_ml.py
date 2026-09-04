# -*- coding: utf-8 -*-
"""条款重排微调分类器（FireEval 点式学习排序）。

对 (问题, 条款文本) 预测是否为金标依据；推理时对候选打分并与规则重排融合。
"""
from __future__ import annotations

import os
from typing import Optional

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models"
)
RERANK_MODEL_PATH = os.path.join(MODEL_DIR, "rerank_clf.joblib")

_clf = None
_loaded = False


def _load():
    global _clf, _loaded
    if _loaded:
        return _clf
    _loaded = True
    path = os.getenv("RERANK_MODEL_PATH", RERANK_MODEL_PATH)
    if not os.path.exists(path):
        _clf = None
        return None
    try:
        import joblib
        _clf = joblib.load(path)
    except Exception as e:
        print(f"[RerankML] 加载失败（{e}），跳过微调重排")
        _clf = None
    return _clf


def available() -> bool:
    """默认关闭：需显式 RERANK_ML=1，避免弱分类器干扰规则重排。"""
    if os.getenv("RERANK_ML", "").strip() not in ("1", "true", "True", "yes"):
        return False
    return _load() is not None


def _pair_text(question: str, article_text: str) -> str:
    return f"{question}||{(article_text or '')[:800]}"


def score(question: str, article_text: str) -> float:
    clf = _load()
    if clf is None:
        return 0.0
    proba = clf.predict_proba([_pair_text(question, article_text)])[0]
    # 正类概率
    classes = list(clf.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return float(proba[-1])


def score_many(question: str, articles: list[tuple[str, str]]) -> dict[str, float]:
    """articles: [(article_id, text), ...] → {article_id: score}"""
    clf = _load()
    if clf is None or not articles:
        return {}
    texts = [_pair_text(question, text) for _, text in articles]
    probas = clf.predict_proba(texts)
    classes = list(clf.classes_)
    pos = classes.index(1) if 1 in classes else -1
    out = {}
    for (aid, _), proba in zip(articles, probas):
        out[aid] = float(proba[pos])
    return out
