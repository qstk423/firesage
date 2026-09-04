# -*- coding: utf-8 -*-
"""领域意图分类器（FireEval 微调产物）。

INTENT_BACKEND:
  rules  — 仅规则
  ml     — 仅分类器
  hybrid — 规则硬约束 + 分类器（有模型时的默认）
"""
from __future__ import annotations

import os
from typing import Optional

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models"
)
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_clf.joblib")

_clf = None
_loaded = False


def _load():
    global _clf, _loaded
    if _loaded:
        return _clf
    _loaded = True
    path = os.getenv("INTENT_MODEL_PATH", INTENT_MODEL_PATH)
    if not os.path.exists(path):
        _clf = None
        return None
    try:
        import joblib
        _clf = joblib.load(path)
    except Exception as e:
        print(f"[IntentML] 加载失败（{e}），回退规则")
        _clf = None
    return _clf


def available() -> bool:
    return _load() is not None


def predict_proba(question: str) -> Optional[dict]:
    clf = _load()
    if clf is None:
        return None
    proba = clf.predict_proba([question])[0]
    return {c: float(p) for c, p in zip(clf.classes_, proba)}


def predict(question: str, min_confidence: float = 0.42) -> Optional[str]:
    scores = predict_proba(question)
    if not scores:
        return None
    label, conf = max(scores.items(), key=lambda x: x[1])
    if conf < min_confidence:
        return None
    return label
