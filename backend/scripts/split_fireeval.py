#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 FireEval v1 数据集拆分为 train/dev/test（用于阶段性评测与微调）。

说明：
- 这是“评测框架重构/数据划分”，不引入新的业务逻辑。
- 采用按 intent 分层抽样，保证各分割的路由/拒答分布相近。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(BACKEND_DIR, "eval")


def _split_by_intent(cases: list[dict], train_ratio: float, dev_ratio: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        buckets[str(c.get("intent", "unknown"))].append(c)

    out = {"train": [], "dev": [], "test": []}
    for intent, group in buckets.items():
        group = list(group)
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_ratio))
        n_dev = int(round(n * dev_ratio))
        # 保底：确保 test 非空
        n_test = n - n_train - n_dev
        if n_test < 1:
            # 把 dev 的一个样本挪到 test
            if n_dev > 0:
                n_dev -= 1
                n_test = n - n_train - n_dev
            if n_test < 1:
                # 再退一步：挪一个到 test（极小样本兜底）
                n_train = max(0, n_train - 1)
                n_test = n - n_train - n_dev

        out["train"].extend(group[:n_train])
        out["dev"].extend(group[n_train:n_train + n_dev])
        out["test"].extend(group[n_train + n_dev:n_train + n_dev + n_test])

    # 洗牌，避免分割内按 id 顺序
    for k in out:
        rng.shuffle(out[k])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(EVAL_DIR, "fireeval_v1.json"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError("input dataset must be a non-empty JSON list")

    splits = _split_by_intent(cases, args.train_ratio, args.dev_ratio, args.seed)
    for name, items in splits.items():
        out_path = os.path.join(EVAL_DIR, f"fireeval_v1_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"[split] {name}: {len(items)} cases -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()

