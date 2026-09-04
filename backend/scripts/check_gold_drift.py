#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""金标漂移检查：expected_articles 是否仍在语料；must_conclusions 是否出现在金标条款正文。

用法：
    python3 backend/scripts/check_gold_drift.py
    python3 backend/scripts/check_gold_drift.py --split all --strict
"""
from __future__ import annotations

import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.graphrag import ARCHIVE_PATHS  # noqa: E402


def _load_article_bodies() -> dict[str, str]:
    bodies = {}
    for path, fallback in ARCHIVE_PATHS:
        with open(path, encoding="utf-8") as f:
            law = json.load(f)
        abbr = law.get("law_abbr", fallback)
        for ch in law.get("chapters", []):
            for art in ch.get("articles", []):
                key = f"{abbr}·{art['num']}"
                bodies[key] = art.get("text", "")
    return bodies


def _datasets(split: str) -> list[str]:
    eval_dir = os.path.join(BACKEND_DIR, "eval")
    if split == "all":
        return [
            os.path.join(eval_dir, "fireeval_v1.json"),
            os.path.join(eval_dir, "fireeval_v1_train.json"),
            os.path.join(eval_dir, "fireeval_v1_dev.json"),
            os.path.join(eval_dir, "fireeval_v1_test.json"),
        ]
    return [os.path.join(eval_dir, f"fireeval_v1_{split}.json")]


def check(paths: list[str]) -> dict:
    bodies = _load_article_bodies()
    corpus_keys = set(bodies)
    missing_articles = []
    must_not_in_gold = []
    seen_cases = 0

    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            cases = json.load(f)
        for case in cases:
            seen_cases += 1
            expected = case.get("expected_articles") or []
            for art in expected:
                if art not in corpus_keys:
                    missing_articles.append({
                        "file": os.path.basename(path),
                        "id": case.get("id"),
                        "article": art,
                    })
            musts = case.get("must_conclusions") or []
            if not expected or not musts:
                continue
            joined = " ".join(bodies.get(a, "") for a in expected)
            for m in musts:
                # 应急/拒答题常把 119 等放在 must，跳过非法规字面
                if m in ("119",):
                    continue
                if m and m not in joined:
                    must_not_in_gold.append({
                        "file": os.path.basename(path),
                        "id": case.get("id"),
                        "must": m,
                        "expected": expected,
                    })

    # 同 id 跨 split 重复报错去重
    def _dedupe(items, key_fn):
        seen, out = set(), []
        for it in items:
            k = key_fn(it)
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    missing_articles = _dedupe(missing_articles, lambda x: (x["id"], x["article"]))
    must_not_in_gold = _dedupe(must_not_in_gold, lambda x: (x["id"], x["must"]))

    return {
        "cases_scanned": seen_cases,
        "corpus_articles": len(corpus_keys),
        "missing_article_count": len(missing_articles),
        "must_not_in_gold_count": len(must_not_in_gold),
        "missing_articles": missing_articles[:40],
        "must_not_in_gold": must_not_in_gold[:40],
        "ok": len(missing_articles) == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    parser.add_argument("--strict", action="store_true",
                        help="缺条款则 exit 1；must 仅告警除非 --strict-must")
    parser.add_argument("--strict-must", action="store_true",
                        help="must 不在金标正文也失败（偏严，口语题可能误报）")
    args = parser.parse_args()

    report = check(_datasets(args.split))
    print(json.dumps(report, ensure_ascii=False, indent=2))

    failed = False
    if args.strict and report["missing_article_count"]:
        failed = True
        print(f"\n[FAIL] {report['missing_article_count']} expected_articles 不在语料",
              file=sys.stderr)
    if report["must_not_in_gold_count"]:
        print(f"\n[WARN] {report['must_not_in_gold_count']} must_conclusions 未出现在金标条款正文"
              f"（可能是口语同义或跨条结论）", file=sys.stderr)
        if args.strict_must:
            failed = True
    if failed:
        raise SystemExit(1)
    print("\n金标条款存在性检查通过。" if report["ok"] else "\n存在缺失条款。")


if __name__ == "__main__":
    main()
