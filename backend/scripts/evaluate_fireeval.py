#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行 FireEval v1：评估意图路由、检索排序、结论正确性、引用准确率、安全拒答与响应耗时。

用法：
    python3 scripts/evaluate_fireeval.py                # 默认 200 题 v1 数据集
    python3 scripts/evaluate_fireeval.py --v0           # 旧 18 题基线
    python3 scripts/evaluate_fireeval.py --strict       # 按发布门槛判定（CI 门禁）

发布门槛（v1）：
    Hit@1 ≥ 85%，Hit@3 ≥ 95%，引用准确率 ≥ 95%，
    拒答准确率 ≥ 95%，严重错误率 ≤ 1%，平均响应 ≤ 8s。
"""
import argparse
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.pipeline import Pipeline  # noqa: E402

# 发布门槛
THRESHOLDS = {
    "hit_at_1": 0.85,
    "hit_at_3": 0.95,
    "citation_accuracy": 0.95,
    "refusal_accuracy": 0.95,
    "severe_error_rate": 0.01,
    "avg_latency_s": 8.0,
}


def evaluate(dataset_path, quiet=False, warmup=True):
    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as source:
        pipeline = Pipeline(json.load(source))
    with open(dataset_path, encoding="utf-8") as source:
        cases = json.load(source)

    # 预热：首次查询含模型加载与向量构建，不计入耗时统计
    if warmup:
        t = time.time()
        pipeline.ask("消防控制室的值班要求是什么？")
        if not quiet:
            print(f"[warmup] {time.time() - t:.1f}s（模型加载与索引构建，不计入耗时）",
                  file=sys.stderr)

    route_ok = retrieval_cases = hit1 = hit3 = 0
    reciprocal_rank = 0.0
    refusal_cases = refusal_ok = 0
    citation_cases = citation_ok = 0
    must_cases = must_ok = 0
    severe_errors = 0
    total_latency = 0.0
    failures = []

    for case in cases:
        t0 = time.time()
        response = pipeline.ask(case["question"], case.get("previous_question"),
                                use_cache=False)  # 评测绕过缓存，耗时统计才真实
        latency = time.time() - t0
        total_latency += latency

        # 意图路由
        if response["intent"] == case["intent"]:
            route_ok += 1
        else:
            failures.append({"id": case["id"], "kind": "route",
                             "got": response["intent"], "want": case["intent"]})

        # 拒答判定
        if "should_refuse" in case:
            refusal_cases += 1
            if response["refused"] == case["should_refuse"]:
                refusal_ok += 1
            else:
                failures.append({"id": case["id"], "kind": "refusal",
                                 "got": response["refused"], "want": case["should_refuse"]})

        # 严重错误：禁止出现的结论出现在回答中
        answer_text = response.get("answer", "")
        for forbidden in case.get("forbidden_conclusions", []):
            if forbidden in answer_text:
                severe_errors += 1
                failures.append({"id": case["id"], "kind": "severe",
                                 "detail": f"回答中出现禁止结论: {forbidden}"})

        # 检索排序
        expected = set(case.get("expected_articles", []))
        if expected:
            retrieval_cases += 1
            ranked = [item["article"] for item in response.get("references", [])]
            if ranked and ranked[0] in expected:
                hit1 += 1
            if expected.intersection(ranked[:3]):
                hit3 += 1
            first_rank = next(
                (i for i, a in enumerate(ranked, 1) if a in expected), None)
            if first_rank:
                reciprocal_rank += 1 / first_rank
            else:
                failures.append({"id": case["id"], "kind": "retrieval",
                                 "got": ranked[:3], "want": sorted(expected)})

            # 必须出现的结论（在回答或引用原文中任一出现即算支撑）
            musts = case.get("must_conclusions", [])
            if musts:
                must_cases += 1
                corpus = answer_text + " " + " ".join(
                    r.get("text", "") for r in response.get("references", []))
                if all(m in corpus for m in musts):
                    must_ok += 1
                else:
                    missing = [m for m in musts if m not in corpus]
                    failures.append({"id": case["id"], "kind": "must",
                                     "detail": f"缺少必须结论: {missing}"})

            # 引用准确率：回答中出现的条款引用必须来自命中的依据
            cited = set()
            answer_compact = answer_text.replace(" ", "")
            for ref in response.get("references", []):
                article = ref["article"]
                law, num = article.split("·", 1)
                if num in answer_compact:
                    cited.add(article)
            if cited:
                citation_cases += 1
                # 抽取式/LLM 模式都要求引用 ⊆ 依据
                if cited <= set(ranked):
                    citation_ok += 1
                else:
                    failures.append({"id": case["id"], "kind": "citation",
                                     "detail": f"引用了未返回的条款: {cited - set(ranked)}"})

    n = len(cases)
    metrics = {
        "dataset": os.path.basename(dataset_path),
        "cases": n,
        "route_accuracy": round(route_ok / max(1, n), 4),
        "retrieval_cases": retrieval_cases,
        "hit_at_1": round(hit1 / max(1, retrieval_cases), 4),
        "hit_at_3": round(hit3 / max(1, retrieval_cases), 4),
        "mrr": round(reciprocal_rank / max(1, retrieval_cases), 4),
        "must_conclusion_accuracy": round(must_ok / max(1, must_cases), 4),
        "citation_accuracy": round(citation_ok / max(1, citation_cases), 4),
        "refusal_accuracy": round(refusal_ok / max(1, refusal_cases), 4),
        "severe_error_rate": round(severe_errors / max(1, n), 4),
        "avg_latency_s": round(total_latency / max(1, n), 3),
        "failures": failures,
    }
    if not quiet:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def strict_check(metrics, thresholds):
    """发布门槛检查，返回 (通过, 明细)。"""
    checks = []
    ok = True
    for key, bound in thresholds.items():
        value = metrics[key]
        if key in ("severe_error_rate", "avg_latency_s"):
            passed = value <= bound
        else:
            passed = value >= bound
        checks.append({"metric": key, "value": value, "bound": bound, "passed": passed})
        ok = ok and passed
    return ok, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--v0", action="store_true", help="运行旧 18 题基线")
    parser.add_argument("--strict", action="store_true", help="按发布门槛判定（CI 门禁）")
    args = parser.parse_args()

    if args.dataset:
        dataset = args.dataset
    elif args.v0:
        dataset = os.path.join(BACKEND_DIR, "eval", "fireeval_v0.json")
    else:
        dataset = os.path.join(BACKEND_DIR, "eval", "fireeval_v1.json")

    result = evaluate(dataset)

    if args.strict:
        passed, checks = strict_check(result, THRESHOLDS)
        print("\n=== 发布门槛 ===")
        for c in checks:
            mark = "PASS" if c["passed"] else "FAIL"
            print(f"[{mark}] {c['metric']}: {c['value']} (bound: {c['bound']})")
        if not passed:
            raise SystemExit(1)
        print("\n全部门槛通过。")
