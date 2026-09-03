#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行 FireEval v0：评估路由、检索排序与安全拒答。"""
import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from rag.pipeline import Pipeline  # noqa: E402


def evaluate(dataset_path, quiet=False):
    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as source:
        pipeline = Pipeline(json.load(source))
    with open(dataset_path, encoding="utf-8") as source:
        cases = json.load(source)

    route_ok = retrieval_cases = hit1 = hit3 = 0
    reciprocal_rank = 0.0
    refusal_cases = refusal_ok = 0
    failures = []
    for case in cases:
        response = pipeline.ask(case["question"])
        if response["intent"] == case["intent"]:
            route_ok += 1
        else:
            failures.append({"id": case["id"], "kind": "route", "got": response["intent"]})

        if "should_refuse" in case:
            refusal_cases += 1
            if response["refused"] == case["should_refuse"]:
                refusal_ok += 1
            else:
                failures.append({"id": case["id"], "kind": "refusal", "got": response["refused"]})

        expected = set(case.get("expected_articles", []))
        if expected:
            retrieval_cases += 1
            ranked = [item["article"] for item in response.get("references", [])]
            if ranked and ranked[0] in expected:
                hit1 += 1
            if expected.intersection(ranked[:3]):
                hit3 += 1
            first_rank = next((index for index, article in enumerate(ranked, 1) if article in expected), None)
            if first_rank:
                reciprocal_rank += 1 / first_rank
            else:
                failures.append({"id": case["id"], "kind": "retrieval", "got": ranked[:3]})

    metrics = {
        "cases": len(cases),
        "route_accuracy": round(route_ok / max(1, len(cases)), 4),
        "retrieval_cases": retrieval_cases,
        "hit_at_1": round(hit1 / max(1, retrieval_cases), 4),
        "hit_at_3": round(hit3 / max(1, retrieval_cases), 4),
        "mrr": round(reciprocal_rank / max(1, retrieval_cases), 4),
        "refusal_accuracy": round(refusal_ok / max(1, refusal_cases), 4),
        "failures": failures,
    }
    if not quiet:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=os.path.join(BACKEND_DIR, "eval", "fireeval_v0.json"))
    parser.add_argument("--strict", action="store_true", help="指标低于基线时返回失败")
    args = parser.parse_args()
    result = evaluate(args.dataset)
    if args.strict and (result["hit_at_1"] < 0.75 or result["hit_at_3"] < 0.9
                        or result["route_accuracy"] < 0.9 or result["refusal_accuracy"] < 1.0):
        raise SystemExit(1)
