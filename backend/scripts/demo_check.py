#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键验收：意图对比 + FireEval + 烟雾 + 演示题。

用法：
    python3 backend/scripts/demo_check.py --preflight  # 答辩日：烟雾+演示题+runtime（推荐）
    python3 backend/scripts/demo_check.py --quick      # 再加 intent compare + FireEval dev
    python3 backend/scripts/demo_check.py              # 全量（含 test）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(ROOT, "backend")

# 与 docs/校赛演示材料包.md 三问 + 隐患对照一致
DEMO_CASES = [
    ("楼道堆放杂物违反什么规定", {"refused": False}),
    ("家里着火了现在怎么办", {"intent": "emergency"}),
    ("推荐几部好看的电影", {"refused": True}),
    ("发现火灾隐患怎么办", {"intent": "law"}),  # 不得进应急
]


def run(cmd: list[str], timeout: int = 300) -> tuple[int, str]:
    print("\n$", " ".join(cmd))
    p = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout
    )
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        print(out[-2000:])
    return p.returncode, out


def parse_eval(stdout: str) -> dict:
    idx = stdout.rfind('"dataset"')
    if idx < 0:
        return {}
    start = stdout.rfind("{", 0, idx)
    depth = 0
    for i in range(start, len(stdout)):
        if stdout[i] == "{":
            depth += 1
        elif stdout[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stdout[start:i + 1])
    return {}


def run_demo_questions(summary: dict) -> None:
    sys.path.insert(0, BACKEND)
    from rag.pipeline import Pipeline

    with open(os.path.join(BACKEND, "data", "chunks.json"), encoding="utf-8") as f:
        pl = Pipeline(json.load(f))
    demos = []
    for q, expect in DEMO_CASES:
        r = pl.ask(q, use_cache=False)
        ok = all(r.get(k) == v for k, v in expect.items())
        demos.append({
            "q": q,
            "ok": ok,
            "intent": r.get("intent"),
            "refused": r.get("refused"),
            "expect": expect,
        })
        if not ok:
            summary["ok"] = False
    summary["steps"].append({
        "name": "demo_questions",
        "ok": all(d["ok"] for d in demos),
        "demos": demos,
    })


def run_runtime_probe(summary: dict) -> None:
    code, out = run([sys.executable, "backend/scripts/check_runtime.py"], timeout=180)
    tier = None
    try:
        # 取最后一段 JSON
        start = out.rfind("{")
        end = out.rfind("}")
        if start >= 0 and end > start:
            tier = json.loads(out[start:end + 1]).get("quality_tier")
    except Exception:
        tier = None
    # 预检不强制 full：lite/partial 也能演示；仅脚本崩溃算失败
    summary["steps"].append({
        "name": "runtime",
        "ok": code == 0,
        "quality_tier": tier,
    })
    if code != 0:
        summary["ok"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="intent + FireEval dev + smoke + demos")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="答辩日预检：smoke + 演示题 + runtime（最快）",
    )
    args = parser.parse_args()

    py = sys.executable
    summary = {"ok": True, "steps": [], "mode": "full"}
    if args.preflight:
        summary["mode"] = "preflight"
    elif args.quick:
        summary["mode"] = "quick"

    if not args.preflight:
        code, out = run([py, "backend/scripts/train_intent.py", "--compare"])
        summary["steps"].append({"name": "intent_compare", "ok": code == 0})
        if code != 0:
            summary["ok"] = False

        splits = ["dev"] if args.quick else ["dev", "test"]
        for split in splits:
            code, out = run([py, "backend/scripts/evaluate_fireeval.py", "--split", split])
            metrics = parse_eval(out)
            step = {
                "name": f"fireeval_{split}",
                "ok": code == 0,
                "hit_at_1": metrics.get("hit_at_1"),
                "hit_at_3": metrics.get("hit_at_3"),
                "route_accuracy": metrics.get("route_accuracy"),
                "refusal_accuracy": metrics.get("refusal_accuracy"),
                "failures": len(metrics.get("failures") or []),
            }
            summary["steps"].append(step)
            if code != 0:
                summary["ok"] = False
            print(json.dumps(step, ensure_ascii=False))

    code, out = run([py, "-m", "unittest", "tests.test_smoke", "-q"], timeout=300)
    summary["steps"].append({"name": "smoke", "ok": code == 0})
    if code != 0:
        summary["ok"] = False

    run_demo_questions(summary)

    if args.preflight:
        run_runtime_probe(summary)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
