#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一键验收：意图对比 + 重排训练（可选）+ FireEval dev/test + 烟雾测试。

用法：
    python3 backend/scripts/demo_check.py
    python3 backend/scripts/demo_check.py --quick   # 只跑 dev + smoke
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKEND = os.path.join(ROOT, "backend")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    summary = {"ok": True, "steps": []}

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

    code, out = run([py, "-m", "unittest", "tests.test_smoke", "-q"])
    summary["steps"].append({"name": "smoke", "ok": code == 0})
    if code != 0:
        summary["ok"] = False

    # 演示三问冒烟（不依赖 LLM）
    sys.path.insert(0, BACKEND)
    from rag.pipeline import Pipeline
    with open(os.path.join(BACKEND, "data", "chunks.json"), encoding="utf-8") as f:
        pl = Pipeline(json.load(f))
    demos = []
    for q, expect in [
        ("楼道堆放杂物违反什么规定", {"refused": False}),
        ("家里着火了现在怎么办", {"intent": "emergency"}),
        ("推荐几部好看的电影", {"refused": True}),
    ]:
        r = pl.ask(q, use_cache=False)
        ok = all(r.get(k) == v for k, v in expect.items())
        demos.append({"q": q, "ok": ok, "intent": r.get("intent"), "refused": r.get("refused")})
        if not ok:
            summary["ok"] = False
    summary["steps"].append({"name": "demo_questions", "ok": all(d["ok"] for d in demos), "demos": demos})

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    sys.exit(0 if summary["ok"] else 1)


if __name__ == "__main__":
    main()
