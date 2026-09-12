#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索消融：对比 BM25 / +向量 / 三路融合 / 全链路重排 的 Hit@1、Hit@3、MRR。

用法：
    python3 scripts/ablation_retrieval.py --split dev
    python3 scripts/ablation_retrieval.py --split all
    python3 scripts/ablation_retrieval.py --split all --markdown docs/消融表-retrieval.md

说明：
    - 追问用例会合并 previous_question（与线上 pipeline 一致的短追问规则）
    - 输出含向量通道是否降级，便于答辩区分「全量模型 / TF-IDF 降级」两套数
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)

from rag.intent import route  # noqa: E402
from rag.pipeline import FOLLOW_UP_MARKERS  # noqa: E402
from rag.retriever import HybridRetriever  # noqa: E402
from rag.scene import structure  # noqa: E402

MODES = (
    ("bm25", "仅 BM25"),
    ("bm25_vector", "BM25 + 向量 RRF"),
    ("hybrid", "三路 RRF（无重排）"),
    ("full", "三路 RRF + 法规重排 + CE"),
)


def _dataset_path(split: str) -> str:
    if split == "all":
        return os.path.join(BACKEND_DIR, "eval", "fireeval_v1.json")
    return os.path.join(BACKEND_DIR, "eval", f"fireeval_v1_{split}.json")


def _retrieval_question(case: dict) -> str:
    question = (case.get("question") or "").strip()
    previous = (case.get("previous_question") or "").strip()
    context_used = bool(
        previous
        and (len(question) <= 24 or any(m in question for m in FOLLOW_UP_MARKERS))
    )
    full = f"{previous}；追问：{question}" if context_used else question
    scene = structure(full)
    if scene.get("rewrite") and scene["rewrite"] != question:
        return f"{scene['rewrite']} {full}"
    return full


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return round(ordered[index], 2)


def _metrics_for_mode(retriever: HybridRetriever, cases: list[dict], mode: str) -> dict:
    retrieval_cases = hit1 = hit3 = hit5 = 0
    reciprocal_rank = 0.0
    case_latency_ms = []
    stage_values: dict[str, list[float]] = {}
    t0 = time.time()
    for case in cases:
        expected = set(case.get("expected_articles") or [])
        if not expected or case.get("should_refuse"):
            continue
        if case.get("intent") == "emergency":
            continue
        # 拒答意图题不做检索消融
        if route(case.get("question") or "") == "refuse" and not expected:
            continue
        retrieval_cases += 1
        rq = _retrieval_question(case)
        case_started = time.perf_counter()
        ranked, summary = retriever.retrieve(rq, top_k=5, mode=mode)
        case_latency_ms.append((time.perf_counter() - case_started) * 1000)
        for key, value in (summary.get("timing") or {}).items():
            stage_values.setdefault(key, []).append(float(value))
        arts = [item["article"] for item in ranked]
        if arts and arts[0] in expected:
            hit1 += 1
        if expected.intersection(arts[:3]):
            hit3 += 1
        if expected.intersection(arts[:5]):
            hit5 += 1
        first_rank = next((i for i, a in enumerate(arts, 1) if a in expected), None)
        if first_rank:
            reciprocal_rank += 1 / first_rank
    n = max(1, retrieval_cases)
    return {
        "mode": mode,
        "retrieval_cases": retrieval_cases,
        "hit_at_1": round(hit1 / n, 4),
        "hit_at_3": round(hit3 / n, 4),
        "hit_at_5": round(hit5 / n, 4),
        "mrr": round(reciprocal_rank / n, 4),
        "elapsed_s": round(time.time() - t0, 2),
        "latency_ms": {
            "p50": _percentile(case_latency_ms, 0.50),
            "p95": _percentile(case_latency_ms, 0.95),
            "max": round(max(case_latency_ms), 2) if case_latency_ms else None,
        },
        "stage_ms_p50": {
            key: _percentile(values, 0.50) for key, values in stage_values.items()
        },
    }


def _to_markdown(payload: dict) -> str:
    rows = payload["ablation"]
    lines = [
        "# FireSage 检索消融表",
        "",
        f"- split: `{payload['split']}`",
        f"- vector_channel: `{payload['vector_channel']}`",
        f"- degraded_vector: `{payload['degraded_vector']}`",
        f"- channels: `{', '.join(payload['channels'])}`",
        f"- generated_at: `{payload['generated_at']}`",
        "",
        "| 模式 | 说明 | Hit@1 | Hit@3 | Hit@5 | MRR | n | P50(ms) | P95(ms) |",
        "|------|------|------:|------:|------:|----:|--:|--------:|--------:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['mode']}` | {row['label']} | {row['hit_at_1']:.4f} | "
            f"{row['hit_at_3']:.4f} | {row['hit_at_5']:.4f} | {row['mrr']:.4f} | "
            f"{row['retrieval_cases']} | {row['latency_ms']['p50']} | "
            f"{row['latency_ms']['p95']} |"
        )
    lines.extend([
        "",
        "## 读表提示",
        "",
        "- `bm25` → `bm25_vector`：向量通道贡献",
        "- `bm25_vector` → `hybrid`：GraphRAG 通道贡献",
        "- `hybrid` → `full`：法规意图重排 + CrossEncoder 贡献",
        "- 若 `degraded_vector=true`，本表为 TF-IDF 降级环境，答辩需同时标注「全量模型待复跑」",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
    parser.add_argument(
        "--markdown",
        default="",
        help="可选：写入 Markdown 表路径（相对仓库根或绝对路径）",
    )
    args = parser.parse_args()

    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(_dataset_path(args.split), encoding="utf-8") as f:
        cases = json.load(f)

    retriever = HybridRetriever(chunks)
    rows = []
    for mode, label in MODES:
        row = _metrics_for_mode(retriever, cases, mode)
        row["label"] = label
        rows.append(row)
        print(
            f"[{mode}] Hit@1={row['hit_at_1']:.4f} Hit@3={row['hit_at_3']:.4f} "
            f"Hit@5={row['hit_at_5']:.4f} MRR={row['mrr']:.4f} "
            f"n={row['retrieval_cases']} P50={row['latency_ms']['p50']}ms "
            f"P95={row['latency_ms']['p95']}ms — {label}"
        )

    payload = {
        "split": args.split,
        "degraded_vector": bool(getattr(retriever, "degraded_vector", False)),
        "vector_channel": retriever.vector.name,
        "channels": list(getattr(retriever, "channels", [])),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ablation": rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.markdown:
        out = args.markdown
        if not os.path.isabs(out):
            out = os.path.join(ROOT_DIR, out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(_to_markdown(payload))
        print(f"[write] {out}")


if __name__ == "__main__":
    main()
