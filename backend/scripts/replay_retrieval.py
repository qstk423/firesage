# -*- coding: utf-8 -*-
"""回放 35 题检索（不进 LLM），采集 fused score + cross_encoder 分布。

目的：为拒答门槛计算提供完整检索分数依据（top1 fused / CE）。
用法（backend 目录）：python -B scripts\replay_retrieval.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
TEST_PATH = os.path.join(BACKEND_DIR, "data", "gen_sft_test.jsonl")
OUT_PATH = os.path.join(BACKEND_DIR, "eval_reports", "replay_retrieval.json")

import re


def extract_question(row):
    first = (row.get("input") or "").split("\n", 1)[0]
    return re.sub(r"^用户问题[：:]", "", first).strip()


def main():
    from rag.pipeline import Pipeline
    from rag.scene import structure, decompose_queries, detect_query_mode

    with open(os.path.join(BACKEND_DIR, "data", "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[加载] chunks={len(chunks)}，BGE-M3 走 CPU…", flush=True)
    pipe = Pipeline(chunks=chunks)
    print("[就绪] 开始回放检索", flush=True)

    rows = []
    with open(TEST_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    out = []
    for i, row in enumerate(rows, 1):
        if row.get("intent") != "law":
            continue
        q = extract_question(row)
        scene = structure(q)
        rq = f"{scene['rewrite']} {q}" if scene.get("rewrite") and scene["rewrite"] != q else q
        subs = decompose_queries(q, scene)
        if rq not in subs:
            subs = [rq] + [s for s in subs if s != rq]
        mode = detect_query_mode(q, scene)
        fused, _ = pipe.retriever.retrieve(rq, top_k=5, sub_queries=subs, query_mode=mode)
        rec = {
            "id": row.get("id"),
            "question": q,
            "expected_articles": row.get("expected_articles"),
            "tops": [{"article": f["article"], "score": round(f["score"], 3),
                      "ce": round(f["cross_encoder"], 3) if f.get("cross_encoder") is not None else None}
                     for f in fused[:5]],
        }
        out.append(rec)
        print(f"  [{i}/{len(rows)}] {row.get('id')} top1={rec['tops'][0]['score'] if rec['tops'] else '-'} "
              f"ce1={rec['tops'][0]['ce'] if rec['tops'] else '-'}", flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[保存] {OUT_PATH}")


if __name__ == "__main__":
    main()
