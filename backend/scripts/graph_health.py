#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图谱健康度：有边条款占比、各来源边数、枢纽节点、孤儿条款。

用法：
    python3 backend/scripts/graph_health.py
    python3 backend/scripts/graph_health.py --markdown docs/图谱健康度.md
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(BACKEND_DIR, "data")


def compute() -> dict:
    with open(os.path.join(DATA_DIR, "chunks.json"), encoding="utf-8") as f:
        chunks = json.load(f)
    with open(os.path.join(DATA_DIR, "graph.json"), encoding="utf-8") as f:
        graph = json.load(f)

    arts = {c["article"] for c in chunks}
    edge_arts = {e.get("article") for e in graph["edges"] if e.get("article")}
    linked = arts & edge_arts
    orphans = sorted(arts - edge_arts)

    by_law_edges: dict[str, int] = Counter()
    by_law_arts: dict[str, set] = defaultdict(set)
    for e in graph["edges"]:
        art = e.get("article") or ""
        if "·" not in art:
            continue
        law = art.split("·", 1)[0]
        by_law_edges[law] += 1
        by_law_arts[law].add(art)

    orphan_by_law: dict[str, int] = Counter()
    for art in orphans:
        orphan_by_law[art.split("·", 1)[0]] += 1

    deg = Counter()
    for e in graph["edges"]:
        deg[e["source"]] += 1
        deg[e["target"]] += 1

    type_dist = Counter(n.get("type", "?") for n in graph["nodes"])
    rel_dist = Counter(e.get("relation", "?") for e in graph["edges"])

    return {
        "date": date.today().isoformat(),
        "semantic_sents": len(chunks),
        "articles": len(arts),
        "entities": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "articles_with_edge": len(linked),
        "orphan_articles": len(orphans),
        "edge_coverage": round(len(linked) / max(1, len(arts)), 4),
        "by_law": [
            {
                "law": law,
                "edges": by_law_edges[law],
                "linked_articles": len(by_law_arts[law]),
                "orphan_articles": orphan_by_law.get(law, 0),
            }
            for law in sorted(set(by_law_edges) | set(orphan_by_law),
                              key=lambda x: (-by_law_edges.get(x, 0), x))
        ],
        "hubs": [{"id": nid, "degree": d} for nid, d in deg.most_common(12)],
        "type_dist": dict(sorted(type_dist.items(), key=lambda x: -x[1])),
        "relation_dist": dict(sorted(rel_dist.items(), key=lambda x: -x[1])),
        "orphan_sample": orphans[:24],
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# 图谱健康度",
        "",
        f"> 生成日期：{report['date']}（`python3 backend/scripts/graph_health.py --markdown`）",
        "",
        "## 总览",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| 语义句 | {report['semantic_sents']} |",
        f"| 条款数 | {report['articles']} |",
        f"| 实体 / 边 | {report['entities']} / {report['edges']} |",
        f"| 有边条款 | {report['articles_with_edge']} |",
        f"| 孤儿条款 | {report['orphan_articles']} |",
        f"| 边覆盖率 | {report['edge_coverage']:.1%} |",
        "",
        "## 各来源",
        "",
        "| 法规 | 边数 | 有边条款 | 孤儿条款 |",
        "|------|------|----------|----------|",
    ]
    for row in report["by_law"]:
        lines.append(
            f"| {row['law']} | {row['edges']} | {row['linked_articles']} | {row['orphan_articles']} |"
        )
    lines += [
        "",
        "## 枢纽节点（度 Top）",
        "",
        "| 实体 | 度 |",
        "|------|-----|",
    ]
    for h in report["hubs"]:
        lines.append(f"| `{h['id']}` | {h['degree']} |")
    lines += [
        "",
        "## 说明",
        "",
        "- **孤儿条款**：入库有正文，但词典未抽到行为边（总则/定义/施行日等常见）。",
        "- **39号令**边偏少时：优先补 `graphrag.py` 娱乐场所专用行为词典后重建。",
        "- 覆盖率不必追求 100%；核心禁止/义务句有边即可支撑 GraphRAG。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", default=None, help="写入 Markdown 路径（相对仓库根）")
    args = parser.parse_args()
    report = compute()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.markdown:
        path = args.markdown
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(ROOT_DIR, path))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(to_markdown(report))
        print(f"\n[wrote] {path}", flush=True)


if __name__ == "__main__":
    main()
