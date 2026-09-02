# -*- coding: utf-8 -*-
"""混合检索引擎：BM25 + GraphRAG 加权融合

BM25(0.65) + 图谱多跳(0.35)，每条引用标注 graph_hit / bm25 / graph 贡献。
"""
import os

try:
    import jieba
except ImportError:
    jieba = None
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .graphrag import GraphRetriever, load_graph

# 权重
W_BM25 = 0.65
W_GRAPH = 0.35

# 同义词扩展
SYNONYMS = {
    "楼道": ["楼道", "走廊", "过道", "楼梯间"],
    "堆放": ["堆放", "堆", "堆积", "放置"],
    "杂物": ["杂物", "物品", "东西", "垃圾"],
    "违反": ["违反", "违规", "违犯", "触犯"],
    "消防通道": ["消防车通道", "消防通道", "防火通道"],
    "怎么处罚": ["处罚", "罚款", "怎么罚", "后果", "责任"],
    "事情": [],
    "怎么办": ["怎么办", "处理", "处置", "如何处理"],
    "电动车": ["电动自行车", "电瓶车", "电动车"],
    "楼道充电": ["公共门厅充电", "疏散走道充电", "楼梯间充电", "电动自行车充电"],
    "高层住宅": ["高层住宅建筑", "高层民用建筑", "住宅楼"],
    "消控室": ["消防控制室", "消控室"],
    "物业": ["物业服务企业", "统一管理人", "物业公司"],
}


def _tokenize(text):
    if jieba:
        return list(jieba.cut(text))
    # 简化降级：按常用标点+2-gram
    res = []
    for i in range(len(text) - 1):
        res.append(text[i:i + 2])
    return res


def _expand(question):
    """同义词扩展，拼接进查询"""
    expanded = [question]
    for w, syns in SYNONYMS.items():
        if w in question:
            expanded.extend(syns)
    return expanded


class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.articles = {}  # article -> list of chunk
        self.bm25 = None
        if BM25Okapi:
            corpus = [_tokenize(c["text"]) for c in chunks]
            self.bm25 = BM25Okapi(corpus)

    def search(self, question, top_k=5):
        if not self.bm25:
            return []
        expanded = _expand(question)
        best_avail = {}
        for q in expanded:
            scores = self.bm25.get_scores(_tokenize(q))
            for cid, s in enumerate(scores):
                atcl = self.chunks[cid]["article"]
                if s > best_avail.get(atcl, 0):
                    best_avail[atcl] = s
        if not best_avail:
            return []
        mx = max(best_avail.values()) or 1.0
        ranked = sorted(best_avail.items(), key=lambda x: -x[1])[:top_k]
        return [(atcl, s / mx) for atcl, s in ranked]


class HybridRetriever:
    def __init__(self, chunks=None):
        graph = load_graph()
        self.bm25 = BM25Index(chunks)
        self.graph = GraphRetriever(graph)

    def retrieve(self, question, top_k=5):
        bm25_hits = {atcl: s for atcl, s in self.bm25.search(question, top_k)}
        graph_hits = {}
        for atcl, s, path in self.graph.search(question, top_k):
            graph_hits[atcl] = (s, path)

        # 融合
        fused = {}
        summary = {"bm25_articles": set(bm25_hits), "graph_articles": set(graph_hits)}
        for atcl in set(bm25_hits) | set(graph_hits):
            b = bm25_hits.get(atcl, 0.0)
            g, gpath = graph_hits.get(atcl, (0.0, None))
            score = W_BM25 * b + W_GRAPH * g
            bm25_part = W_BM25 * b
            graph_part = W_GRAPH * g
            contribution_total = bm25_part + graph_part
            fused[atcl] = {
                "article": atcl,
                "score": score,
                "bm25": round(b, 3),
                "graph": round(g, 3),
                "bm25_share": round(bm25_part / contribution_total, 3) if contribution_total else 0.0,
                "graph_share": round(graph_part / contribution_total, 3) if contribution_total else 0.0,
                "graph_hit": atcl in graph_hits,
                "graph_path": gpath,
            }
        ranked = sorted(fused.values(), key=lambda x: -x["score"])[:top_k]
        return ranked, summary
