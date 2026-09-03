# -*- coding: utf-8 -*-
"""混合检索引擎：BM25 + 本地向量 + GraphRAG + 二阶段重排。

三路召回使用 RRF 融合，避免各检索器分数量纲不同导致错误置信度。
"""
import math
from collections import Counter

try:
    import jieba
except ImportError:
    jieba = None
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from .graphrag import GraphRetriever, load_graph
from .reranker import LegalReranker
from .vector_index import TfidfVectorIndex

# RRF 权重与平滑常数。BM25 保留较高权重，图谱负责补充关系证据。
CHANNEL_WEIGHTS = {"bm25": 0.42, "vector": 0.33, "graph": 0.25}
RRF_K = 60

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
        else:
            self.corpus = [_tokenize(c["text"]) for c in chunks]
            self.doc_freq = Counter()
            for tokens in self.corpus:
                self.doc_freq.update(set(tokens))
            self.avg_len = sum(map(len, self.corpus)) / max(1, len(self.corpus))

    def _fallback_scores(self, tokens):
        """无 rank_bm25 时使用标准 BM25 公式，避免静默丢失关键词召回。"""
        n_docs = max(1, len(self.corpus))
        scores = []
        for document in self.corpus:
            counts = Counter(document)
            score = 0.0
            for token in tokens:
                frequency = counts.get(token, 0)
                if not frequency:
                    continue
                df = self.doc_freq.get(token, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(document) / max(1, self.avg_len))
                score += idf * frequency * 2.5 / denominator
            scores.append(score)
        return scores

    def search(self, question, top_k=5):
        expanded = _expand(question)
        best_avail = {}
        for q in expanded:
            tokens = _tokenize(q)
            scores = self.bm25.get_scores(tokens) if self.bm25 else self._fallback_scores(tokens)
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
        self.chunks = chunks or []
        self.bm25 = BM25Index(self.chunks)
        self.vector = TfidfVectorIndex(self.chunks)
        self.graph = GraphRetriever(graph)
        self.reranker = LegalReranker(self.chunks)
        self.channels = ["BM25", self.vector.name, "GraphRAG", self.reranker.name]

    def retrieve(self, question, top_k=5):
        pool_size = max(12, top_k * 4)
        bm25_list = self.bm25.search(question, pool_size)
        vector_list = self.vector.search(" ".join(_expand(question)), pool_size)
        graph_list = self.graph.search(question, pool_size)
        raw = {
            "bm25": {article: score for article, score in bm25_list},
            "vector": {article: score for article, score in vector_list},
            "graph": {article: score for article, score, _ in graph_list},
        }
        graph_paths = {article: path for article, _, path in graph_list}
        rrf_parts = {name: {} for name in raw}
        for name, ranked in (("bm25", bm25_list), ("vector", vector_list),
                             ("graph", [(a, s) for a, s, _ in graph_list])):
            for rank, (article, _) in enumerate(ranked, 1):
                rrf_parts[name][article] = CHANNEL_WEIGHTS[name] / (RRF_K + rank)

        articles = set().union(*(set(hits) for hits in raw.values()))
        max_rrf = sum(weight / (RRF_K + 1) for weight in CHANNEL_WEIGHTS.values())
        candidates = []
        for article in articles:
            parts = {name: rrf_parts[name].get(article, 0.0) for name in rrf_parts}
            total = sum(parts.values())
            candidates.append({
                "article": article,
                "retrieval_score": total / max_rrf if max_rrf else 0.0,
                "bm25": round(raw["bm25"].get(article, 0.0), 3),
                "vector": round(raw["vector"].get(article, 0.0), 3),
                "graph": round(raw["graph"].get(article, 0.0), 3),
                "bm25_share": round(parts["bm25"] / total, 3) if total else 0.0,
                "vector_share": round(parts["vector"] / total, 3) if total else 0.0,
                "graph_share": round(parts["graph"] / total, 3) if total else 0.0,
                "graph_hit": article in raw["graph"],
                "graph_path": graph_paths.get(article),
            })
        candidates.sort(key=lambda item: (-item["retrieval_score"], item["article"]))
        ranked = self.reranker.rerank(question, candidates[:pool_size])[:top_k]
        summary = {
            "bm25_articles": set(raw["bm25"]),
            "vector_articles": set(raw["vector"]),
            "graph_articles": set(raw["graph"]),
        }
        return ranked, summary
