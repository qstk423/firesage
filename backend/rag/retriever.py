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
from .semantic_index import CrossEncoderReranker, SemanticVectorIndex

# RRF 权重与平滑常数。BM25 保留较高权重，图谱负责补充关系证据。
CHANNEL_WEIGHTS = {"bm25": 0.42, "vector": 0.33, "graph": 0.25}
# 向量模型降级为 TF-IDF 时：加大 BM25，降低弱语义/图谱通道噪声
CHANNEL_WEIGHTS_DEGRADED = {"bm25": 0.72, "vector": 0.10, "graph": 0.18}
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
    # Stage4：补强责任主体 / 定义条款 / 隐患整改 / 共用责任 / 演练
    "消防安全责任人": ["法定代表人", "主要负责人", "对本单位的消防安全工作全面负责"],
    "谁是单位的消防安全责任人": ["法定代表人或者非法人单位的主要负责人是单位的消防安全责任人"],
    "多少米": ["建筑高度大于", "用语的含义", "27米", "24米"],
    "高层住宅建筑和高层公共建筑": ["建筑高度大于27米", "建筑高度大于24米"],
    "共用": ["两个以上单位", "共用的疏散通道", "统一管理"],
    "多家公司": ["同一建筑物由两个以上单位", "共用的疏散通道"],
    "不能确保消防安全": ["不能确保消防安全", "停产停业整改", "危险部位停产停业"],
    "火灾隐患": ["火灾隐患", "整改", "当场改正"],
    "消防演练": ["消防演练", "灭火和应急疏散预案", "组织进行有针对性的消防演练"],
    "着火了往哪跑": ["消防演练", "灭火和应急疏散预案"],
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
        self.vector = SemanticVectorIndex(self.chunks)
        self.graph = GraphRetriever(graph)
        self.reranker = LegalReranker(self.chunks)
        self.cross_encoder = CrossEncoderReranker(self.chunks)
        self.degraded_vector = not getattr(self.vector, "available", True)
        self.channels = ["BM25", self.vector.name, "GraphRAG",
                         self.reranker.name, self.cross_encoder.name]

    def _channel_weights(self):
        return CHANNEL_WEIGHTS_DEGRADED if self.degraded_vector else CHANNEL_WEIGHTS

    def retrieve(self, question, top_k=5, mode="full"):
        """混合检索。

        mode:
          - bm25: 仅 BM25
          - bm25_vector: BM25 + 向量 RRF
          - hybrid: 三路 RRF，不做重排
          - full: 三路 RRF + 法规意图重排 + CrossEncoder（默认）
        """
        pool_size = max(12, top_k * 4)
        use_vector = mode in ("bm25_vector", "hybrid", "full")
        use_graph = mode in ("hybrid", "full")
        use_rerank = mode == "full"

        bm25_list = self.bm25.search(question, pool_size)
        vector_list = self.vector.search(" ".join(_expand(question)), pool_size) if use_vector else []
        graph_list = self.graph.search(question, pool_size) if use_graph else []

        base_weights = self._channel_weights()
        weights = {"bm25": base_weights["bm25"]}
        if use_vector:
            weights["vector"] = base_weights["vector"]
        if use_graph:
            weights["graph"] = base_weights["graph"]
        # 消融时按启用通道重归一化，避免总分虚低
        weight_sum = sum(weights.values()) or 1.0
        weights = {k: v / weight_sum for k, v in weights.items()}

        raw = {
            "bm25": {article: score for article, score in bm25_list},
            "vector": {article: score for article, score in vector_list},
            "graph": {article: score for article, score, _ in graph_list},
        }
        graph_paths = {article: path for article, _, path in graph_list}
        rrf_parts = {name: {} for name in weights}
        channel_lists = [("bm25", bm25_list)]
        if use_vector:
            channel_lists.append(("vector", vector_list))
        if use_graph:
            channel_lists.append(("graph", [(a, s) for a, s, _ in graph_list]))
        for name, ranked in channel_lists:
            for rank, (article, _) in enumerate(ranked, 1):
                rrf_parts[name][article] = weights[name] / (RRF_K + rank)

        articles = set().union(*(set(raw[n]) for n in weights))
        max_rrf = sum(weight / (RRF_K + 1) for weight in weights.values())
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
                "bm25_share": round(parts.get("bm25", 0.0) / total, 3) if total else 0.0,
                "vector_share": round(parts.get("vector", 0.0) / total, 3) if total else 0.0,
                "graph_share": round(parts.get("graph", 0.0) / total, 3) if total else 0.0,
                "graph_hit": article in raw["graph"],
                "graph_path": graph_paths.get(article),
            })
        candidates.sort(key=lambda item: (-item["retrieval_score"], item["article"]))
        if use_rerank:
            reranked = self.reranker.rerank(question, candidates[:pool_size])
            # CrossEncoder 语义精排：与法规意图重排线性组合（语义为主、规则为辅）
            ce_scores = self.cross_encoder.score(question, reranked)
            if ce_scores:
                for item in reranked:
                    ce = ce_scores.get(item["article"], 0.0)
                    item["cross_encoder"] = round(ce, 4)
                    item["rerank_score"] = round(0.55 * ce + 0.45 * item["rerank_score"], 4)
                    item["score"] = item["rerank_score"]
                reranked.sort(key=lambda item: (-item["rerank_score"], item["article"]))
        else:
            for item in candidates:
                item["rerank_score"] = item["retrieval_score"]
                item["score"] = item["retrieval_score"]
            reranked = candidates[:pool_size]
            ce_scores = None
        ranked = reranked[:top_k]
        summary = {
            "bm25_articles": set(raw["bm25"]),
            "vector_articles": set(raw["vector"]),
            "graph_articles": set(raw["graph"]),
            "cross_encoder_used": bool(ce_scores),
            "mode": mode,
        }
        return ranked, summary
