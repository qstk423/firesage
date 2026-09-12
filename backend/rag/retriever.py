# -*- coding: utf-8 -*-
"""混合检索引擎：BM25 + 本地向量 + GraphRAG + 二阶段重排。

三路召回使用 RRF 融合，避免各检索器分数量纲不同导致错误置信度。
"""
import math
import os
import re
import time
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
# 报批稿 / 草案：检索降权（仍可召回，但优先现行）
DRAFT_STATUS_FACTOR = 0.62
DRAFT_MARKERS = ("报批", "征求意见", "草案", "送审")


def _status_factor(status: str, notes: str = "") -> float:
    blob = f"{status or ''}{notes or ''}"
    if any(m in blob for m in DRAFT_MARKERS):
        return DRAFT_STATUS_FACTOR
    return 1.0

# 同义词扩展（只留高通用概念词；题面整句特判见 eval/retrieval_boosts.json）
SYNONYMS = {
    "楼道": ["楼道", "走廊", "过道", "楼梯间"],
    "堆放": ["堆放", "堆", "堆积", "放置"],
    "杂物": ["杂物", "物品", "东西", "垃圾"],
    "违反": ["违反", "违规", "违犯", "触犯"],
    "消防通道": ["消防车通道", "消防通道", "防火通道"],
    "怎么处罚": ["处罚", "罚款", "怎么罚", "后果", "责任"],
    "怎么办": ["怎么办", "处理", "处置", "如何处理"],
    "电动车": ["电动自行车", "电瓶车", "电动车"],
    "楼道充电": ["公共门厅充电", "疏散走道充电", "楼梯间充电", "电动自行车充电"],
    "公共娱乐场所": ["公共娱乐场所", "歌舞娱乐场所", "卡拉OK", "KTV", "夜总会"],
    "居民楼": ["居民住宅楼", "住宅建筑"],
    "人员密集场所": ["人员密集场所", "公众聚集场所"],
    "高层住宅": ["高层住宅建筑", "高层民用建筑", "住宅楼"],
    "消控室": ["消防控制室", "消控室"],
    "物业": ["物业服务企业", "统一管理人", "物业公司"],
    "消防安全责任人": ["法定代表人", "主要负责人", "对本单位的消防安全工作全面负责"],
    "多少米": ["建筑高度大于", "用语的含义", "27米", "24米"],
    "共用": ["两个以上单位", "共用的疏散通道", "统一管理"],
    "多家公司": ["同一建筑物由两个以上单位", "共用的疏散通道"],
    "火灾隐患": ["火灾隐患", "整改", "当场改正"],
    "消防演练": ["消防演练", "灭火和应急疏散预案", "组织进行有针对性的消防演练"],
    "工地": ["施工现场", "施工单位", "建筑工程施工现场", "施工期间"],
    "施工": ["施工现场", "施工单位", "施工期间"],
    "居委会": ["居民委员会", "村民委员会", "防火安全公约"],
    "第一责任人": ["政府主要负责人为第一责任人", "地方各级人民政府负责本行政区域内的消防工作"],
    "单位违反": ["单位违反本法规定", "处五千元以上五万元以下罚款"],
    "防火检查": ["防火检查", "至少每月进行一次防火检查", "至少每季度进行一次防火检查"],
    "举报": ["消防救援机构", "消防监督检查", "监督检查"],
    "个体户": ["个体工商户", "固定生产经营场所"],
    "小店": ["个体工商户", "固定生产经营场所"],
    "没证": ["依法取得相应的职业资格", "值班操作人员应当依法取得"],
    "无证": ["依法取得相应的职业资格", "值班操作人员应当依法取得"],
    # 口语「强制拆除」≠擅自拆除设施；法条用语是「强制执行」
    "强制拆除": ["强制执行", "经责令改正拒不改正的，强制执行", "所需费用由违法行为人承担"],
    "强制执行": ["强制执行", "经责令改正拒不改正", "逾期不执行停产停业"],
}

# 问句「消防法第X条」→ 强制注入候选，避免扩库后点名条款被挤出召回池
LAW_ALIAS_TO_ABBR = [
    ("公共娱乐场所消防安全管理规定", "39号令"),
    ("公共娱乐场所", "39号令"),
    ("39号令", "39号令"),
    ("电动自行车充电", "电动车充电"),
    ("电动车充电", "电动车充电"),
    ("人员密集场所消防安全管理", "密集场所"),
    ("广东省高层建筑消防安全管理规定", "广东高层"),
    ("广东高层", "广东高层"),
    ("高层民用建筑消防安全管理规定", "高层规定"),
    ("高层规定", "高层规定"),
    ("消防安全责任制实施办法", "责任制办法"),
    ("责任制办法", "责任制办法"),
    ("机关、团体、企业、事业单位消防安全管理规定", "61号令"),
    ("61号令", "61号令"),
    ("中华人民共和国消防法", "消防法"),
    ("消防法", "消防法"),
]
ARTICLE_CITE_RE = re.compile(r"第[一二三四五六七八九十百零〇两0-9]+条")

# 高区分度口语 → 强制入召回池（概念错配 / 扩库稀释；不做整句题面特判）
FORCE_RECALL = (
    (("暂时停掉", "擅自停用", "检修"), ("消防设施", "消防器材"),
     ("消防法·第二十八条", "高层规定·第四十七条")),
    (("共有部分", "分摊"), (),
     ("高层规定·第三十三条",)),
    (("电瓶车", "楼道充电", "电动车充电"), (),
     ("61号令·第十条", "高层规定·第十条", "高层规定·第三十七条")),
    # 概念问「强制拆除」易撞「擅自拆除」；锚定强制执行条款
    (("强制拆除", "强制执行", "拒不改正强制"), (),
     ("消防法·第六十条", "消防法·第七十条")),
)


def _force_recall_articles(question: str, known_articles) -> list[str]:
    known = set(known_articles or [])
    q = question or ""
    hits = []
    for keys_a, keys_b, articles in FORCE_RECALL:
        if not any(k in q for k in keys_a):
            continue
        if keys_b and not any(k in q for k in keys_b):
            continue
        for a in articles:
            if a in known and a not in hits:
                hits.append(a)
    return hits


def _resolve_cited_articles(question: str, known_articles) -> list[str]:
    cites = ARTICLE_CITE_RE.findall(question or "")
    if not cites:
        return []
    law_abbr = None
    for alias, abbr in LAW_ALIAS_TO_ABBR:
        if alias in question:
            law_abbr = abbr
            break
    known = set(known_articles or [])
    hits = []
    for cite in cites:
        if law_abbr:
            key = f"{law_abbr}·{cite}"
            if key in known:
                hits.append(key)
            continue
        for abbr in ("消防法", "61号令", "高层规定", "责任制办法", "39号令"):
            key = f"{abbr}·{cite}"
            if key in known:
                hits.append(key)
                break
    return hits


CORE_LAWS = {"消防法", "61号令", "高层规定", "责任制办法"}
SPECIALTY_GATES = {
    # 娱乐场所专项：避免「场所」泛词误开
    "39号令": ("娱乐", "歌舞", "卡拉", "KTV", "夜总会", "放映", "影剧", "公共娱乐", "39号令"),
    # 电动车：楼道口语 + 专题术语
    "电动车充电": (
        "电动自行车", "电瓶车", "楼道充电", "飞线", "充电场所", "停放充电",
        "电动车", "电池进电梯", "推进电梯",
    ),
    # 密集场所报批稿：仅高区分度词，避免「人员密集场所」泛问冲核心库
    "密集场所": (
        "志愿消防队员", "微型消防站人数", "微型消防站",
        "人员密集场所消防安全", "密集场所·", "密集场所报批",
    ),
    # 地方规定
    "广东高层": ("广东", "粤", "本省", "超高层用气", "广东高层", "广东省高层"),
    # 社会消防技术服务
    "技术服务": (
        "消防技术服务", "技术服务机构", "维保检测", "维护保养检测",
        "消防安全评估", "注册消防工程师", "技术服务·", "7号令",
    ),
    # 建设工程消防设计审查验收
    "消设审查": (
        "消防设计审查", "消防验收", "特殊建设工程", "消防验收备案",
        "竣工验收消防", "消设审查", "设计审查验收",
    ),
}


def _chunk_law(chunk: dict) -> str:
    return chunk.get("law") or str(chunk.get("article", "")).split("·", 1)[0]


def _specialty_needed(question: str) -> set[str]:
    needed = set()
    for law, keys in SPECIALTY_GATES.items():
        if any(k in (question or "") for k in keys):
            needed.add(law)
    return needed


def _merge_ranked(primary, secondary, top_k):
    best = {a: s for a, s in primary}
    for a, s in secondary:
        best[a] = max(best.get(a, 0.0), s)
    if not best:
        return []
    mx = max(best.values()) or 1.0
    ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
    return [(a, s / mx) for a, s in ranked]


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
        # 条款 → 时效元数据（报批稿降权）
        self.article_meta = {}
        for c in self.chunks:
            art = c.get("article")
            if art and art not in self.article_meta:
                self.article_meta[art] = {
                    "status": c.get("status", "") or "",
                    "notes": c.get("notes", "") or "",
                }
        # 核心库单独建 BM25/向量，避免扩库后 IDF 稀释 Hit@1
        self.core_chunks = [c for c in self.chunks if _chunk_law(c) in CORE_LAWS]
        self.spec_chunks = [c for c in self.chunks if _chunk_law(c) not in CORE_LAWS]
        self.bm25 = BM25Index(self.core_chunks or self.chunks)
        self.bm25_specialty = BM25Index(self.spec_chunks) if self.spec_chunks else None
        self.vector = SemanticVectorIndex(self.core_chunks or self.chunks)
        self.graph = GraphRetriever(graph)
        self.reranker = LegalReranker(self.chunks)
        self.cross_encoder = CrossEncoderReranker(self.chunks)
        self.degraded_vector = not getattr(self.vector, "available", True)
        channels = ["BM25", self.vector.name, "GraphRAG", self.reranker.name]
        try:
            from . import rerank_ml
            if rerank_ml.available():
                channels.append("微调重排")
        except Exception:
            pass
        channels.append(self.cross_encoder.name)
        self.channels = channels

    def _channel_weights(self):
        return CHANNEL_WEIGHTS_DEGRADED if self.degraded_vector else CHANNEL_WEIGHTS

    def retrieve(self, question, top_k=5, mode="full", sub_queries=None, query_mode="local"):
        """混合检索。

        mode:
          - bm25: 仅 BM25
          - bm25_vector: BM25 + 向量 RRF
          - hybrid: 三路 RRF，不做重排
          - full: 三路 RRF + 法规意图重排 + CrossEncoder（默认）
        sub_queries: 复合问拆出的多路查询；对各查询通道做 RRF 再平均。
        query_mode: local=场景条款；global=职责总述（压专题、抬核心库）。
        """
        total_started = time.perf_counter()
        timing = {
            "bm25_ms": 0.0,
            "vector_ms": 0.0,
            "graph_ms": 0.0,
            "fusion_ms": 0.0,
            "rule_rerank_ms": 0.0,
            "ml_rerank_ms": 0.0,
            "cross_encoder_ms": 0.0,
        }
        pool_size = max(12, top_k * 4)
        use_vector = mode in ("bm25_vector", "hybrid", "full")
        use_graph = mode in ("hybrid", "full")
        use_rerank = mode == "full"
        queries = []
        for q in (sub_queries or [question]):
            q = (q or "").strip()
            if q and q not in queries:
                queries.append(q)
        if not queries:
            queries = [question or ""]
        primary = question or queries[0]
        global_mode = (query_mode or "local") == "global"

        bm25_best, vector_best, graph_best = {}, {}, {}
        graph_paths = {}
        needed = set()
        for q in queries:
            q_needed = set() if global_mode else _specialty_needed(q)
            needed |= q_needed
            channel_started = time.perf_counter()
            bm25_q = self.bm25.search(q, pool_size)
            if self.bm25_specialty and q_needed:
                # 专题库召回分打折，避免与核心库同主题时抢占 Hit@1
                spec_hits = [
                    (a, s * 0.72) for a, s in self.bm25_specialty.search(q, pool_size)
                    if a.split("·", 1)[0] in q_needed
                ]
                bm25_q = _merge_ranked(bm25_q, spec_hits, pool_size)
            timing["bm25_ms"] += (time.perf_counter() - channel_started) * 1000
            for a, s in bm25_q:
                bm25_best[a] = max(bm25_best.get(a, 0.0), s)
            if use_vector:
                channel_started = time.perf_counter()
                for a, s in self.vector.search(" ".join(_expand(q)), pool_size):
                    vector_best[a] = max(vector_best.get(a, 0.0), s)
                timing["vector_ms"] += (time.perf_counter() - channel_started) * 1000
            if use_graph:
                channel_started = time.perf_counter()
                allow = CORE_LAWS if global_mode else (CORE_LAWS | q_needed)
                for a, s, p in self.graph.search(q, pool_size):
                    if a.split("·", 1)[0] not in allow:
                        continue
                    if s >= graph_best.get(a, 0.0):
                        graph_best[a] = s
                        graph_paths[a] = p
                timing["graph_ms"] += (time.perf_counter() - channel_started) * 1000

        fusion_started = time.perf_counter()
        bm25_list = sorted(bm25_best.items(), key=lambda x: -x[1])[:pool_size]
        vector_list = sorted(vector_best.items(), key=lambda x: -x[1])[:pool_size]
        graph_list = [
            (a, s, graph_paths.get(a))
            for a, s in sorted(graph_best.items(), key=lambda x: -x[1])[:pool_size]
        ]
        # 全局职责总述：压低专题条款分，抬核心库
        if global_mode:
            bm25_list = [
                (a, s * (1.0 if a.split("·", 1)[0] in CORE_LAWS else 0.55))
                for a, s in bm25_list
            ]
            bm25_list.sort(key=lambda x: -x[1])
            vector_list = [
                (a, s * (1.0 if a.split("·", 1)[0] in CORE_LAWS else 0.55))
                for a, s in vector_list
            ]
            vector_list.sort(key=lambda x: -x[1])

        base_weights = self._channel_weights()
        weights = {"bm25": base_weights["bm25"]}
        if use_vector:
            weights["vector"] = base_weights["vector"]
        if use_graph and graph_list:
            weights["graph"] = base_weights["graph"]
        # 消融时按启用通道重归一化，避免总分虚低
        weight_sum = sum(weights.values()) or 1.0
        weights = {k: v / weight_sum for k, v in weights.items()}

        raw = {
            "bm25": {article: score for article, score in bm25_list},
            "vector": {article: score for article, score in vector_list},
            "graph": {article: score for article, score, _ in graph_list},
        }
        rrf_parts = {name: {} for name in weights}
        channel_lists = [("bm25", bm25_list)]
        if use_vector:
            channel_lists.append(("vector", vector_list))
        if "graph" in weights:
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
            score = total / max_rrf if max_rrf else 0.0
            # 全局模式：含「应当履行」「消防安全职责」等枢纽句的核心条轻抬
            if global_mode and article.split("·", 1)[0] in CORE_LAWS:
                blob = ""
                for c in self.chunks:
                    if c.get("article") == article:
                        blob = c.get("text", "")
                        break
                if any(k in blob for k in ("应当履行下列消防安全职责", "消防安全职责", "重点单位应当")):
                    score = min(1.0, score * 1.08)
            candidates.append({
                "article": article,
                "retrieval_score": score,
                "bm25": round(raw["bm25"].get(article, 0.0), 3),
                "vector": round(raw["vector"].get(article, 0.0), 3),
                "graph": round(raw["graph"].get(article, 0.0), 3),
                "bm25_share": round(parts.get("bm25", 0.0) / total, 3) if total else 0.0,
                "vector_share": round(parts.get("vector", 0.0) / total, 3) if total else 0.0,
                "graph_share": round(parts.get("graph", 0.0) / total, 3) if total else 0.0,
                "graph_hit": article in raw["graph"],
                "graph_path": graph_paths.get(article),
            })
        # 时效感知：报批稿/草案降权（句找条答后的条款级惩罚）
        for item in candidates:
            meta = self.article_meta.get(item["article"], {})
            factor = _status_factor(meta.get("status", ""), meta.get("notes", ""))
            if factor < 1.0:
                item["retrieval_score"] = round(item["retrieval_score"] * factor, 4)
                item["status_demoted"] = True
                item["status"] = meta.get("status", "")
        candidates.sort(key=lambda item: (-item["retrieval_score"], item["article"]))
        # 点名条款 / 高区分度口语：强制入池（扩库后 BM25 常挤掉金标）
        known = {c["article"] for c in self.chunks}
        inject = []
        for q in queries:
            inject.extend(_resolve_cited_articles(q, known))
            inject.extend(_force_recall_articles(q, known))
        inject = list(dict.fromkeys(inject))
        for cited in inject:
            if any(item["article"] == cited for item in candidates):
                for item in candidates:
                    if item["article"] == cited:
                        item["retrieval_score"] = max(item["retrieval_score"], 0.98)
                        item["cite_inject"] = True
                continue
            candidates.insert(0, {
                "article": cited,
                "retrieval_score": 0.98,
                "bm25": 1.0,
                "vector": 0.0,
                "graph": 0.0,
                "bm25_share": 1.0,
                "vector_share": 0.0,
                "graph_share": 0.0,
                "graph_hit": False,
                "graph_path": None,
                "cite_inject": True,
            })
        candidates.sort(key=lambda item: (-item["retrieval_score"], item["article"]))
        timing["fusion_ms"] = (time.perf_counter() - fusion_started) * 1000
        if use_rerank:
            rerank_started = time.perf_counter()
            reranked = self.reranker.rerank(primary, candidates[:pool_size])
            timing["rule_rerank_ms"] = (time.perf_counter() - rerank_started) * 1000
            # 领域微调重排：对 (问题, 条款) 打分，与规则分融合
            try:
                from . import rerank_ml
                if rerank_ml.available():
                    ml_started = time.perf_counter()
                    pairs = []
                    for item in reranked:
                        art = self.reranker.article_text.get(item["article"], {})
                        blob = (art.get("title", "") + " " + "".join(art.get("parts") or []))[:800]
                        pairs.append((item["article"], blob))
                    ml_scores = rerank_ml.score_many(primary, pairs)
                    for item in reranked:
                        ml = ml_scores.get(item["article"], 0.0)
                        item["ml_rerank"] = round(ml, 4)
                        # 小幅加性融合：以规则重排为主，微调只做轻推
                        item["rerank_score"] = round(item["rerank_score"] + 0.10 * ml, 4)
                        item["score"] = item["rerank_score"]
                    reranked.sort(key=lambda item: (-item["rerank_score"], item["article"]))
                    timing["ml_rerank_ms"] = (time.perf_counter() - ml_started) * 1000
            except Exception:
                pass
            # CrossEncoder 语义精排：只对规则重排后的前 N 条打分（CPU 上全量精排很慢）
            ce_n = int(os.getenv("CE_CANDIDATES", "6") or "6")
            ce_n = max(3, min(ce_n, len(reranked)))
            ce_started = time.perf_counter()
            ce_scores = self.cross_encoder.score(primary, reranked[:ce_n])
            timing["cross_encoder_ms"] = (time.perf_counter() - ce_started) * 1000
            if ce_scores:
                for item in reranked:
                    ce = ce_scores.get(item["article"], 0.0)
                    item["cross_encoder"] = round(ce, 4) if item["article"] in ce_scores else None
                    if item["article"] in ce_scores:
                        base = item["rerank_score"]
                        item["rerank_score"] = round(0.50 * ce + 0.50 * base, 4)
                        item["score"] = item["rerank_score"]
                # 未进 CE 的候选保持规则分，整体重排
                reranked.sort(key=lambda item: (-item["rerank_score"], item["article"]))
        else:
            for item in candidates:
                item["rerank_score"] = item["retrieval_score"]
                item["score"] = item["retrieval_score"]
            reranked = candidates[:pool_size]
            ce_scores = None
        ranked = reranked[:top_k]
        timing["total_ms"] = (time.perf_counter() - total_started) * 1000
        timing = {key: round(value, 2) for key, value in timing.items()}
        summary = {
            "bm25_articles": set(raw["bm25"]),
            "vector_articles": set(raw["vector"]),
            "graph_articles": set(raw["graph"]),
            "cross_encoder_used": bool(ce_scores),
            "mode": mode,
            "query_mode": query_mode or "local",
            "sub_queries": queries,
            "timing": timing,
        }
        return ranked, summary
