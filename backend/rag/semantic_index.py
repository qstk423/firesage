# -*- coding: utf-8 -*-
"""中文语义向量索引（BAAI/bge-m3 + FAISS）与 CrossEncoder 精排（BAAI/bge-reranker-v2-m3）。

设计要点：
1. 首选加载本地 bge-m3；模型或依赖缺失时自动降级为 TF-IDF 字符向量，保证系统永不因模型问题不可用。
2. 语料向量按语义句版本缓存（.npy），语料未变不重复编码。
3. 精排模型对三路召回的候选做 question-article 交叉编码打分，
   与法规意图重排（LegalReranker）线性组合：语义精排为主，意图规则为辅。
"""
import hashlib
import os

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
CACHE_DIR = os.path.join(BASE_DIR, "data", ".semantic_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# HuggingFace 直连不稳定时走镜像（中国大陆环境）
if os.getenv("HF_ENDPOINT") is None and os.getenv("USE_HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _corpus_fingerprint(chunks):
    raw = "|".join(f"{c['article']}:{c.get('text', '')}" for c in (chunks or []))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


class _Embedder:
    """bge-m3 编码器单例（进程内只加载一次）。"""

    _instance = None

    def __init__(self):
        self.model = None
        self.error = None
        try:
            from sentence_transformers import SentenceTransformer
            try:
                self.model = SentenceTransformer(EMBED_MODEL)
            except Exception:
                # 本地已有缓存但在线校验失败（弱网/SSL 中断）→ 离线模式重试
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:  # 模型缺失/依赖缺失/加载失败
            self.error = str(e)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)


class SemanticVectorIndex:
    """按条款聚合的 bge-m3 语义向量 + FAISS 内积检索。"""

    name = "bge-m3 语义向量"

    def __init__(self, chunks):
        self.embedder = _Embedder.get()
        self.available = self.embedder.model is not None
        self.fallback = None
        if not self.available:
            print(f"[SemanticVectorIndex] bge-m3 不可用（{self.embedder.error}），降级为 TF-IDF 字符向量")
            from .vector_index import TfidfVectorIndex
            self.fallback = TfidfVectorIndex(chunks)
            self.name = self.fallback.name
            return

        # 按条款聚合文本
        article_parts = {}
        article_titles = {}
        for chunk in chunks or []:
            article_parts.setdefault(chunk["article"], []).append(chunk.get("text", ""))
            article_titles[chunk["article"]] = chunk.get("title", "")
        self.articles = sorted(article_parts)
        corpus = [article_titles[a] + "。" + "".join(article_parts[a]) for a in self.articles]

        # 缓存：语料指纹一致则直接复用
        fp = _corpus_fingerprint(chunks)
        cache_vec = os.path.join(CACHE_DIR, f"embed_{EMBED_MODEL.replace('/', '_')}_{fp}.npy")
        cache_articles = os.path.join(CACHE_DIR, f"articles_{fp}.json")
        if os.path.exists(cache_vec) and os.path.exists(cache_articles):
            import json
            with open(cache_articles, encoding="utf-8") as f:
                if json.load(f) == self.articles:
                    self.matrix = np.load(cache_vec)
                    self._init_faiss()
                    return
        vectors = self.embedder.encode(corpus)
        self.matrix = np.asarray(vectors, dtype="float32")
        np.save(cache_vec, self.matrix)
        import json
        with open(cache_articles, "w", encoding="utf-8") as f:
            json.dump(self.articles, f, ensure_ascii=False)
        self._init_faiss()
        print(f"[SemanticVectorIndex] bge-m3 索引就绪：{len(self.articles)} 条款")

    def _init_faiss(self):
        import faiss
        dim = self.matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.matrix)

    def search(self, query, top_k=10):
        if self.fallback is not None:
            return self.fallback.search(query, top_k)
        vector = np.asarray(self.embedder.encode([query]), dtype="float32")
        scores, ids = self.index.search(vector, min(top_k, len(self.articles)))
        return [(self.articles[i], float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]


class CrossEncoderReranker:
    """bge-reranker-v2-m3 语义精排；不可用时返回 None，由调用方保留规则重排结果。"""

    name = "CrossEncoder 语义精排"

    def __init__(self, chunks):
        self.model = None
        self.article_text = {}
        for chunk in chunks or []:
            record = self.article_text.setdefault(
                chunk["article"], {"title": chunk.get("title", ""), "parts": []})
            record["parts"].append(chunk.get("text", ""))
        self.available = False
        try:
            from sentence_transformers import CrossEncoder
            try:
                self.model = CrossEncoder(RERANK_MODEL, max_length=512)
            except Exception:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                self.model = CrossEncoder(RERANK_MODEL, max_length=512)
            self.available = True
        except Exception as e:
            print(f"[CrossEncoderReranker] bge-reranker-v2-m3 不可用（{e}），保留法规意图重排")

    def score(self, question, candidates):
        """对候选列表做交叉编码打分，返回 {article: sigmoid 分数}。"""
        if not self.available or not candidates:
            return {}
        pairs = []
        keys = []
        for candidate in candidates:
            record = self.article_text.get(
                candidate["article"], {"title": "", "parts": [""]})
            text = record["title"] + "。" + "".join(record["parts"])[:480]
            pairs.append((question, text))
            keys.append(candidate["article"])
        raw = self.model.predict(pairs)
        # logits → sigmoid 归一到 0~1
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype="float64")))
        return {k: float(s) for k, s in zip(keys, scores)}
