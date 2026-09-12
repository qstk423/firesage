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

# Docker / 演示轻量模式：跳过 bge 下载与加载，直接 TF-IDF + 规则重排
FIRESAGE_LITE = os.getenv("FIRESAGE_LITE", "").strip() in ("1", "true", "True", "yes")

# HuggingFace 直连不稳定时走镜像（中国大陆环境）
if os.getenv("HF_ENDPOINT") is None and os.getenv("USE_HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _hf_cache_ready(model_id: str) -> bool:
    """本机 HF hub 是否已有该模型快照（有则优先离线加载，避免 403/弱网校验失败）。"""
    return _local_snapshot_path(model_id) is not None


def _local_snapshot_path(model_id: str):
    """返回可用的本地 snapshot 目录（含 config.json + 权重），否则 None。"""
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    dirname = "models--" + model_id.replace("/", "--")
    snap_root = os.path.join(hub, dirname, "snapshots")
    if not os.path.isdir(snap_root):
        return None
    candidates = []
    for name in os.listdir(snap_root):
        path = os.path.join(snap_root, name)
        if not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, "config.json")):
            continue
        has_weight = any(
            os.path.exists(os.path.join(path, f))
            for f in ("pytorch_model.bin", "model.safetensors", "model.safetensors.index.json")
        )
        if has_weight:
            candidates.append(path)
    if not candidates:
        return None
    # 选体积更大的完整快照（避免半下载目录）
    return max(candidates, key=lambda p: sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(p) for f in files
    ))


def _load_sentence_transformer(model_id: str):
    from sentence_transformers import SentenceTransformer
    local = _local_snapshot_path(model_id)
    if local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            return SentenceTransformer(local, local_files_only=True)
        except TypeError:
            return SentenceTransformer(local)
        except Exception as e:
            print(f"[SemanticVectorIndex] 本地快照加载失败（{e}），尝试模型 ID")
    prefer_offline = os.getenv("EMBED_OFFLINE", "1").strip() in ("1", "true", "True")
    if prefer_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return SentenceTransformer(model_id)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
    return SentenceTransformer(model_id)


def _load_cross_encoder(model_id: str):
    from sentence_transformers import CrossEncoder
    local = _local_snapshot_path(model_id)
    if local:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            return CrossEncoder(local, max_length=512, local_files_only=True)
        except TypeError:
            return CrossEncoder(local, max_length=512)
        except Exception as e:
            print(f"[CrossEncoderReranker] 本地快照加载失败（{e}），尝试模型 ID")
    prefer_offline = os.getenv("EMBED_OFFLINE", "1").strip() in ("1", "true", "True")
    if prefer_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            return CrossEncoder(model_id, max_length=512)
        except Exception:
            os.environ.pop("HF_HUB_OFFLINE", None)
    return CrossEncoder(model_id, max_length=512)


class _OnnxInt8CrossEncoder:
    """bge-reranker-v2-m3 的 ONNX INT8 CPU 推理（onnxruntime 直驱，无 optimum 依赖）。

    接口对齐 sentence_transformers.CrossEncoder.predict：num_labels==1 时
    返回 sigmoid(logits)——下游 score() 的双重 sigmoid 口径与 PyTorch 路径完全一致。
    产物由 scripts/export_reranker_onnx.py 生成（模型 + tokenizer + meta.json）。
    """

    def __init__(self, model_dir):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = os.path.join(model_dir, "model_int8.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"缺少 {onnx_path}（先运行 scripts/export_reranker_onnx.py）")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        threads = os.getenv("RERANK_ONNX_THREADS", "").strip()
        if threads.isdigit():
            opts.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.max_length = 512

    def predict(self, pairs):
        """pairs: [(question, text), ...] → sigmoid(logits)，与 ST CrossEncoder 同口径。"""
        import numpy as np

        enc = self.tokenizer(list(pairs), return_tensors="np", padding=True,
                             truncation=True, max_length=self.max_length)
        feeds = {
            "input_ids": np.asarray(enc["input_ids"], dtype="int64"),
            "attention_mask": np.asarray(enc["attention_mask"], dtype="int64"),
        }
        logits = np.asarray(self.session.run(["logits"], feeds)[0]).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits))


def _load_reranker():
    """按 RERANK_BACKEND 选择精排后端：onnx_int8（CPU 提速）或默认 PyTorch。"""
    backend = os.getenv("RERANK_BACKEND", "").strip().lower()
    if backend == "onnx_int8":
        onnx_dir = os.path.join(BASE_DIR, "models", "bge-reranker-v2-m3-onnx-int8")
        try:
            model = _OnnxInt8CrossEncoder(onnx_dir)
            print("[CrossEncoderReranker] ONNX INT8 后端已启用（bge-reranker-v2-m3, CPU）")
            return model
        except Exception as e:
            print(f"[CrossEncoderReranker] ONNX INT8 加载失败（{type(e).__name__}: {e}），"
                  f"回退 PyTorch CrossEncoder")
    return _load_cross_encoder(RERANK_MODEL)


def _corpus_fingerprint(chunks):
    raw = "|".join(f"{c['article']}:{c.get('text', '')}" for c in (chunks or []))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


class _Embedder:
    """bge-m3 编码器单例（进程内只加载一次）。"""

    _instance = None

    def __init__(self):
        self.model = None
        self.error = None
        if FIRESAGE_LITE:
            self.error = "FIRESAGE_LITE=1，跳过 bge-m3"
            return
        try:
            self.model = _load_sentence_transformer(EMBED_MODEL)
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
    """句级 bge-m3 向量 + FAISS；召回后按条款取 max 分（句找 → 条答）。"""

    name = "bge-m3 语义向量（句级）"

    def __init__(self, chunks):
        self.embedder = _Embedder.get()
        self.available = self.embedder.model is not None
        self.fallback = None
        self.unit_articles = []
        self.articles = []
        if not self.available:
            print(f"[SemanticVectorIndex] bge-m3 不可用（{self.embedder.error}），降级为 TF-IDF 字符向量")
            from .vector_index import TfidfVectorIndex
            self.fallback = TfidfVectorIndex(chunks)
            self.name = self.fallback.name
            return

        units = []
        corpus = []
        for chunk in chunks or []:
            article = chunk.get("article") or ""
            if not article:
                continue
            title = chunk.get("title", "") or ""
            text = chunk.get("text", "") or ""
            units.append({"id": chunk.get("id", ""), "article": article})
            corpus.append(f"{title}。{text}" if title else text)
        self.unit_articles = [u["article"] for u in units]
        self.articles = sorted(set(self.unit_articles))
        if not units:
            self.matrix = np.zeros((0, 0), dtype="float32")
            self.index = None
            return

        fp = _corpus_fingerprint(chunks)
        cache_vec = os.path.join(CACHE_DIR, f"embed_sent_{EMBED_MODEL.replace('/', '_')}_{fp}.npy")
        cache_meta = os.path.join(CACHE_DIR, f"units_sent_{fp}.json")
        if os.path.exists(cache_vec) and os.path.exists(cache_meta):
            import json
            with open(cache_meta, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("kind") == "sentence" and meta.get("units") == units:
                self.matrix = np.load(cache_vec)
                if self.matrix.shape[0] == len(units):
                    self._init_faiss()
                    return
        vectors = self.embedder.encode(corpus)
        self.matrix = np.asarray(vectors, dtype="float32")
        np.save(cache_vec, self.matrix)
        import json
        with open(cache_meta, "w", encoding="utf-8") as f:
            json.dump({"kind": "sentence", "units": units}, f, ensure_ascii=False)
        self._init_faiss()
        print(f"[SemanticVectorIndex] bge-m3 句级索引就绪：{len(units)} 句 / {len(self.articles)} 条款")

    def _init_faiss(self):
        import faiss
        if self.matrix is None or self.matrix.ndim < 2 or self.matrix.shape[0] == 0:
            self.index = None
            return
        dim = self.matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.matrix)

    def search(self, query, top_k=10):
        if self.fallback is not None:
            return self.fallback.search(query, top_k)
        if self.index is None or not self.unit_articles:
            return []
        # 多取句级候选，再聚合成条款
        pool = min(len(self.unit_articles), max(top_k * 6, 24))
        vector = np.asarray(self.embedder.encode([query]), dtype="float32")
        scores, ids = self.index.search(vector, pool)
        best = {}
        for i, s in zip(ids[0], scores[0]):
            if i < 0 or s <= 0:
                continue
            article = self.unit_articles[i]
            if s > best.get(article, 0.0):
                best[article] = float(s)
        ranked = sorted(best.items(), key=lambda x: -x[1])[:top_k]
        return ranked


class CrossEncoderReranker:
    """bge-reranker-v2-m3 语义精排；对整条（父节点）打分，配合句级召回。"""

    name = "CrossEncoder 语义精排"

    def __init__(self, chunks):
        self.model = None
        self.article_text = {}
        for chunk in chunks or []:
            article = chunk["article"]
            record = self.article_text.setdefault(
                article, {"title": chunk.get("title", ""), "parts": [], "parent": ""})
            record["parts"].append(chunk.get("text", ""))
            if chunk.get("parent_text") and not record["parent"]:
                record["parent"] = chunk["parent_text"]
        self.available = False
        if FIRESAGE_LITE:
            print("[CrossEncoderReranker] FIRESAGE_LITE=1，跳过精排模型")
            return
        try:
            self.model = _load_reranker()
            self.available = True
            if isinstance(self.model, _OnnxInt8CrossEncoder):
                self.name = "CrossEncoder 语义精排（ONNX INT8）"
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
                candidate["article"], {"title": "", "parts": [""], "parent": ""})
            body = record.get("parent") or "".join(record.get("parts") or [])
            text = (record.get("title", "") + "。" + body)[:480]
            pairs.append((question, text))
            keys.append(candidate["article"])
        raw = self.model.predict(pairs)
        # logits → sigmoid 归一到 0~1
        scores = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype="float64")))
        return {k: float(s) for k, s in zip(keys, scores)}
