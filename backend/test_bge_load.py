# -*- coding: utf-8 -*-
"""验证 bge-m3 / reranker 快照能被 semantic_index.py 离线加载。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.semantic_index import (
    _local_snapshot_path,
    _load_sentence_transformer,
    _load_cross_encoder,
)

for mid in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
    print(f"[snapshot] {mid} -> {_local_snapshot_path(mid)}")

print("[load] bge-m3 加载中（CPU 首次加载需 10-30 秒）...", flush=True)
m = _load_sentence_transformer("BAAI/bge-m3")
print(f"[ok] SentenceTransformer: {type(m).__name__}")

vecs = m.encode(["消防通道被杂物堵住了", "楼道堆放纸箱违反规定吗"], normalize_embeddings=True)
print(f"[ok] 编码输出形状: {vecs.shape}, 样例: {vecs[0][:3].round(4)}")

print("[load] reranker 加载中...", flush=True)
c = _load_cross_encoder("BAAI/bge-reranker-v2-m3")
score = c.predict([("消防通道被堵怎么处罚", "占用消防通道的，对单位处五千元以上五万元以下罚款")])
print(f"[ok] CrossEncoder 相关性得分: {score}")

print("ALL OK")
